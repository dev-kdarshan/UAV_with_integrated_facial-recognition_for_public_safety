import os
import csv
from datetime import datetime
from flask import Flask, render_template, send_from_directory, request, url_for, redirect, session, Response
import uuid
import cv2
import numpy as np
import face_recognition
import db as dbm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

# Paths to recognition logs
SINGLE_LOG_DIR = os.path.join(PROJECT_ROOT, 'Single Person Face-recognition', 'recognition_logs')
MULTI_LOG_DIR = os.path.join(PROJECT_ROOT, 'Multiple people Face-recognition', 'recognition_logs')
SINGLE_ENCODINGS = os.path.join(PROJECT_ROOT, 'Single Person Face-recognition', 'encodings.pkl')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'), static_folder=os.path.join(BASE_DIR, 'static'))
dbm.init_db()
app.secret_key = os.environ.get('APP_SECRET', 'dev-secret-key')


# --- Auth helpers ---
def current_user():
    return {
        'id': session.get('user_id'),
        'username': session.get('username'),
        'role': session.get('role')
    } if session.get('user_id') else None


def require_role(*roles):
    def decorator(fn):
        from functools import wraps
        @wraps(fn)
        def wrapper(*args, **kwargs):
            user = current_user()
            if not user:
                return redirect(url_for('login', next=request.path))
            if roles and user['role'] not in roles:
                return redirect(url_for('index'))
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def read_csv_events(log_dir, source_label):
    events = []
    if not os.path.isdir(log_dir):
        return events

    # Prefer a known file name, else read all .csv
    candidate_files = []
    preferred = os.path.join(log_dir, 'recognition_log.csv')
    if os.path.exists(preferred):
        candidate_files.append(preferred)
    else:
        candidate_files.extend([
            os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.lower().endswith('.csv')
        ])

    for csv_path in candidate_files:
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts = row.get('Timestamp') or row.get('timestamp')
                    name = row.get('Name') or row.get('name')
                    category = row.get('Category') or row.get('category')
                    # Prefer full annotated image when available
                    full_image_file = row.get('FullImageFile') or row.get('full_image') or row.get('full_image_file')
                    # Handle legacy CSV headers where extra fields are appended without header
                    if not full_image_file and None in row and isinstance(row[None], list) and row[None]:
                        tail = row[None][-1]
                        if isinstance(tail, str) and tail.lower().endswith(('.jpg', '.jpeg', '.png')):
                            full_image_file = tail
                    image_file = full_image_file or row.get('ImageFile') or row.get('image') or row.get('image_path')

                    # Build image route only if the file exists
                    image_path = None
                    if image_file:
                        img_full = os.path.join(log_dir, image_file)
                        if os.path.exists(img_full):
                            image_path = f"/images/{source_label}/{image_file}"

                    # Parse timestamp to sort; fallback to original string
                    ts_parsed = None
                    if ts:
                        # Try common formats: 'YYYY-MM-DD HH:MM:SS' and ISO 'YYYY-MM-DDTHH:MM:SS(.ms)[Z]'
                        try:
                            ts_parsed = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                        except Exception:
                            try:
                                clean = ts.replace('Z', '')
                                ts_parsed = datetime.fromisoformat(clean)
                            except Exception:
                                ts_parsed = None

                    # Map category to status used in UI
                    raw_cat = (category or '-').strip()
                    status = 'Pending' if raw_cat.lower() in ['unknown', '-', 'uncategorized'] else ('Failed' if raw_cat.lower() in ['blacklisted', 'blocked'] else 'Passed')

                    # Synthetic score purely for visualization (does not affect recognition)
                    score = 90 if status == 'Passed' else (30 if status == 'Failed' else 50)

                    events.append({
                        'timestamp': ts,
                        'timestamp_parsed': ts_parsed,
                        'date': ts_parsed.date().isoformat() if ts_parsed else None,
                        'name': name or 'Unknown',
                        'category': raw_cat,
                        'status': status,
                        'score': score,
                        'image_path': image_path,
                        'source': source_label,
                        'id': f"{source_label}:{image_file}" if image_file else f"{source_label}:{ts}",
                    })
        except Exception:
            # If a CSV is malformed, skip it
            continue

    return events


@app.route('/')
def index():
    # Filters from query string
    q_name = (request.args.get('name') or '').strip().lower()
    q_category = (request.args.get('category') or '').strip().lower()
    q_date = (request.args.get('date') or '').strip()

    # Gather events from CSV and DB
    events = []
    # CSV legacy events
    events.extend(read_csv_events(SINGLE_LOG_DIR, 'single'))
    events.extend(read_csv_events(MULTI_LOG_DIR, 'multiple'))
    # DB detections
    dets = dbm.list_detections(limit=500)
    for d in dets:
        # Map DB detection to dashboard event structure
        status = 'Passed' if (d.get('status') == 'Known') else ('Failed' if (d.get('status') == 'Unknown') else 'Pending')
        try:
            ts_parsed = datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S')
        except Exception:
            try:
                ts_parsed = datetime.fromisoformat(d['timestamp'])
            except Exception:
                ts_parsed = None
        events.append({
            'timestamp': d['timestamp'],
            'timestamp_parsed': ts_parsed,
            'date': ts_parsed.date().isoformat() if ts_parsed else None,
            'name': d.get('name') or 'Unknown',
            'category': d.get('status') or 'Unknown',
            'status': status,
            'score': int(d.get('confidence') or 0),
            'image_path': d.get('image_path') or None,
            'source': 'db',
            'id': f"db:{d['id']}",
        })

    # Sort by timestamp desc when available
    events.sort(key=lambda e: e.get('timestamp_parsed') or datetime.min, reverse=True)

    # Apply filters
    def match(e):
        ok_name = True if not q_name else (e['name'] and q_name in e['name'].lower())
        ok_cat = True if not q_category else (e['category'] and q_category in e['category'].lower())
        ok_date = True if not q_date else (e['date'] == q_date)
        return ok_name and ok_cat and ok_date

    filtered = [e for e in events if match(e)]

    # Metrics
    total = len(filtered)
    passed = sum(1 for e in filtered if e['status'] == 'Passed')
    failed = sum(1 for e in filtered if e['status'] == 'Failed')
    pending = sum(1 for e in filtered if e['status'] == 'Pending')
    pass_percentage = int(100 * passed / max(1, (passed + failed))) if (passed + failed) > 0 else 0
    avg_score = int(sum(e['score'] for e in filtered) / max(1, total)) if total > 0 else 0

    # Progress overview (visual only)
    progress = {
        'completed': int(100 * passed / max(1, total)) if total > 0 else 0,
        'pending': int(100 * pending / max(1, total)) if total > 0 else 0,
        'in_progress': int(100 * failed / max(1, total)) if total > 0 else 0,
    }

    # Trend data per date
    trend = {}
    for e in filtered:
        if e['date']:
            trend[e['date']] = trend.get(e['date'], 0) + 1
    trend_labels = sorted(trend.keys())
    trend_values = [trend[d] for d in trend_labels]

    # Recent table
    recent = filtered[:20]

    return render_template(
        'index.html',
        events=filtered[:500],
        total=total,
        pass_percentage=pass_percentage,
        avg_score=avg_score,
        pending=pending,
        trend_labels=trend_labels,
        trend_values=trend_values,
        progress=progress,
        recent=recent,
        active='dashboard',
    )


@app.route('/images/<source>/<filename>')
def serve_image(source, filename):
    if source == 'single':
        directory = SINGLE_LOG_DIR
    elif source == 'multiple':
        directory = MULTI_LOG_DIR
    else:
        directory = SINGLE_LOG_DIR
    return send_from_directory(directory, filename)


@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route('/event/<source>/<path:filename>')
def event_detail(source, filename):
    # CSV or DB event detail
    if source == 'db':
        try:
            det_id = int(filename)
        except Exception:
            det_id = None
        d = dbm.get_detection(det_id) if det_id else None
        if not d:
            return redirect(url_for('results_list'))
        # Map to event shape
        try:
            ts_parsed = datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S')
        except Exception:
            try:
                ts_parsed = datetime.fromisoformat(d['timestamp'])
            except Exception:
                ts_parsed = None
        status = 'Passed' if (d.get('status') == 'Known') else 'Failed'
        event = {
            'timestamp': d['timestamp'],
            'timestamp_parsed': ts_parsed,
            'date': ts_parsed.date().isoformat() if ts_parsed else None,
            'name': d.get('name') or 'Unknown',
            'category': d.get('status') or 'Unknown',
            'status': status,
            'score': int(d.get('confidence') or 0),
            'image_path': d.get('image_path') or None,
            'source': 'db',
            'id': f"db:{d['id']}",
        }
        # Breakdown using DB detections for same name
        dets = dbm.list_detections({'name': event['name']}, limit=200)
        person_events = []
        breakdown = {'Passed': 0, 'Pending': 0, 'Failed': 0}
        for dd in dets:
            s = 'Passed' if dd.get('status') == 'Known' else 'Failed'
            breakdown[s] = breakdown.get(s, 0) + 1
            person_events.append({
                'timestamp': dd['timestamp'],
                'name': dd['name'],
                'category': dd['status'],
                'status': s,
                'image_path': dd['image_path'],
                'source': 'db',
                'id': f"db:{dd['id']}",
            })
        return render_template('result.html', event=event, breakdown=breakdown, person_events=person_events[:50], active='results')

    # Legacy CSV event detail
    events = []
    events.extend(read_csv_events(SINGLE_LOG_DIR, 'single'))
    events.extend(read_csv_events(MULTI_LOG_DIR, 'multiple'))
    target_id = f"{source}:{filename}"
    event = next((e for e in events if e['id'] == target_id), None)
    if not event:
        # Fallback: find by source and name
        event = next((e for e in events if e['source']==source and e.get('image_path') and filename in e['image_path']), None)

    person_events = [e for e in events if event and e['name'] == event['name']]
    breakdown = {'Passed': 0, 'Pending': 0, 'Failed': 0}
    for pe in person_events:
        breakdown[pe['status']] = breakdown.get(pe['status'], 0) + 1

    return render_template('result.html', event=event, breakdown=breakdown, person_events=person_events[:50], active='results')


@app.route('/results')
def results_list():
    # Filters
    q_name = (request.args.get('name') or '').strip()
    q_date = (request.args.get('date') or '').strip()
    try:
        q_min_conf = float(request.args.get('min_conf')) if request.args.get('min_conf') else None
    except Exception:
        q_min_conf = None

    detections = dbm.list_detections({
        'name': q_name,
        'date': q_date,
        'min_conf': q_min_conf
    }, limit=300)
    return render_template('results_list.html', detections=detections, active='results')


@app.route('/reports')
def reports():
    # Basic summary using DB logs
    detections = dbm.list_detections(limit=500)
    total = len(detections)
    known = sum(1 for d in detections if d['status'] == 'Known')
    unknown = sum(1 for d in detections if d['status'] == 'Unknown')
    avg_conf = round(sum(d.get('confidence') or 0 for d in detections) / max(1, total), 1) if total else 0
    return render_template('reports.html', active='reports', total=total, known=known, unknown=unknown, avg_conf=avg_conf)


@app.route('/reports/download')
def reports_download():
    fmt = (request.args.get('format') or 'csv').lower()
    detections = dbm.list_detections(limit=10000)
    if fmt == 'csv':
        from io import StringIO
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['id','timestamp','face_id','name','confidence','status','image_path','location','spoof_score','source','verified','false_positive'])
        for d in detections:
            writer.writerow([d['id'], d['timestamp'], d['face_id'], d['name'], d['confidence'], d['status'], d['image_path'], d['location'], d['spoof_score'], d['source'], d['verified'], d['false_positive']])
        resp = Response(output.getvalue(), mimetype='text/csv')
        resp.headers['Content-Disposition'] = 'attachment; filename="detection_report.csv"'
        return resp
    # Fallback: simple HTML export
    return render_template('reports_export.html', detections=detections)


@app.route('/users')
@require_role('admin', 'operator')
def users():
    faces = dbm.get_faces()
    return render_template('users.html', faces=faces, active='users', user=current_user())


@app.route('/users/<int:face_id>')
@require_role('admin', 'operator')
def user_profile(face_id):
    face = dbm.get_face(face_id)
    return render_template('user_profile.html', face=face, active='users', user=current_user())


@app.route('/users/<int:face_id>/edit', methods=['GET', 'POST'])
@require_role('admin')
def user_edit(face_id):
    face = dbm.get_face(face_id)
    if not face:
        return redirect(url_for('users'))
    if request.method == 'POST':
        dbm.update_face(face_id,
                        request.form.get('name') or face['name'],
                        request.form.get('person_id') or face['person_id'],
                        request.form.get('department') or face['department'],
                        request.form.get('category') or face['category'])
        return redirect(url_for('user_profile', face_id=face_id))
    return render_template('user_edit.html', face=face, active='users', user=current_user())


@app.route('/users/<int:face_id>/delete')
@require_role('admin')
def user_delete(face_id):
    dbm.delete_face(face_id)
    return redirect(url_for('users'))


@app.route('/settings', methods=['GET', 'POST'])
@require_role('admin')
def settings():
    if request.method == 'POST':
        dbm.set_setting('recognition_threshold', request.form.get('recognition_threshold') or '0.50')
        dbm.set_setting('camera_source', request.form.get('camera_source') or '0')
        dbm.set_setting('storage_dir', request.form.get('storage_dir') or SINGLE_LOG_DIR)
        dbm.set_setting('auto_start', '1' if request.form.get('auto_start') else '0')
        return redirect(url_for('settings'))
    return render_template('settings.html',
                           threshold=dbm.get_setting('recognition_threshold', '0.50'),
                           camera_source=dbm.get_setting('camera_source', '0'),
                           storage_dir=dbm.get_setting('storage_dir', SINGLE_LOG_DIR),
                           auto_start=dbm.get_setting('auto_start', '0') == '1',
                           active='settings', user=current_user())


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html', active='login', user=current_user())
    username = (request.form.get('username') or '').strip()
    password = (request.form.get('password') or '').strip()
    user = dbm.get_user_by_username(username)
    if not user:
        return render_template('login.html', error='Invalid credentials', active='login')
    import hashlib
    if user['password_hash'] != hashlib.sha256(password.encode('utf-8')).hexdigest():
        return render_template('login.html', error='Invalid credentials', active='login')
    session['user_id'] = user['id']
    session['username'] = user['username']
    session['role'] = user['role']
    nxt = request.args.get('next')
    return redirect(nxt or url_for('index'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.context_processor
def inject_user():
    return {'current_user': current_user()}


# --- Surveillance Stream ---
def _open_camera(src_str):
    try:
        if src_str.isdigit():
            return cv2.VideoCapture(int(src_str))
        return cv2.VideoCapture(src_str)
    except Exception:
        return None


def _annotate_frame(bgr, threshold=0.50):
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)
        known_enc, known_names, known_cats = load_encodings_unified()
        for enc, (top, right, bottom, left) in zip(encodings, locations):
            name = 'Unknown'
            category = 'Uncategorized'
            conf = 0.0
            if known_enc:
                dists = face_recognition.face_distance(known_enc, enc)
                best = int(np.argmin(dists))
                best_dist = float(dists[best])
                if best_dist <= threshold:
                    name = known_names[best]
                    category = known_cats[best]
                conf = max(0.0, min(100.0, (1.0 - min(best_dist, 0.6) / 0.6) * 100))
            face_bgr = bgr[top:bottom, left:right]
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F).var()
            suspicious = lap < 50.0
            color = (80, 220, 120) if name != 'Unknown' else ((255, 180, 0) if suspicious else (255, 80, 80))
            label = f"{name} • {category} • {conf:.0f}%"
            if suspicious:
                label += " • Suspicious"
            cv2.rectangle(bgr, (left, top), (right, bottom), color, 2)
            cv2.putText(bgr, label, (left, max(0, top-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    except Exception:
        pass
    return bgr


@app.route('/surveillance')
@require_role('admin', 'operator')
def surveillance():
    return render_template('surveillance.html', active='surveillance', user=current_user())


@app.route('/stream')
@require_role('admin', 'operator')
def stream():
    src = dbm.get_setting('camera_source', '0')
    cap = _open_camera(src)
    threshold = float(dbm.get_setting('recognition_threshold', '0.50'))
    if not cap or not cap.isOpened():
        # Generate a placeholder image
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(img, 'Camera unavailable', (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        _, buf = cv2.imencode('.jpg', img)
        data = buf.tobytes()
        return Response(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n",
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def gen():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = _annotate_frame(frame, threshold=threshold)
            _, buf = cv2.imencode('.jpg', frame)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------------- Web Recognition ---------------- #
@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'GET':
        try:
            th = float(dbm.get_setting('recognition_threshold', '0.50'))
        except Exception:
            th = 0.50
        return render_template('recognize.html', step='upload', threshold=th, active='recognize')

    # Handle upload and detection
    file = request.files.get('image')
    if not file:
        return render_template('recognize.html', step='upload', error='Please select an image.', active='recognize')

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
        return render_template('recognize.html', step='upload', error='Unsupported file type.', active='recognize')

    uid = uuid.uuid4().hex
    filename = f"{uid}{ext}"
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)

    # Load and detect faces
    # Threshold for recognition (from form or settings)
    try:
        threshold = float(request.form.get('threshold')) if request.form.get('threshold') else float(dbm.get_setting('recognition_threshold', '0.50'))
    except Exception:
        threshold = 0.50
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    known_enc, known_names, known_cats = load_encodings_unified()
    results = []
    for enc, (top, right, bottom, left) in zip(encodings, locations):
        name = 'Unknown'
        category = 'Uncategorized'
        if known_enc:
            dists = face_recognition.face_distance(known_enc, enc)
            best = int(np.argmin(dists))
            if dists[best] <= threshold:
                name = known_names[best]
                category = known_cats[best]
        # Prepare a small crop for UI
        crop = bgr[top:bottom, left:right]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        suspicious = lap < 50.0
        # Persist crop to uploads for display
        crop_name = f"{uid}_crop_{left}_{top}.jpg"
        cv2.imwrite(os.path.join(UPLOAD_DIR, crop_name), crop)
        results.append({
            'name': name,
            'category': category,
            'top': top,
            'right': right,
            'bottom': bottom,
            'left': left,
            'crop_path': f"/uploads/{crop_name}",
            'suspicious': suspicious
        })

    # Draw overlays for preview
    preview = bgr.copy()
    for r in results:
        color = (80, 220, 120) if r['name'] != 'Unknown' else (255, 80, 80)
        cv2.rectangle(preview, (r['left'], r['top']), (r['right'], r['bottom']), color, 2)
        label = f"{r['name']} • {r['category']}"
        if r.get('suspicious'):
            label += " • Suspicious"
        cv2.putText(preview, label, (r['left'], max(0, r['top']-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    preview_name = f"{uid}_preview.jpg"
    cv2.imwrite(os.path.join(UPLOAD_DIR, preview_name), preview)

    return render_template('recognize.html', step='review', upload_file=filename, preview_path=f"/uploads/{preview_name}", faces=results, threshold=threshold, active='recognize')


@app.route('/recognize/register', methods=['POST'])
def recognize_register():
    image_file = request.form.get('upload_file')
    if not image_file:
        return redirect(url_for('recognize'))
    path = os.path.join(UPLOAD_DIR, image_file)
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Collect registrations from form
    # Threshold for matching (from form or settings)
    try:
        threshold = float(request.form.get('threshold')) if request.form.get('threshold') else float(dbm.get_setting('recognition_threshold', '0.50'))
    except Exception:
        threshold = 0.50
    known_enc, known_names, known_cats = load_encodings_unified()
    registrations = []
    # Expect multiple entries: fields like name_N, category_N, left_N, top_N, right_N, bottom_N
    # We'll scan indexes from form keys
    indices = set()
    for k in request.form.keys():
        if '_' in k:
            try:
                idx = int(k.split('_')[-1])
                indices.add(idx)
            except Exception:
                pass
    for idx in sorted(indices):
        nm = (request.form.get(f'name_{idx}') or '').strip()
        dept = (request.form.get(f'dept_{idx}') or '').strip()
        pid = (request.form.get(f'pid_{idx}') or '').strip()
        ct = (request.form.get(f'category_{idx}') or '').strip() or 'Normal'
        try:
            left = int(request.form.get(f'left_{idx}'))
            top = int(request.form.get(f'top_{idx}'))
            right = int(request.form.get(f'right_{idx}'))
            bottom = int(request.form.get(f'bottom_{idx}'))
        except Exception:
            continue
        if nm:
            # Compute encoding for the face bbox and append
            face_location = (top, right, bottom, left)
            enc = face_recognition.face_encodings(rgb, [face_location])
            if enc:
                known_enc.append(enc[0])
                known_names.append(nm)
                known_cats.append(ct)
                registrations.append({'name': nm, 'category': ct, 'bbox': (top, right, bottom, left), 'dept': dept, 'pid': pid})

    # Save encodings
    if registrations:
        save_encodings_unified(known_enc, known_names, known_cats)
        # Save to DB with encrypted encoding bytes and store crop image paths
        for reg in registrations:
            (top, right, bottom, left) = reg['bbox']
            crop = bgr[top:bottom, left:right]
            crop_name = f"face_{uuid.uuid4().hex}.jpg"
            cv2.imwrite(os.path.join(SINGLE_LOG_DIR, crop_name), crop)
            # Serialize encoding bytes
            enc_bytes = known_enc[-1].tobytes() if known_enc else b''
            dbm.add_face(reg['name'], reg.get('pid') or '', reg.get('dept') or '', reg.get('category') or 'Normal', enc_bytes, f"/images/single/{crop_name}")

    # Draw final annotated frame and save to single log dir
    annotated = bgr.copy()
    # First, match all faces for logging
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)
    for enc, (top, right, bottom, left) in zip(encodings, locations):
        name = 'Unknown'
        category = 'Uncategorized'
        if known_enc:
            dists = face_recognition.face_distance(known_enc, enc)
            best = int(np.argmin(dists))
            if dists[best] <= threshold:
                name = known_names[best]
                category = known_cats[best]
        color = (80, 220, 120) if name != 'Unknown' else (255, 80, 80)
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
        label = f"{name} • {category}"
        cv2.putText(annotated, label, (left, max(0, top-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    full_name = f"frame_{uuid.uuid4().hex}.jpg"
    ensure_single_log()
    try:
        cv2.imwrite(os.path.join(SINGLE_LOG_DIR, full_name), annotated)
    except Exception:
        full_name = ''

    # Log each face to CSV and DB
    for enc, (top, right, bottom, left) in zip(encodings, locations):
        name = 'Unknown'
        category = 'Uncategorized'
        confidence = 0.0
        if known_enc:
            dists = face_recognition.face_distance(known_enc, enc)
            best = int(np.argmin(dists))
            best_dist = float(dists[best])
            if best_dist <= threshold:
                name = known_names[best]
                category = known_cats[best]
            # Map distance to a rough confidence [0,100]
            confidence = max(0.0, min(100.0, (1.0 - min(best_dist, 0.6) / 0.6) * 100))
        status = 'Known' if name != 'Unknown' else 'Unknown'
        face_bgr = bgr[top:bottom, left:right]
        log_event_single(name, category, face_bgr, full_name)
        # Simple anti-spoofing heuristic (lower is better): blur score
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        spoof_score = float(50.0 - min(50.0, lap / 2.0))
        # Persist to DB
        dbm.log_detection(
            face_id=None,
            name=name,
            confidence=confidence,
            status=status,
            image_path=f"/images/single/{full_name}" if full_name else '',
            location='',
            spoof_score=spoof_score,
            source='web'
        )

    # Show result page with link back
    return render_template('recognize.html', step='done', full_image=f"/images/single/{full_name}", active='recognize')


@app.route('/recognize/capture')
def recognize_capture():
    """Capture a single frame from the configured camera and run recognition."""
    try:
        threshold = float(dbm.get_setting('recognition_threshold', '0.50'))
    except Exception:
        threshold = 0.50
    src = dbm.get_setting('camera_source', '0')
    cap = _open_camera(str(src))
    if not cap or not cap.isOpened():
        return render_template('recognize.html', step='upload', error='Camera unavailable', threshold=threshold, active='recognize')
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return render_template('recognize.html', step='upload', error='Failed to capture frame', threshold=threshold, active='recognize')

    uid = uuid.uuid4().hex
    filename = f"{uid}.jpg"
    path = os.path.join(UPLOAD_DIR, filename)
    cv2.imwrite(path, frame)

    bgr = frame
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)
    known_enc, known_names, known_cats = load_encodings_unified()
    results = []
    for enc, (top, right, bottom, left) in zip(encodings, locations):
        name = 'Unknown'
        category = 'Uncategorized'
        if known_enc:
            dists = face_recognition.face_distance(known_enc, enc)
            best = int(np.argmin(dists))
            if dists[best] <= threshold:
                name = known_names[best]
                category = known_cats[best]
        crop = bgr[top:bottom, left:right]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        suspicious = lap < 50.0
        crop_name = f"{uid}_crop_{left}_{top}.jpg"
        cv2.imwrite(os.path.join(UPLOAD_DIR, crop_name), crop)
        results.append({'name': name, 'category': category, 'top': top, 'right': right, 'bottom': bottom, 'left': left, 'crop_path': f"/uploads/{crop_name}", 'suspicious': suspicious})

    preview = bgr.copy()
    for r in results:
        color = (80, 220, 120) if r['name'] != 'Unknown' else (255, 80, 80)
        cv2.rectangle(preview, (r['left'], r['top']), (r['right'], r['bottom']), color, 2)
        label = f"{r['name']} • {r['category']}"
        if r.get('suspicious'):
            label += " • Suspicious"
        cv2.putText(preview, label, (r['left'], max(0, r['top']-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    preview_name = f"{uid}_preview.jpg"
    cv2.imwrite(os.path.join(UPLOAD_DIR, preview_name), preview)

    return render_template('recognize.html', step='review', upload_file=filename, preview_path=f"/uploads/{preview_name}", faces=results, threshold=threshold, active='recognize')


@app.route('/results/verify/<int:det_id>')
def results_verify(det_id):
    dbm.mark_verified(det_id, 1)
    return redirect(url_for('results_list'))


@app.route('/results/false/<int:det_id>')
def results_false(det_id):
    dbm.mark_false_positive(det_id, 1)
    return redirect(url_for('results_list'))


def load_encodings_unified():
    enc, names, cats = [], [], []
    if os.path.exists(SINGLE_ENCODINGS):
        try:
            import pickle
            with open(SINGLE_ENCODINGS, 'rb') as f:
                data = pickle.load(f)
            # Handle dict-based format
            if isinstance(data, dict):
                enc = data.get('encodings', [])
                names = data.get('names', [])
                cats = data.get('categories', []) or ['Normal'] * len(names)
            else:
                # Tuple format (enc, names, categories)
                enc, names, cats = data
                if len(cats) != len(names):
                    cats = ['Normal'] * len(names)
        except Exception:
            pass
    return enc, names, cats


def save_encodings_unified(enc, names, cats):
    import pickle
    # Persist in dict format for compatibility with single-person script
    try:
        with open(SINGLE_ENCODINGS, 'wb') as f:
            pickle.dump({'encodings': enc, 'names': names, 'categories': cats}, f)
    except Exception:
        pass


def ensure_single_log():
    os.makedirs(SINGLE_LOG_DIR, exist_ok=True)
    csv_path = os.path.join(SINGLE_LOG_DIR, 'recognition_log.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Name', 'Category', 'ImageFile', 'FullImageFile'])
    return csv_path


def log_event_single(name, category, face_bgr, full_image_file):
    ensure_single_log()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    safe = (name or 'Unknown').replace(' ', '_')
    face_filename = f"{safe}_{category}_{ts}.jpg"
    try:
        cv2.imwrite(os.path.join(SINGLE_LOG_DIR, face_filename), face_bgr)
    except Exception:
        face_filename = ''
    with open(os.path.join(SINGLE_LOG_DIR, 'recognition_log.csv'), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            name,
            category,
            face_filename,
            full_image_file or ''
        ])


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)