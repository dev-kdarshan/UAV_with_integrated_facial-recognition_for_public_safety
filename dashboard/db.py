import os
import sqlite3
import base64
import hashlib
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'db.sqlite')
KEY_PATH = os.path.join(BASE_DIR, 'secret.key')


def _get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS registered_faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        person_id TEXT,
        department TEXT,
        category TEXT,
        encoding BLOB,
        image_path TEXT,
        date TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS detection_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        face_id INTEGER,
        name TEXT,
        confidence REAL,
        status TEXT,
        image_path TEXT,
        location TEXT,
        spoof_score REAL,
        source TEXT,
        verified INTEGER DEFAULT 0,
        false_positive INTEGER DEFAULT 0
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password_hash TEXT,
        role TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """)
    conn.commit()
    conn.close()
    ensure_default_admin()


# --- Simple encryption helpers ---
def _load_key():
    if os.path.exists(KEY_PATH):
        with open(KEY_PATH, 'rb') as f:
            return f.read()
    # derive a random-ish key if not present
    seed = hashlib.sha256(os.urandom(32)).digest()
    with open(KEY_PATH, 'wb') as f:
        f.write(seed)
    return seed


def encrypt_blob(data_bytes):
    # Fallback XOR-based obfuscation using derived key (not strong crypto)
    key = _load_key()
    out = bytearray()
    for i, b in enumerate(data_bytes):
        out.append(b ^ key[i % len(key)])
    return bytes(out)


def decrypt_blob(enc_bytes):
    key = _load_key()
    out = bytearray()
    for i, b in enumerate(enc_bytes):
        out.append(b ^ key[i % len(key)])
    return bytes(out)


# --- Faces ---
def add_face(name, person_id, department, category, encoding_bytes, image_path):
    enc = encrypt_blob(encoding_bytes)
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO registered_faces (name, person_id, department, category, encoding, image_path, date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (name, person_id, department, category, enc, image_path, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()


def get_faces():
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, name, person_id, department, category, image_path, date FROM registered_faces ORDER BY date DESC")
    rows = cur.fetchall()
    conn.close()
    faces = []
    for r in rows:
        faces.append({
            'id': r[0], 'name': r[1], 'person_id': r[2], 'department': r[3], 'category': r[4], 'image_path': r[5], 'date': r[6]
        })
    return faces


# --- Detection Logs ---
def log_detection(face_id, name, confidence, status, image_path, location, spoof_score, source):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO detection_logs (timestamp, face_id, name, confidence, status, image_path, location, spoof_score, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), face_id, name, confidence, status, image_path, location, spoof_score, source))
    conn.commit()
    conn.close()


def list_detections(filters=None, limit=200):
    filters = filters or {}
    conn = _get_conn()
    cur = conn.cursor()
    q = "SELECT id, timestamp, face_id, name, confidence, status, image_path, location, spoof_score, source, verified, false_positive FROM detection_logs"
    where = []
    args = []
    if filters.get('name'):
        where.append('LOWER(name) LIKE ?')
        args.append('%' + filters['name'].lower() + '%')
    if filters.get('date'):
        where.append('timestamp LIKE ?')
        args.append(filters['date'] + '%')
    if filters.get('min_conf') is not None:
        where.append('confidence >= ?')
        args.append(filters['min_conf'])
    if where:
        q += ' WHERE ' + ' AND '.join(where)
    q += ' ORDER BY timestamp DESC LIMIT ?'
    args.append(limit)
    cur.execute(q, args)
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({
            'id': r[0], 'timestamp': r[1], 'face_id': r[2], 'name': r[3], 'confidence': r[4], 'status': r[5],
            'image_path': r[6], 'location': r[7], 'spoof_score': r[8], 'source': r[9], 'verified': r[10], 'false_positive': r[11]
        })
    return out


def get_detection(det_id: int):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, timestamp, face_id, name, confidence, status, image_path, location, spoof_score, source, verified, false_positive FROM detection_logs WHERE id=?", (det_id,))
    r = cur.fetchone()
    conn.close()
    if not r:
        return None
    return {
        'id': r[0], 'timestamp': r[1], 'face_id': r[2], 'name': r[3], 'confidence': r[4], 'status': r[5],
        'image_path': r[6], 'location': r[7], 'spoof_score': r[8], 'source': r[9], 'verified': r[10], 'false_positive': r[11]
    }


def mark_verified(det_id, value=1):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute('UPDATE detection_logs SET verified=? WHERE id=?', (value, det_id))
    conn.commit()
    conn.close()


def mark_false_positive(det_id, value=1):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute('UPDATE detection_logs SET false_positive=? WHERE id=?', (value, det_id))
    conn.commit()
    conn.close()


# --- Settings ---
def set_setting(key, value):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)', (key, value))
    conn.commit()
    conn.close()


def get_setting(key, default=None):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute('SELECT value FROM settings WHERE key=?', (key,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else default


# --- Users (auth) ---
def _hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode('utf-8')).hexdigest()


def ensure_default_admin():
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute('SELECT COUNT(1) FROM users')
    count = cur.fetchone()[0]
    if (count or 0) == 0:
        cur.execute('INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)', (
            'admin', _hash_password('admin123'), 'admin'
        ))
        conn.commit()
    conn.close()


def get_user_by_username(username: str):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute('SELECT id, username, password_hash, role FROM users WHERE username=?', (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {'id': row[0], 'username': row[1], 'password_hash': row[2], 'role': row[3]}


def add_user(username: str, password: str, role: str = 'operator'):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute('INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)', (username, _hash_password(password), role))
    conn.commit()
    conn.close()


# --- Faces management ---
def get_face(face_id: int):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute('SELECT id, name, person_id, department, category, image_path, date FROM registered_faces WHERE id=?', (face_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {'id': row[0], 'name': row[1], 'person_id': row[2], 'department': row[3], 'category': row[4], 'image_path': row[5], 'date': row[6]}


def update_face(face_id: int, name: str, person_id: str, department: str, category: str):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute('UPDATE registered_faces SET name=?, person_id=?, department=?, category=? WHERE id=?', (name, person_id, department, category, face_id))
    conn.commit()
    conn.close()


def delete_face(face_id: int):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute('DELETE FROM registered_faces WHERE id=?', (face_id,))
    conn.commit()
    conn.close()