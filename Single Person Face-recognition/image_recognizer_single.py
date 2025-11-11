import os
import cv2
import pickle
import csv
import face_recognition
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

# ---------------- Config ----------------
ENCODINGS_PATH = "encodings.pkl"
LOG_DIR = "recognition_logs"
LOG_CSV = os.path.join(LOG_DIR, "recognition_log.csv")
THRESHOLD = 0.50           # face_distance threshold (lower = stricter)
THUMB_SIZE = (150, 150)
MAX_DISPLAY_SCALE = 0.8
# ----------------------------------------

def load_encodings():
    """Load encodings with backward compatibility."""
    if not os.path.exists(ENCODINGS_PATH):
        return [], [], []
    try:
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
        enc = data.get("encodings", [])
        names = data.get("names", [])
        cats = data.get("categories", [])
        if len(cats) != len(names):
            cats = ["Normal"] * len(names)
        return enc, names, cats
    except Exception:
        return [], [], []

def save_encodings(encodings, names, categories):
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names, "categories": categories}, f)

def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Include a column for full-frame annotated image
            writer.writerow(["Timestamp","Name","Category","ImageFile","FullImageFile"])

def log_recognition(name, category, face_bgr, image_basename, full_image_filename=None):
    """Save cropped face and append CSV row (face_bgr should be BGR numpy array)."""
    ensure_log_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe = name.replace(" ", "_")
    filename = f"{safe}_{category}_{ts}.jpg"
    outpath = os.path.join(LOG_DIR, filename)
    try:
        cv2.imwrite(outpath, face_bgr)
    except Exception as e:
        print("Warning: could not save face image:", e)
        outpath = ""
    with open(LOG_CSV, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write extended row including full-frame annotated image filename if provided
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            name,
            category,
            filename,
            full_image_filename or ""
        ])

def choose_image_file():
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title="Select an image", filetypes=[("Images","*.jpg *.jpeg *.png *.bmp")])
    root.destroy()
    return path

def choose_folder():
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes("-topmost", True)
    path = filedialog.askdirectory(title="Select folder with images")
    root.destroy()
    return path

def preview_register_dialog(face_rgb):
    """
    Show a modal dialog with face preview and Name + Category entries.
    face_rgb: numpy array in RGB
    Returns: (name, category) or (None, None) if skipped, or ('CANCEL', None) if user chose cancel all.
    """
    # Build modal window
    top = tk.Toplevel()
    top.title("Register face?")
    top.attributes('-topmost', True)
    # Convert face_rgb to PIL and resize for preview
    pil = Image.fromarray(face_rgb)
    pil_thumb = pil.resize((250,250))
    imgtk = ImageTk.PhotoImage(pil_thumb)
    lbl = tk.Label(top, image=imgtk)
    lbl.image = imgtk
    lbl.grid(row=0, column=0, columnspan=3, padx=8, pady=8)

    tk.Label(top, text="Name:").grid(row=1,column=0, sticky='e')
    entry_name = tk.Entry(top, width=30)
    entry_name.grid(row=1,column=1, columnspan=2, sticky='w', padx=4, pady=4)
    tk.Label(top, text="Category:").grid(row=2,column=0, sticky='e')
    var_cat = tk.StringVar(value="Normal")
    opt = tk.OptionMenu(top, var_cat, "Normal","VIP","Blacklisted")
    opt.grid(row=2,column=1, sticky='w', padx=4, pady=4)

    # Buttons: Register, Skip, Cancel All
    res = {"name": None, "cat": None, "action": None}
    def on_register():
        res["name"] = entry_name.get().strip()
        res["cat"] = var_cat.get()
        res["action"] = "register"
        top.destroy()
    def on_skip():
        res["action"] = "skip"
        top.destroy()
    def on_cancel_all():
        res["action"] = "cancel"
        top.destroy()

    btn_register = tk.Button(top, text="Register", command=on_register, width=10)
    btn_skip = tk.Button(top, text="Skip", command=on_skip, width=10)
    btn_cancel = tk.Button(top, text="Cancel All", command=on_cancel_all, width=10)
    btn_register.grid(row=3,column=0,padx=6,pady=8)
    btn_skip.grid(row=3,column=1,padx=6,pady=8)
    btn_cancel.grid(row=3,column=2,padx=6,pady=8)

    # Make modal
    top.grab_set()
    top.focus_force()
    top.wait_window()

    if res["action"] == "register":
        if not res["name"]:
            messagebox.showwarning("No name", "Empty name — skipping registration.")
            return None, None
        return res["name"], res["cat"]
    if res["action"] == "skip":
        return None, None
    return "CANCEL", None

def process_single_image(image_path, known_encodings, known_names, known_categories):
    """
    Process one image: detect faces, match, ask to register unknowns, show result,
    log each detection. Returns possibly-updated encodings/names/categories.
    """
    original_rgb = face_recognition.load_image_file(image_path)   # RGB
    img_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)      # for drawing
    basename = os.path.basename(image_path)
    # Create a session-level annotated frame filename so CSV rows can reference it consistently
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    frame_filename = f"frame_{os.path.splitext(basename)[0]}_{session_ts}.jpg"

    # detect faces
    face_locations = face_recognition.face_locations(original_rgb)
    face_encodings = face_recognition.face_encodings(original_rgb, face_locations)

    if not face_locations:
        messagebox.showinfo("No faces", "No faces found in this image.")
        return known_encodings, known_names, known_categories

    face_names = []
    face_cats = []

    cancel_all = False
    for (top, right, bottom, left), f_encoding in zip(face_locations, face_encodings):
        name = "Unknown"; cat = "Unknown"
        # if DB exists, compute distances
        if known_encodings:
            dists = face_recognition.face_distance(known_encodings, f_encoding)
            best_idx = int(dists.argmin())
            if dists[best_idx] <= THRESHOLD:
                name = known_names[best_idx]; cat = known_categories[best_idx]
            else:
                # unknown -> show preview and ask register
                # prepare a small RGB crop for preview
                crop_rgb = original_rgb[max(0,top-10):min(original_rgb.shape[0],bottom+10),
                                        max(0,left-10):min(original_rgb.shape[1],right+10)]
                if crop_rgb.size == 0:
                    # fallback: use whole face bbox converted
                    crop_rgb = original_rgb[top:bottom, left:right]
                # show preview register dialog
                reg_name, reg_cat = preview_register_dialog(crop_rgb)
                if reg_name == "CANCEL":
                    cancel_all = True
                    break
                if reg_name:
                    known_encodings.append(f_encoding)
                    known_names.append(reg_name)
                    known_categories.append(reg_cat if reg_cat else "Normal")
                    save_encodings(known_encodings, known_names, known_categories)
                    name = reg_name; cat = reg_cat if reg_cat else "Normal"
                else:
                    # skipped: keep Unknown
                    name = "Unknown"; cat = "Unknown"
        else:
            # no DB -> ask to register
            crop_rgb = original_rgb[max(0,top-10):min(original_rgb.shape[0],bottom+10),
                                    max(0,left-10):min(original_rgb.shape[1],right+10)]
            reg_name, reg_cat = preview_register_dialog(crop_rgb)
            if reg_name == "CANCEL":
                cancel_all = True
                break
            if reg_name:
                known_encodings.append(f_encoding)
                known_names.append(reg_name)
                known_categories.append(reg_cat if reg_cat else "Normal")
                save_encodings(known_encodings, known_names, known_categories)
                name = reg_name; cat = reg_cat if reg_cat else "Normal"
            else:
                name = "Unknown"; cat = "Unknown"

        face_names.append(name)
        face_cats.append(cat)

        # crop BGR face for logging
        crop_bgr = img_bgr[max(0,top-10):min(img_bgr.shape[0],bottom+10),
                           max(0,left-10):min(img_bgr.shape[1],right+10)]
        if crop_bgr.size != 0:
            # Pass the annotated full-frame filename to CSV rows; saved later after overlays
            log_recognition(name, cat, crop_bgr, basename, full_image_filename=frame_filename)
        else:
            # still log entry but no image
            log_recognition(name, cat, np.zeros((10,10,3), dtype=np.uint8), basename, full_image_filename=frame_filename)

        if cancel_all:
            break

    if cancel_all:
        messagebox.showinfo("Cancelled", "Operation cancelled by user.")
    # draw & display results + thumbnails
    draw_labelled_faces_and_show(original_rgb, img_bgr, face_locations, face_names, face_cats, basename, frame_filename)

    return known_encodings, known_names, known_categories

def draw_labelled_faces_and_show(original_rgb, img_bgr, face_locations, face_names, face_cats, image_basename, frame_filename=None):
    """Draw styled overlays and thumbnail cards; keep recognition unchanged. Also save annotated full frame."""
    thumbs = []

    def draw_corner_box(image, left, top, right, bottom, color, thickness=2, length=12):
        # Corner lines for a modern look
        cv2.line(image, (left, top), (left+length, top), color, thickness)
        cv2.line(image, (left, top), (left, top+length), color, thickness)
        cv2.line(image, (right, top), (right-length, top), color, thickness)
        cv2.line(image, (right, top), (right, top+length), color, thickness)
        cv2.line(image, (left, bottom), (left+length, bottom), color, thickness)
        cv2.line(image, (left, bottom), (left, bottom-length), color, thickness)
        cv2.line(image, (right, bottom), (right-length, bottom), color, thickness)
        cv2.line(image, (right, bottom), (right, bottom-length), color, thickness)

    for (top, right, bottom, left), name, cat in zip(face_locations, face_names, face_cats):
        # clamp coords
        top = max(0, top); left = max(0,left)
        bottom = min(original_rgb.shape[0], bottom); right = min(original_rgb.shape[1], right)

        # Determine color palette
        color = (80, 220, 120) if cat=="Normal" else (60, 120, 255) if cat=="Blacklisted" else (40, 200, 200)
        face_w = max(1, right-left)
        thick = max(3, int(face_w/14))

        # Draw corner-styled box
        draw_corner_box(img_bgr, left, top, right, bottom, color, thickness=thick, length=max(12, int(face_w*0.08)))

        # Label background (semi-transparent)
        label = f"{name} • {cat}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        lb_h = th + 12
        lb_w = tw + 16
        lb_x = left
        lb_y = max(0, top - lb_h - 6)
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (lb_x, lb_y), (lb_x+lb_w, lb_y+lb_h), color, -1)
        cv2.addWeighted(overlay, 0.25, img_bgr, 0.75, 0, img_bgr)
        cv2.putText(img_bgr, label, (lb_x+8, lb_y+lb_h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        # Thumbnail card with accent bar and cleaner typography
        crop_rgb = original_rgb[max(0, top-10):min(original_rgb.shape[0], bottom+10),
                                max(0, left-10):min(original_rgb.shape[1], right+10)]
        if crop_rgb.size != 0:
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            thumb = cv2.resize(crop_bgr, THUMB_SIZE)

            card_w = THUMB_SIZE[0] + 240
            card_h = THUMB_SIZE[1]
            card = np.zeros((card_h, card_w, 3), dtype=np.uint8)
            # gradient background
            for i in range(card_w):
                alpha = i / max(1, card_w-1)
                card[:, i] = (20 + int(20*alpha), 20 + int(25*alpha), 25 + int(30*alpha))
            # place thumb
            card[0:THUMB_SIZE[1], 0:THUMB_SIZE[0]] = thumb
            # accent bar
            cv2.rectangle(card, (THUMB_SIZE[0], 0), (THUMB_SIZE[0]+6, card_h), color, -1)
            # text block
            x0 = THUMB_SIZE[0] + 16
            cv2.putText(card, name, (x0, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240,240,240), 2, cv2.LINE_AA)
            cv2.putText(card, f"Category: {cat}", (x0, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210,210,210), 2, cv2.LINE_AA)
            thumbs.append(card)

    # If a target annotated filename is provided, save the full original-size annotated frame
    if frame_filename:
        ensure_log_dir()
        outpath = os.path.join(LOG_DIR, frame_filename)
        try:
            cv2.imwrite(outpath, img_bgr)
        except Exception as e:
            print("Warning: could not save annotated frame:", e)

    # compute screen scale
    root = tk.Tk(); root.withdraw()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    scale = min((sw / img_bgr.shape[1]) * MAX_DISPLAY_SCALE, (sh / img_bgr.shape[0]) * MAX_DISPLAY_SCALE, 1.0)
    full = cv2.resize(img_bgr, (int(img_bgr.shape[1]*scale), int(img_bgr.shape[0]*scale)), interpolation=cv2.INTER_AREA)

    if thumbs:
        strip = cv2.hconcat(thumbs)
        if strip.shape[1] < full.shape[1]:
            pad = full.shape[1] - strip.shape[1]
            strip = cv2.copyMakeBorder(strip, 0,0,0,pad, cv2.BORDER_CONSTANT, value=(50,50,50))
        combined = cv2.vconcat([full, strip])
    else:
        combined = full

    window = "Recognition Results"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(window, 50, 50)
    cv2.imshow(window, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def manage_db_console(enc, names, cats):
    """Console-based manage DB: list, delete by index, update category."""
    if not names:
        print("DB empty.")
        return enc, names, cats
    print("\nRegistered faces:")
    for i, (n,c) in enumerate(zip(names,cats)):
        print(f"[{i}] {n} ({c})")
    cmd = input("Type 'd' to delete by index, 'u' to update category, or Enter to skip: ").strip().lower()
    if cmd == 'd':
        idx = input("Index to delete: ").strip()
        if idx.isdigit():
            idx = int(idx)
            if 0 <= idx < len(names):
                print(f"Deleting {names[idx]}")
                enc.pop(idx); names.pop(idx); cats.pop(idx)
                save_encodings(enc, names, cats)
    elif cmd == 'u':
        idx = input("Index to update: ").strip()
        if idx.isdigit():
            idx = int(idx)
            if 0 <= idx < len(names):
                newc = input("Enter new category (Normal/VIP/Blacklisted): ").strip().title()
                if newc:
                    cats[idx] = newc
                    save_encodings(enc, names, cats)
    return enc, names, cats

def batch_process_folder(folder, known_encodings, known_names, known_categories):
    """Process all images in folder (jpg/png). Prompts for unknown registration per image/face."""
    exts = ('.jpg','.jpeg','.png','.bmp')
    files = [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith(exts)]
    files.sort()
    for f in files:
        print("Processing:", f)
        known_encodings, known_names, known_categories = process_single_image(f, known_encodings, known_names, known_categories)
    return known_encodings, known_names, known_categories

def print_menu():
    print("\n=== Image Recognizer Menu ===")
    print("1) Recognize / Register a single image")
    print("2) Recognize / Register all images in a folder (batch)")
    print("3) Manage DB (delete/update)")
    print("4) Settings (threshold)")
    print("5) Exit")

def main():
    global THRESHOLD
    
    known_encodings, known_names, known_categories = load_encodings()
    ensure_log_dir()

    while True:
        print_menu()
        ch = input("Choose: ").strip()
        if ch == '1':
            path = choose_image_file()
            if path:
                known_encodings, known_names, known_categories = process_single_image(path, known_encodings, known_names, known_categories)
        elif ch == '2':
            folder = choose_folder()
            if folder:
                known_encodings, known_names, known_categories = batch_process_folder(folder, known_encodings, known_names, known_categories)
        elif ch == '3':
            known_encodings, known_names, known_categories = manage_db_console(known_encodings, known_names, known_categories)
        elif ch == '4':
            val = input(f"Current threshold = {THRESHOLD}. Enter new threshold (0.3 - 0.8) or press Enter to keep: ").strip()
            if val:
                try:
                    nv = float(val)
                    if 0.3 <= nv <= 0.8:
                        
                        THRESHOLD = nv
                        print("Threshold updated.")
                    else:
                        print("Out of range.")
                except Exception:
                    print("Invalid value.")
        elif ch == '5':
            print("Exiting.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
