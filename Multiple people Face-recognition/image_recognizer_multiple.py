import face_recognition
import cv2
import numpy as np
import pickle
import os
import csv
import datetime
from tkinter import Tk, simpledialog, messagebox, filedialog, Toplevel, Label, Button, OptionMenu, StringVar, Entry
from PIL import Image, ImageTk
from PIL import Image, ImageDraw, ImageFont


THRESHOLD = 0.45
ENCODINGS_FILE = "encodings.pkl"
LOG_DIR = "recognition_logs"
LOG_CSV = os.path.join(LOG_DIR, "recognition_log.csv")
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------- Utility ---------------- #
def save_encodings(encodings, names, categories):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump((encodings, names, categories), f)

def load_encodings():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return [], [], []

def crop_face(image, location):
    top, right, bottom, left = location
    face_image = image[top:bottom, left:right]
    # The dialog expects an RGB image
    return cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

def ensure_log_header():
    newfile = not os.path.exists(LOG_CSV)
    if newfile:
        with open(LOG_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Name", "Category", "ImageFile", "FullImageFile"])

def log_recognition(name, category, face_bgr, full_image_filename):
    # Save cropped face image
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    safe = name.replace(" ", "_") or "Unknown"
    filename = f"{safe}_{category}_{ts}.jpg"
    filepath = os.path.join(LOG_DIR, filename)
    try:
        cv2.imwrite(filepath, face_bgr)
    except Exception as e:
        print("Warning: could not save face image:", e)
        filename = ""

    # Append CSV row
    ensure_log_header()
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            name,
            category,
            filename,
            full_image_filename or ""
        ])

# ---------------- Tkinter Dialog ---------------- #
class FacePreviewDialog:
    def __init__(self, parent, face_img):
        self.top = Toplevel(parent)
        self.top.title("Register New Face")
        self.result = None

        # Convert numpy array to a PhotoImage
        img = Image.fromarray(face_img)
        img = img.resize((150, 150))
        photo = ImageTk.PhotoImage(img)
        
        # Display the face preview
        label_img = Label(self.top, image=photo)
        label_img.image = photo
        label_img.pack(pady=10)

        Label(self.top, text="Enter Name:").pack()
        self.name_entry = Entry(self.top)
        self.name_entry.pack(padx=20)

        Label(self.top, text="Select Category:").pack()
        self.category = StringVar(value="Normal")
        OptionMenu(self.top, self.category, "Normal", "VIP", "Blacklisted").pack()

        Button(self.top, text="Save", command=self.save).pack(side="left", padx=20, pady=10)
        Button(self.top, text="Cancel", command=self.cancel).pack(side="right", padx=20, pady=10)
        
        self.top.grab_set()
        parent.wait_window(self.top)

    def save(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Name cannot be empty", parent=self.top)
            return
        self.result = (name, self.category.get())
        self.top.destroy()

    def cancel(self):
        self.result = None
        self.top.destroy()

# ---------------- Recognition ---------------- #

def find_faces_in_image(image_path, known_encodings, known_names, known_categories):
    # Load the image
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect all faces
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    recognized_faces = []

    # Match each face encoding
    for encoding, location in zip(face_encodings, face_locations):
        name = "Unknown"
        category = "Uncategorized"

        if len(known_encodings) > 0:
            distances = face_recognition.face_distance(known_encodings, encoding)
            best_match_index = np.argmin(distances)
            if distances[best_match_index] < THRESHOLD: # Using the defined threshold
                name = known_names[best_match_index]
                category = known_categories[best_match_index]

        recognized_faces.append((name, category, location))

    # Return the results for this image, the original image, and the new encodings found
    return recognized_faces, image, face_encodings


# ---------------- Display ---------------- #
def display_recognized_faces(image, recognized_faces, image_path):
    # Convert BGR -> RGB and create Pillow canvas with alpha support
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image, "RGBA")

    # Load fonts
    try:
        font_legend = ImageFont.truetype("arial.ttf", size=36)
        font_label = ImageFont.truetype("arial.ttf", size=28)
        font_badge = ImageFont.truetype("arial.ttf", size=22)
    except IOError:
        print("Arial font not found. Using default font.")
        font_legend = ImageFont.load_default()
        font_label = ImageFont.load_default()
        font_badge = ImageFont.load_default()

    # Legend panel (top-left), rounded, semi-transparent
    if recognized_faces:
        box_x, box_y = 14, 14
        box_w = 420
        box_h = 54 + (len(recognized_faces) * 34)
        draw.rounded_rectangle([box_x, box_y, box_x + box_w, box_y + box_h], radius=14, fill=(16, 20, 28, 180), outline=(90, 140, 255, 160), width=2)
        draw.text((box_x + 16, box_y + 12), "Recognized Individuals", font=font_legend, fill=(220, 240, 255, 255))
        y_text = box_y + 54
        for i, (name, category, _) in enumerate(recognized_faces):
            text = f"{i + 1}. {name}"
            draw.text((box_x + 16, y_text), text, font=font_label, fill=(240, 240, 240, 255))
            # category badge
            badge_text = f"{category}"
            badge_w = max(70, int(draw.textlength(badge_text, font=font_badge)) + 16)
            bx = box_x + box_w - badge_w - 16
            by = y_text - 6
            draw.rounded_rectangle([bx, by, bx + badge_w, by + 28], radius=8, fill=(90, 140, 255, 160))
            draw.text((bx + 10, by + 6), badge_text, font=font_badge, fill=(20, 30, 40, 255))
            y_text += 34

    # Draw per-face overlays with rounded label background
    for (name, category, (top, right, bottom, left)) in recognized_faces:
        # palette
        color = (80, 220, 120, 255) if name != "Unknown" else (255, 80, 80, 255)
        # corner lines (via short rectangles for width control)
        ln = max(12, int((right-left) * 0.08))
        thickness = 4
        # top-left
        draw.rectangle([left, top, left+ln, top+thickness], fill=color)
        draw.rectangle([left, top, left+thickness, top+ln], fill=color)
        # top-right
        draw.rectangle([right-ln, top, right, top+thickness], fill=color)
        draw.rectangle([right-thickness, top, right, top+ln], fill=color)
        # bottom-left
        draw.rectangle([left, bottom-thickness, left+ln, bottom], fill=color)
        draw.rectangle([left, bottom-ln, left+thickness, bottom], fill=color)
        # bottom-right
        draw.rectangle([right-ln, bottom-thickness, right, bottom], fill=color)
        draw.rectangle([right-thickness, bottom-ln, right, bottom], fill=color)

        # Label background (rounded)
        label_text = name + " • " + category
        label_w = int(draw.textlength(label_text, font=font_label)) + 20
        label_h = 32
        lx = left
        ly = max(0, top - label_h - 6)
        draw.rounded_rectangle([lx, ly, lx + label_w, ly + label_h], radius=10, fill=(20, 80, 80, 170))
        draw.text((lx + 10, ly + 6), label_text, font=font_label, fill=(240, 240, 240, 255))

    # Convert back to BGR for OpenCV display
    display_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Resize for viewing
    max_width, max_height = 1280, 720
    h, w = display_image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        resized_image = cv2.resize(display_image, (new_w, new_h))
    else:
        resized_image = display_image

    # Save full annotated frame for dashboard display
    try:
        base = os.path.splitext(os.path.basename(image_path or "frame"))[0]
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        full_name = f"frame_{base}_{ts}.jpg"
        full_path = os.path.join(LOG_DIR, full_name)
        cv2.imwrite(full_path, display_image)
        # Also log per-face entries referencing this full image
        for (name, category, (top, right, bottom, left)) in recognized_faces:
            face_bgr = image[top:bottom, left:right]
            log_recognition(name, category, face_bgr, full_name)
    except Exception as e:
        print("Warning: could not save or log annotated frame:", e)

    cv2.imshow("Recognition Results", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------------- Main ---------------- #
def main():
    root = Tk()
    root.withdraw()
    known_encodings, known_names, known_categories = load_encodings()

    while True:
        choice = simpledialog.askstring("Menu", 
            "Choose an option:\n\n1. Recognize from Image\n2. Exit", parent=root)
        
        if choice == "1":
            path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
            if not path:
                continue

            # This function now returns the findings from the image
            recognized_faces, image, new_encodings = find_faces_in_image(
                path, known_encodings, known_names, known_categories
            )

            # Now, iterate through the findings and handle "Unknown" faces
            for i, (name, category, location) in enumerate(recognized_faces):
                if name == "Unknown":
                    # Crop the unknown face for the preview dialog
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    face_crop = rgb_image[location[0]:location[2], location[3]:location[1]]
                    
                    dialog = FacePreviewDialog(root, face_crop)
                    if dialog.result:
                        new_name, new_category = dialog.result
                        
                        # Add the new face data to our lists
                        unknown_encoding = new_encodings[i]
                        known_encodings.append(unknown_encoding)
                        known_names.append(new_name)
                        known_categories.append(new_category)
                        
                        # Save the updated encodings to the file
                        save_encodings(known_encodings, known_names, known_categories)
                        
                        # Update the recognized_faces list so the display shows the new name
                        recognized_faces[i] = (new_name, new_category, location)
                        
                        messagebox.showinfo("Success", f"Face for '{new_name}' registered successfully.")
            
            # Display the final results after all registrations are done
            if recognized_faces:
                display_recognized_faces(image, recognized_faces, path)
            else:
                messagebox.showinfo("Info", "No faces were detected in the selected image.")


        elif choice == "2" or choice is None:
            break

if __name__ == "__main__":
    main()