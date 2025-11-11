# UAV Integrated KNN-Based Face Recognition System for Public Safety

## - Overview
This project integrates a **micro-UAV** with a **KNN + Euclidean Distance based face recognition system**.  
The system supports **three working modes**:

1.  **Normal Mode** – Single face recognition (real-time)
2.  **Batch Mode** – Multiple face recognition from a single frame
3.  **Training Mode** – Add new user images & generate encodings

A custom **dashboard** is also included for uploading images, managing dataset, and viewing recognition history.

---

##  Features

###  UAV System
- Micro brushed quadcopter  
- PID-based flight control  
- FPV camera for live streaming  
- 3.7V Li-Po (300–350mAh)  
- Lightweight ABS enclosure  

---

##  3 Operating Modes (Software)

### 🔵 **1. Normal Mode (Single-Face Recognition)**
- Used during standard UAV surveillance  
- Detects and identifies **one face at a time**  
- Best for stable tracking  
- Fastest processing mode  
- Outputs:  
  - Name  
  - Euclidean Distance  
  - Timestamp  
  - Frame preview

---

###  **2. Batch Mode (Multi-Face Recognition)**
- Detects **multiple faces** in the same frame  
- Computes embeddings for all faces  
- Performs KNN and Euclidean distance matching  
- Stores **multiple recognition results in a single timestamp**  
- Useful in:  
  - Crowded areas  
  - Events  
  - Public safety scanning  
  - College gatherings

---

###  **3. Training Mode (Add New User Data)**
Used for dataset creation & updating:

- Upload images through dashboard  
- Automatically detect face in image  
- Generate **128-D embedding**  
- Append to dataset  
- Re-train KNN model  
- Save updated:
  - `encodings.pkl`
  - `knn_model.pkl`

 No manual coding needed — fully automated  
 Supports multiple images per user  
 Improves recognition accuracy

---

##  Architecture

UAV → Camera → Video Stream → Ground System →

Normal Mode | Batch Mode | Training Mode → Dashboard → Logs


---

##  KNN Face Recognition Pipeline

1. Detect faces using Haar/face_recognition  
2. Extract embeddings (128-D face vector)  
3. Compare using Euclidean distance  
4. KNN selects nearest labels (K = 3 or 5)  
5. Predict identity with voting mechanism  
6. Store recognition history

---

##  Code Snippet (Supports All Three Modes)

```python
import cv2
import face_recognition
import pickle
from sklearn.neighbors import KNeighborsClassifier

# Load saved encodings + model
with open("encodings.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

with open("knn_model.pkl", "rb") as f:
    knn = pickle.load(f)

MODE = "normal"   # options: normal, batch, train

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    if MODE == "normal" and len(face_encodings) > 0:
        face_encodings = [face_encodings[0]]
        face_locations = [face_locations[0]]

    for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):

        dist, idx = knn.kneighbors([encoding], n_neighbors=3, return_distance=True)
        euclidean = dist[0][0]

        if euclidean < 0.6:
            name = known_names[idx[0][0]]
        else:
            name = "Unknown"

        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, f"{name} ({round(euclidean,3)})", 
                    (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Save to recognition history
        # save_history(name, euclidean, MODE)

    cv2.imshow("UAV Recognition System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

- Dashboard Features

- Upload User Images

- Auto-encoding generation

- Dataset storage

- Recognition history table

- Multiple face logs per frame (Batch Mode)

- Re-training interface

- Power Summary

Component	Current

Motors	1.7–1.8A

Camera	200mA

FC	70mA

Total	~2A

- Flight Time (300mAh): 5–6 min
- Flight Time (350mAh): 6.3–6.5 min

- Applications

Public safety monitoring

Crowd scanning

Campus surveillance

Emergency response

Real-time identity verification

- Installation
git clone https://github.com/your-username/uav-face-recognition.git

cd uav-face-recognition

pip install -r requirements.txt

- License

MIT License