# real_time_recognition.py
from deepface import DeepFace
import cv2, pickle, numpy as np

# --- 1. Load database embeddings ---
with open("/Users/ariopurba37/Documents/dream/coding/50-days-of-code/day-05/embeddings/face_embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    known_embeddings = np.array(data["embeddings"])
    known_labels = np.array(data["labels"])

print(f"[INFO] Database wajah ter-load: {len(known_labels)} data")

# --- 2. Load model deteksi wajah ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --- 3. Jalankan webcam ---
cap = cv2.VideoCapture(0)
print("[INFO] Tekan 'q' untuk keluar")

def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (200,200))
        
        try:
            # --- 4. Dapatkan embedding wajah baru ---
            embedding = DeepFace.represent(img_path=face, model_name="Facenet", enforce_detection=False)[0]["embedding"]
            embedding = np.array(embedding)
            
            # --- 5. Hitung jarak ke seluruh database ---
            distances = np.linalg.norm(known_embeddings - embedding, axis=1)
            idx_min = np.argmin(distances)
            min_dist = distances[idx_min]
            name = known_labels[idx_min]
            
            # --- 6. Threshold keputusan ---
            threshold = 0.7
            if min_dist < threshold:
                confidence = (1 - min_dist) * 100
                label_text = f"{name} ({confidence:.1f}%)"
                color = (0,255,0)
            else:
                label_text = "Unknown"
                color = (0,0,255)
            
            # --- 7. Gambar di frame ---
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        except Exception as e:
            print("[WARN]", e)
            continue

    cv2.imshow("Real-Time Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
