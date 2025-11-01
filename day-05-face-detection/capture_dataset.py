# capture_dataset.py
import cv2
import os

# --- 1. Inisialisasi detektor wajah ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 2. Input nama user (untuk folder dataset) ---
name = input("Masukkan nama Anda: ").lower()
dataset_path = f"./dataset/{name}"

# --- 3. Buat folder jika belum ada ---
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    print(f"[INFO] Folder dataset dibuat: {dataset_path}")

# --- 4. Buka webcam ---
cap = cv2.VideoCapture(0)
count = 0

print("[INFO] Tekan 'q' untuk keluar...")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- 5. Deteksi wajah ---
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # --- 6. Gambar kotak di wajah ---
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        
        # --- 7. Potong dan simpan wajah ---
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        file_name = f"{dataset_path}/{count}.jpg"
        cv2.imwrite(file_name, face)
        count += 1

        cv2.putText(frame, f"Captured: {count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow('Face Capture', frame)

    # --- 8. Hentikan dengan 'q' atau setelah 50 foto ---
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Dataset {name} selesai. Total gambar: {count}")
