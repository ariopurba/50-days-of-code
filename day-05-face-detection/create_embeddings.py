# create_embeddings.py
from deepface import DeepFace
import os, cv2, pickle
import numpy as np

# --- 1. Path dataset dan output ---
dataset_path = "/Users/ariopurba37/Documents/dream/coding/50-days-of-code/day-05/dataset"
output_file = "/Users/ariopurba37/Documents/dream/coding/50-days-of-code/day-05/embeddings/face_embeddings.pkl"

# --- 2. Cek folder output ---
os.makedirs("embeddings", exist_ok=True)

# --- 3. Variabel penyimpanan ---
embeddings = []
labels = []

# --- 4. Loop tiap orang di dataset ---
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    
    if not os.path.isdir(person_folder):
        continue
    
    print(f"[INFO] Proses wajah: {person_name}")
    
    # --- 5. Loop tiap gambar orang tersebut ---
    for file in os.listdir(person_folder):
        file_path = os.path.join(person_folder, file)
        
        try:
            # --- 6. Representasikan wajah (128D embedding) ---
            embedding = DeepFace.represent(
                img_path=file_path, 
                model_name="Facenet", 
                enforce_detection=False
            )
            
            embeddings.append(embedding[0]["embedding"])
            labels.append(person_name)
        
        except Exception as e:
            print(f"[ERROR] Gagal proses {file_path}: {e}")

# --- 7. Simpan ke file pickle ---
with open(output_file, "wb") as f:
    pickle.dump({"embeddings": embeddings, "labels": labels}, f)

print(f"[INFO] Total wajah tersimpan: {len(labels)}")
print(f"[INFO] File database dibuat: {output_file}")
