

# 🧭 **Dokumentasi Proyek: Face Recognition System dengan OpenCV + DeepFace**

## 📘 Tujuan Proyek

Membangun sistem **pengenalan wajah (face recognition)** yang dapat:

* Menangkap wajah lewat webcam.
* Membuat dataset wajah pengguna.
* Mengonversi wajah menjadi **embedding vektor (fitur numerik)** menggunakan model deep learning.
* Mengenali wajah secara real-time dengan perbandingan ke database embedding.

---

## ⚙️ **Lingkungan & Setup**

### Environment

Kamu bekerja di environment conda bernama `ai`:

```
(ai) ariopurba37@arios-MacBook-Air 50-days-of-code %
```

### Python Version

Python 3.11 (via Miniforge/Mac M1)

### Installed Packages

Untuk mendukung TensorFlow di Mac M1, dilakukan instalasi berikut:

```bash
pip uninstall tensorflow tensorflow-macos tensorflow-metal -y
pip install tensorflow-macos==2.13 tensorflow-metal==1.0.1
pip install deepface --no-deps
pip install opencv-python numpy pickle-mixin
```

✅ Tes TensorFlow berhasil dengan:

```python
import tensorflow as tf
tf.__version__        # 2.13.0
tf.config.list_physical_devices('GPU')
# → [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## 🧩 **Struktur Folder Project**

```
50-days-of-code/day-05/
│
├── dataset/                   # hasil capture wajah
│   ├── ario/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│
├── embeddings/                # hasil embedding wajah
│   └── face_embeddings.pkl
│
├── capture_dataset.py         # script tahap 1
├── create_embeddings.py       # script tahap 2
└── real_time_recognition.py   # script tahap 3
```

---

## 🚀 **Tahap 1 – Capture Dataset (Membuat Data Wajah)**

**Tujuan:** Menangkap wajah dari webcam dan menyimpannya sebagai dataset.

```python
# capture_dataset.py
import cv2, os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
name = input("Masukkan nama Anda: ").lower()
dataset_path = f"dataset/{name}"
os.makedirs(dataset_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (200,200))
        cv2.imwrite(f"{dataset_path}/{count}.jpg", face)
        count += 1
    cv2.imshow('Capture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()
```

📦 Output:
50 foto wajah di `dataset/ario/`.

---

## 🧠 **Tahap 2 – Create Embeddings (Konversi ke Vektor Fitur)**

**Tujuan:** Mengubah semua wajah di folder dataset menjadi representasi numerik menggunakan **DeepFace (model: Facenet / ArcFace)**.

```python
# create_embeddings.py
from deepface import DeepFace
import os, pickle, numpy as np

dataset_path = "dataset"
os.makedirs("embeddings", exist_ok=True)

embeddings, labels = [], []

for person_name in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(folder): continue
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        embedding = DeepFace.represent(
            img_path=img_path,
            model_name="ArcFace",           # bisa Facenet / ArcFace
            enforce_detection=False
        )[0]["embedding"]
        emb = np.array(embedding) / np.linalg.norm(embedding)  # normalisasi
        embeddings.append(emb)
        labels.append(person_name)

pickle.dump({"embeddings": embeddings, "labels": labels},
            open("embeddings/face_embeddings.pkl", "wb"))
```

📊 Output:
`face_embeddings.pkl` berisi array 128D untuk tiap wajah dan label nama.

---

## 🎥 **Tahap 3 – Real-Time Recognition**

**Tujuan:** Mengenali wajah lewat webcam dengan membandingkan embedding real-time terhadap database.

```python
# real_time_recognition.py
from deepface import DeepFace
import cv2, pickle, numpy as np

# Load database
data = pickle.load(open("embeddings/face_embeddings.pkl", "rb"))
known_embeddings = np.array(data["embeddings"])
known_labels = np.array(data["labels"])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

def normalize(v):
    v = np.array(v)
    return v / np.linalg.norm(v)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (200,200))

        embedding = DeepFace.represent(img_path=face, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
        embedding = normalize(embedding)

        distances = np.linalg.norm(known_embeddings - embedding, axis=1)
        idx_min = np.argmin(distances)
        min_dist = distances[idx_min]

        threshold = 1.1  # default; bisa disesuaikan
        if min_dist < threshold:
            confidence = (1 - min_dist / threshold) * 100
            name = known_labels[idx_min]
            label_text = f"{name} ({confidence:.1f}%)"
            color = (0,255,0)
        else:
            label_text = "Unknown"
            color = (0,0,255)

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## ❌ **Masalah yang Masih Belum Selesai**

> Sistem sudah berfungsi penuh (deteksi & embedding), namun hasil pengenalan real-time masih selalu menampilkan **"Unknown"**.

### 🔍 Kemungkinan penyebab:

1. Perbedaan domain gambar dataset (BGR) vs kamera (RGB).
2. Threshold pembeda terlalu ketat (`0.7` terlalu kecil).
3. Model `Facenet` tidak konsisten di TensorFlow Mac → akan dicoba `ArcFace`.
4. Embedding belum dinormalisasi → sudah diperbaiki di versi terbaru.
5. Dataset wajah kurang variasi pencahayaan / ekspresi.

### ✅ Solusi yang sudah disiapkan:

* Ubah semua pipeline ke model `"ArcFace"`.
* Tambahkan normalisasi embedding (`/ np.linalg.norm(embedding)`).
* Gunakan `threshold = 1.1` atau `1.2` untuk pengujian awal.
* Tambahkan konversi warna `cv2.COLOR_BGR2RGB`.

---

## 📘 **Langkah Selanjutnya (Nanti Saat Lanjut Coding)**

1. **Tes ulang real_time_recognition.py versi terakhir.**

   * Pastikan model `"ArcFace"`, threshold `1.1`.
   * Perhatikan output distance di terminal.

2. **Tambahkan dataset wajah dengan pencahayaan berbeda.**

   * 2 sesi foto: siang dan malam, ekspresi berbeda.

3. **Evaluasi hasil distance:**

   * Catat nilai `min_dist` rata-rata untuk wajah yang benar dikenali.
   * Gunakan nilai itu untuk menentukan threshold ideal.

4. **Jika sudah akurat:**

   * Lanjutkan ke fitur lanjutan:

     * 🔒 **Face-based Login System**
     * 🧾 **Absensi Otomatis (log ke CSV)**
     * 🌐 **Streamlit Web App untuk demo**

---

## 🧱 **Checklist Teknis Terakhir**

| Tahap                                               | Status | Catatan                         |
| --------------------------------------------------- | ------ | ------------------------------- |
| Environment `ai` aktif dan TensorFlow-Mac berfungsi | ✅      | Sudah dites                     |
| Capture dataset wajah berhasil                      | ✅      | File JPG tersimpan              |
| Embedding wajah dibuat dengan DeepFace              | ✅      | `face_embeddings.pkl` terbentuk |
| Real-time deteksi wajah muncul                      | ✅      | Kotak muncul di webcam          |
| Pengenalan wajah masih “Unknown”                    | ⚠️     | Butuh tuning threshold/model    |
| Model ArcFace siap dicoba                           | 🔄     | Langkah selanjutnya             |
| Dokumentasi lengkap dibuat                          | ✅      | File ini                        |

---
