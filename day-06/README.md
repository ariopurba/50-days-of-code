
---

## **README #2 — Smart Recycling Bin (Trash Classification using MobileNetV2)**
**File:** `README_Trash_Classification.md`


# Smart Recycling Bin – Trash Classification with MobileNetV2 (Transfer Learning)

## Project Summary
Proyek ini bertujuan membuat **model klasifikasi jenis sampah** menggunakan **Transfer Learning (MobileNetV2 pretrained on ImageNet)**.  
Model diharapkan dapat mengklasifikasikan gambar menjadi 5 kategori:
- Plastik  
- Kertas  
- Logam  
- Kaca  
- Organik  

Proyek ini merupakan simulasi *real-world case* untuk sistem **Smart Recycling Bin**, di mana kamera mengenali jenis sampah dan mengarahkan ke wadah yang tepat.

---

## Architecture Overview
- **Backbone:** MobileNetV2 (ImageNet weights, frozen)
- **Classifier Head:** GlobalAveragePooling2D → Dropout(0.2) → Dense(5, softmax)
- **Fine-tuning:** sebagian layer terakhir dibuka setelah training tahap pertama.

---

## Tools & Libraries
- TensorFlow / Keras  
- NumPy, Matplotlib  
- scikit-learn (untuk confusion matrix)
- Kaggle / GitHub dataset: [TrashNet Dataset](https://github.com/garythung/trashnet)

---

## Dataset Status
Dataset belum selesai dimuat.  
 **Progress terakhir:** struktur folder sudah disiapkan, namun dataset belum direload sepenuhnya ke pipeline `image_dataset_from_directory()`.

```python
DATA_DIR = "dataset"
train_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(160,160),
    batch_size=32
)
