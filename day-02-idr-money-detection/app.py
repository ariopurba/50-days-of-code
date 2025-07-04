import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

st.title("💵 Indonesian Money Classification")
st.markdown("Upload an image of Indonesian money and let the AI predict its denomination!")

# Load model
model = tf.keras.models.load_model("uang.h5")

# Ambil nama kelas dari folder dataset (atau bisa juga dari dataset.class_names jika ada)
dataset_path = Path("./dataset/dataset/Uang Baru")
class_names = sorted([folder.name for folder in dataset_path.iterdir() if folder.is_dir()])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Preprocess
    img_dim = (224, 224)
    img = image.resize(img_dim)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    pred_class = np.argmax(predictions, axis=1)[0]
    confidence = tf.nn.softmax(predictions, axis=1).numpy()[0][pred_class]

    st.success(f"Predicted: **{class_names[pred_class]}**")
    st.markdown(f"**Confidence:** `{confidence:.2%}`")
    st.bar_chart(tf.nn.softmax(predictions, axis=1).numpy()[0])