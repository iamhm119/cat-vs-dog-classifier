import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

st.title("🐶🐱 Cat vs Dog Classifier")
st.write("Upload an image and I will predict whether it is a **Cat** or a **Dog**.")

@st.cache_resource
def load_model():
    if not os.path.exists("model_tf.keras"):
        st.error("❌ model_tf.keras not found. Please train the model first.")
        return None
    return tf.keras.models.load_model("model_tf.keras", compile=False)

model = load_model()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = float(model.predict(img)[0][0])

    if prediction > 0.5:
        st.success(f"🐶 Dog (Confidence: {prediction:.2f})")
    else:
        st.success(f"🐱 Cat (Confidence: {1 - prediction:.2f})")
