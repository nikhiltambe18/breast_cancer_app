import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("Breast Cancer Classification")

model = tf.keras.models.load_model("breast_cancer_cnn.h5")

# Read model input shape
input_shape = model.input_shape
HEIGHT = input_shape[1]
WIDTH = input_shape[2]
CHANNELS = input_shape[3]

st.write(f"Model expects input shape: {HEIGHT} x {WIDTH} x {CHANNELS}")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file)

    # Convert to grayscale if model expects 1-channel
    if CHANNELS == 1:
        img = img.convert("L")

    # Resize to required model size
    img = img.resize((WIDTH, HEIGHT))
    st.image(img)

    # Convert to numpy array
    img = np.array(img) / 255.0

    # Expand dims for grayscale
    if CHANNELS == 1:
        img = np.expand_dims(img, axis=-1)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)[0][0]

    if pred > 0.5:
        st.write("🔴 *Malignant*")
    else:
        st.write("🟢 *Benign*")
