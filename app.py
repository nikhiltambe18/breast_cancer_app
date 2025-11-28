import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page config
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load model first
model = tf.keras.models.load_model("breast_cancer_cnn.h5")

# Read model input shape
input_shape = model.input_shape
HEIGHT = input_shape[1]
WIDTH = input_shape[2]
CHANNELS = input_shape[3]

# Title
st.markdown("<h1 style='text-align: center; color: grey;'>Breast Cancer Classification</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload a breast tissue image (jpg, jpeg, png).  
2. The model will predict whether it's Malignant 🔴 or Benign 🟢.  
3. Model input size: {} x {} x {}  
""".format(HEIGHT, WIDTH, CHANNELS))

# Display expected input shape
st.write(f"Model expects input shape: **{HEIGHT} x {WIDTH} x {CHANNELS}**")

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file)

    # Convert to grayscale if model expects 1-channel
    if CHANNELS == 1:
        img = img.convert("L")

    # Resize to model input
    img = img.resize((WIDTH, HEIGHT))

    # Show uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert to numpy array
    img = np.array(img) / 255.0
    if CHANNELS == 1:
        img = np.expand_dims(img, axis=-1)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)[0][0]

    # Show results
    st.markdown("---")
    if pred > 0.5:
        st.markdown("<h2 style='color: red; text-align: center;'>🔴 Malignant</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color: green; text-align: center;'>🟢 Benign</h2>", unsafe_allow_html=True)
