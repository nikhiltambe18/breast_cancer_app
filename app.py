import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("Breast Cancer Classification")

model = tf.keras.models.load_model("breast_cancer_cnn.h5")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img)

    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        st.write("🔴 **Malignant**")
    else:
        st.write("🟢 **Benign**")
