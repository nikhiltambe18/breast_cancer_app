import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -------------------------
# 1. Load trained model
# -------------------------
MODEL_PATH = "model.h5"      # <--- keep model file in same folder as app.py
model = load_model(MODEL_PATH)

# -------------------------
# 2. Preprocessing function
# -------------------------
def preprocess_image(img):
    img = img.resize((224, 224))     # <-- change to your model input size
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------
# 3. Streamlit UI
# -------------------------
st.title("Breast Cancer Classification (CNN)")
st.write("Upload a histopathology image to classify as **Benign** or **Malignant**.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = load_img(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        # 4. Preprocess
        processed = preprocess_image(img)

        # 5. Predict
        prediction = model.predict(processed)[0][0]  # assuming binary output

        if prediction > 0.5:
            result = "Malignant"
            color = "red"
        else:
            result = "Benign"
            color = "green"

        st.markdown(
            f"<h2 style='color:{color}; text-align:center;'>Prediction: {result}</h2>",
            unsafe_allow_html=True
        )
