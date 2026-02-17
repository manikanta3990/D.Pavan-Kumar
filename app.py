import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "src/models/saved_model/model.h5"
IMG_SIZE = 224

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# DIAGNOSIS ENGINE
# -----------------------------
def diagnosis_engine(prediction, confidence, symptoms):
    fever = symptoms.get("Fever")
    fatigue = symptoms.get("Fatigue")
    chills = symptoms.get("Chills")

    if prediction == "Parasitized":
        if fever and chills:
            return "⚠ High Malaria Risk - Immediate medical consultation recommended."
        elif fever:
            return "⚠ Possible Malaria - Consult a physician."
        else:
            return "Parasite detected. Further clinical testing required."
    else:
        if fever:
            return "No parasite detected, but symptoms present. Seek medical advice."
        else:
            return "No parasite detected. Low risk."

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Parasite Detection AI", layout="wide")

st.title("🦠 AI-Powered Parasite Detection System")
st.markdown("Upload a blood smear image to detect malaria parasites.")

uploaded_file = st.file_uploader("Upload Microscopic Image", type=["jpg", "png", "jpeg"])

st.sidebar.header("🩺 Symptom Checker")
fever = st.sidebar.checkbox("Fever")
fatigue = st.sidebar.checkbox("Fatigue")
chills = st.sidebar.checkbox("Chills")

symptoms = {
    "Fever": fever,
    "Fatigue": fatigue,
    "Chills": chills
}

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)
    prediction_prob = model.predict(processed_image)[0][0]

    if prediction_prob > 0.5:
        prediction = "Parasitized"
        confidence = prediction_prob
    else:
        prediction = "Uninfected"
        confidence = 1 - prediction_prob

    diagnosis = diagnosis_engine(prediction, confidence, symptoms)

    with col2:
        st.subheader("🔍 Prediction Result")
        st.write(f"**Class:** {prediction}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")

        if prediction == "Parasitized":
            st.error("Parasite Detected")
        else:
            st.success("No Parasite Detected")

        st.subheader("🩺 Automated Diagnosis")
        st.info(diagnosis)

    st.markdown("---")
    st.subheader("📊 Model Information")
    st.write("Model: CNN (Transfer Learning)")
    st.write("Image Size:", IMG_SIZE)
    st.write("Binary Classification: Parasitized vs Uninfected")

    st.warning("⚠ Disclaimer: This system is for research/demo purposes only and not a replacement for professional medical diagnosis.")
