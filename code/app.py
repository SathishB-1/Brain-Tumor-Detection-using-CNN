import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained CNN model
model = load_model("brain_tumor_cnn_model.h5")
class_names = ['No Tumor', 'Tumor']

# Page setup
st.set_page_config(page_title="Brain Tumor Detection", layout="wide", page_icon="🧠")

# Title and instructions
st.markdown("<h1 style='text-align: center;'>🧠 Brain Tumor Detection from MRI Scans</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>This AI model can predict whether an uploaded brain MRI scan indicates a tumor or not.</p>", unsafe_allow_html=True)
st.markdown("---")

# Layout with columns
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])
    st.markdown("💡 *Use a clear MRI scan image.*")

with right_col:
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ MRI Scan Preview", width=300)

        # Preprocess image
        img = image.resize((150, 150))
        img = np.array(img)
        if img.shape[-1] == 4:
            img = img[..., :3]  # Remove alpha channel
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        with st.spinner("🧪 Analyzing the scan..."):
            prediction = model.predict(img)
            pred_idx = np.argmax(prediction)
            confidence = float(prediction[0][pred_idx])
            label = class_names[pred_idx]

        # Show result
        st.subheader("🔍 Prediction Result")
        if label == "Tumor":
            st.error(f"🧬 **Tumor Detected**\nConfidence: **{confidence:.2%}**")
            st.markdown("⚠️ Please consult a medical specialist.")
        else:
            st.success(f"✅ **No Tumor Detected**\nConfidence: **{confidence:.2%}**")
            st.markdown("🎉 You appear healthy. Still, always confirm with a doctor.")
    else:
        st.info("👈 Upload an MRI image on the left to get started.")
