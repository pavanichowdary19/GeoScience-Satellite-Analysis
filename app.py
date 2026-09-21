import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="GeoScience AI",
    page_icon="🛰️",
    layout="centered"
)

st.title("🛰️ GeoScience Satellite Image Analysis")
st.write("Upload a satellite image to predict its land-cover category.")

class_names = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("geoscience_model.keras")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a satellite image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Satellite Image", use_container_width=True)

    if st.button("Analyze Image"):
        image = image.resize((64, 64))
        image_array = np.array(image)

        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array)

        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = predictions[0][predicted_index] * 100

        st.success(f"Predicted Class: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")

        st.subheader("Prediction probabilities")

        for i, name in enumerate(class_names):
            st.write(f"{name}: {predictions[0][i] * 100:.2f}%")
