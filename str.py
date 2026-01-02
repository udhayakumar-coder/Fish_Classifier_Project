import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Fish Image Classifier",
    page_icon="🐟",
    layout="centered"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.main {
    background-color: #0e1117;
    color: white;
}
.stButton>button {
    background-color: #1f77b4;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
}
.stFileUploader {
    border: 2px dashed #1f77b4;
    border-radius: 10px;
}
.result-box {
    background-color: #1e1e1e;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    text-align: center;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.markdown("<h1 style='text-align:center;'>🐠 Fish Image Classification App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload an image and get instant prediction</p>", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_trained_model():
    return load_model(
    r"D:\project\fish_predict\fish_classifier.keras",
    compile=False
)

model = load_trained_model()

# -----------------------------
# CLASS NAMES
# -----------------------------
class_names = [
    "black_sea_sprat",
    "gilt_head_bream",
    "horse_mackerel",
    "red_mullet",
    "sea_bass",
    "shrimp",
    "striped_red_mullet",
    "trout"
]

IMG_SIZE = (224, 224)

# -----------------------------
# FILE UPLOADER
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload a Fish Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Uploaded Image", use_container_width=True)

    # -----------------------------
    # PREPROCESS IMAGE
    # -----------------------------
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -----------------------------
    # PREDICT BUTTON
    # -----------------------------
    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            pred = model.predict(img_array)
            predicted_class = class_names[np.argmax(pred)]
            confidence = np.max(pred) * 100

        # -----------------------------
        # RESULT
        # -----------------------------
        st.markdown(
            f"""
            <div class="result-box">
                🐟 <b>Predicted Class:</b> {predicted_class}<br>
                📊 <b>Confidence:</b> {confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Built with ❤️ using Streamlit & TensorFlow</p>",
    unsafe_allow_html=True
)