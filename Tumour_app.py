import streamlit as st
import numpy as np
from PIL import Image
from skimage.transform import resize
import tensorflow as tf
import base64
import os
import requests

@st.cache_resource
def load_prediction_model():
    model_path = 'brain_tumor.keras'
    url = "https://drive.google.com/file/d/1CUUrtWOOi4Izi7URntsL0c2HMOCRhCxo/view?usp=sharing"
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive... this may take a minute."):
            try:
                response = requests.get(url)
                with open(model_path, "wb") as f:
                    f.write(response.content)
            except Exception as e:
                st.error(f"Download failed: {e}")
                return None
                
    return tf.keras.models.load_model(model_path)


# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Brain Tumour Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --- 2. HELPER FUNCTIONS ---
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None
img_b64 = get_base64_image("a.jpg")
if img_b64:
    hero_bg_image = f"url('data:image/jpg;base64,{img_b64}')"
else:
    hero_bg_image = "url('https://img.freepik.com/free-photo/purple-brain-digital-art_23-2151121175.jpg')"



@st.cache_resource
def load_prediction_model():
    try:
        loaded_model = tf.keras.models.load_model('brain_tumor.keras')
        return loaded_model
    except:
        return None


# --- 3. LOAD ASSETS ---
cnn_model = load_prediction_model()
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

def process_image(img):
    img_gray = img.convert('L')
    img_array = np.array(img_gray)
    img_resized = resize(img_array, (150, 150, 1))
    img_final = img_resized.reshape(1, 150, 150, 1)
    return img_final

# --- 4. CUSTOM CSS ---
st.markdown(f"""
<style>
    /* 1. Main Background */
    .stApp {{
        background: #f0f2f6;
        background: linear-gradient(315deg, #f0f2f6 0%, #c5cbe3 74%);
    }}

    /* 2. THE LEFT "CARD" STYLE */
    div[data-testid="column"]:nth-of-type(1) > div {{
        background-color: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }}

    /* 3. UPLOADER STYLING */
    [data-testid="stFileUploader"] {{
        background-color: rgba(235, 240, 255, 0.4);
        border: 2px dashed #8c92ac;
        border-radius: 20px;
        padding: 30px 20px; 
        transition: all 0.3s ease-in-out;
    }}
    [data-testid="stFileUploader"]:hover {{
        background-color: rgba(235, 240, 255, 0.7);
        border-color: #5D17EB;
        box-shadow: 0 0 15px rgba(93, 23, 235, 0.1);
    }}
    [data-testid="stFileUploader"] section {{
        background-color: transparent !important;
        padding: 0 !important;
    }}
    [data-testid="stFileUploader"] section > div {{
        color: #36454F !important;
        font-weight: 500;
    }}
    /* The small "Browse files" button inside the uploader */
    [data-testid="stFileUploader"] button {{
        background: linear-gradient(90deg, #4A90E2 0%, #5D17EB 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 8px 25px;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(93, 23, 235, 0.2);
    }}
    [data-testid="stFileUploader"] small {{
        color: #5D17EB !important;
        font-size: 12px;
    }}

    /* 4. RIGHT HERO CARD */
    .hero-container {{
        width: 100%;
        height: 100%;
        min-height: 500px;
        border-radius: 20px;
        background-image: {hero_bg_image};
        background-size: cover;
        background-position: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
    }}
    .hero-overlay {{
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 100%;
        background: linear-gradient(to top, rgba(93, 23, 235, 0.9) 0%, rgba(93, 23, 235, 0.2) 60%, transparent 100%);
        padding: 40px;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        color: white;
    }}

    /* --- 5. ANALYZE BUTTON STYLING (The Big Red Button) --- */
    div.stButton > button {{
        background: linear-gradient(to right, #5D17EB, #1A0F2E ) !important; 
        color: white !important;  /* <--- !important FIXES THE VISIBILITY */
        border: none !important;
        border-radius: 12px !important;
        padding: 15px 30px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        transition: 0.3s;
        box-shadow: 0 5px 15px rgba(93, 23, 235, 0.4);
    }}

    div.stButton > button:active {{
        transform: scale(0.98);
    }}

    /* Typography & Results */
    h3 {{ margin-top: 0; font-family: 'Helvetica Neue', sans-serif; }}
    .result-box {{
        background: #fff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-top: 20px;
    }}
</style>
""", unsafe_allow_html=True)


# --- 5. MAIN LAYOUT ---

# Navbar
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; padding: 5px; margin-bottom: 20px;">
    <div style="font-size: 24px; font-weight: 800; letter-spacing: -1px;
         background: linear-gradient(to right, #5D17EB, #FF00CC);
         -webkit-background-clip: text;
         -webkit-text-fill-color: transparent;">
        BRAIN T
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.2], gap="large")

# === LEFT COLUMN ===
with col1:
    st.markdown("""<div style="text-align: center;"><h3>Upload Medical Scans</h3></div>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")

    # This button will now be styled by section #5 in the CSS above
    analyze_btn = st.button("Analyze Scan", type="primary", use_container_width=True, disabled=not uploaded_file)

    if analyze_btn and uploaded_file:
        if cnn_model is None:
            st.error("Model not found. Please ensure 'brain_tumor.keras' is in the folder.")
        else:
            try:
                image_data = Image.open(uploaded_file)
                processed_input = process_image(image_data)
                predictions = cnn_model.predict(processed_input)
                confidence = np.max(predictions)
                predicted_class = CLASS_NAMES[np.argmax(predictions)]

                bar_color = "#2E7D32" if predicted_class != "No Tumor" else "#1976D2"

                st.markdown(f"""
                <div class="result-box" style="border-left: 5px solid {bar_color};">
                    <div style="font-size: 12px; color: #888; font-weight: 600;">Prediction Results</div>
                    <div style="font-size: 22px; color: {bar_color}; font-weight: 700; margin: 5px 0;">{predicted_class} (Predicted)</div>
                    <div style="font-size: 12px; color: #555;">Confidence Score: <b>{confidence * 100:.1f}%</b></div>
                    <div style="width: 100%; height: 6px; background: #eee; border-radius: 5px; margin-top: 8px;">
                        <div style="width: {confidence * 100}%; height: 100%; background: {bar_color}; border-radius: 5px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

# === RIGHT COLUMN ===
with col2:
    st.markdown("""
    <div class="hero-container">
        <div class="hero-overlay">
            <h1 style="font-size: 32px; font-weight: 700; margin-bottom: 10px;">Brain Tumor<br>Prediction system</h1>
            <p style="font-size: 14px; opacity: 0.9; line-height: 1.6; max-width: 90%;">
                Discover limitless potential to combine medical and technological aspects.
            </p>
            <div style="text-align: right; margin-top: -45px; font-size: 24px;">➝</div>
        </div>
    </div>

    """, unsafe_allow_html=True)
