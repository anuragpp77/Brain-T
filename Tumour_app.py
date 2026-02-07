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
@@ -224,4 +243,5 @@
            <div style="text-align: right; margin-top: -45px; font-size: 24px;">➝</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    """, unsafe_allow_html=True)
