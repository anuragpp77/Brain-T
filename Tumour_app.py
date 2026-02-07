import os
import requests

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Brain Tumour Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. HELPER FUNCTIONS ---

@st.cache_resource
def load_prediction_model():
    model_path = 'brain_tumor.keras'
    file_id = '1CUUrtWOOi4Izi7URntsL0c2HMOCRhCxo'
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    # 1. Download only if file doesn't exist
    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive... this may take a minute."):
        url = "https://drive.google.com/uc?export=download"
        session = requests.Session()
        
        with st.spinner("Downloading model from Google Drive... (This happens once)"):
            try:
                response = requests.get(url)
                with open(model_path, "wb") as f:
                    f.write(response.content)
            except Exception as e:
                st.error(f"Download failed: {e}")
                return None
                # First request to get the confirmation token (for large files)
                response = session.get(url, params={'id': file_id}, stream=True)

    return tf.keras.models.load_model(model_path)
                token = None
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        token = value
                        break

                # If there is a warning token, we need to add it to the request
                if token:
                    params = {'id': file_id, 'confirm': token}
                    response = session.get(url, params=params, stream=True)
                
                # Write the file in chunks to avoid memory issues
                if response.status_code == 200:
                    with open(model_path, "wb") as f:
                        for chunk in response.iter_content(32768):
                            if chunk:
                                f.write(chunk)
                else:
                    st.error(f"Google Drive Error: Status Code {response.status_code}")
                    return None
                    
            except Exception as e:
                st.error(f"Download connection failed: {e}")
                return None
    
    # 2. Load the Model
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        # If this happens, the file is likely corrupt/HTML. Delete it so we try again next time.
        if os.path.exists(model_path):
            os.remove(model_path)
        st.error(f"Error loading model: {e}. Please Reboot the app to re-download.")
        return None
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



# @st.cache_resource
# def load_prediction_model():
#     try:
#         loaded_model = tf.keras.models.load_model('brain_tumor.keras')
#         return loaded_model
#     except:
#         return None


# --- 3. LOAD ASSETS ---
cnn_model = load_prediction_model()
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

def process_image(img):
    img_gray = img.convert('L')
@@ -74,6 +81,17 @@ def process_image(img):
    img_final = img_resized.reshape(1, 150, 150, 1)
    return img_final

# --- 3. LOAD ASSETS ---
cnn_model = load_prediction_model()
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

img_b64 = get_base64_image("a.jpg")
if img_b64:
    hero_bg_image = f"url('data:image/jpg;base64,{img_b64}')"
else:
    hero_bg_image = "url('https://img.freepik.com/free-photo/purple-brain-digital-art_23-2151121175.jpg')"


# --- 4. CUSTOM CSS ---
st.markdown(f"""
<style>
@@ -241,14 +259,11 @@ def process_image(img):
    st.markdown("""
    <div class="hero-container">
        <div class="hero-overlay">
            <h1 style="font-size: 32px; font-weight: 700; margin-bottom: 10px;">Brain Tumor<br>Prediction system</h1>
            <h1 style="font-size: 32px; font-weight: 700; margin-bottom: 10px;">Brain Tumour<br>Prediction system</h1>
            <p style="font-size: 14px; opacity: 0.9; line-height: 1.6; max-width: 90%;">
                Discover limitless potential to combine medical and technological aspects.
            </p>
            <div style="text-align: right; margin-top: -45px; font-size: 24px;">➝</div>
        </div>
    </div>

    """, unsafe_allow_html=True)
