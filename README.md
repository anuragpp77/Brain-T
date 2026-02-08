# Brain-T

# Brain T: Deep Learning Framework for Intracranial Tumor Classification

**Automated Neuropathology Diagnostics via Convolutional Neural Networks**

### Project Abstract

Brain Wave is a high-performance medical imaging application engineered to assist in the automated detection and multi-class classification of brain tumors from Magnetic Resonance Imaging (MRI) scans. Leveraging a custom-trained Convolutional Neural Network (CNN) architecture, the system provides rapid, probabilistic assessments of intracranial anomalies, distinguishing between Glioma, Meningioma, Pituitary tumors, and healthy tissue with high fidelity.

This repository hosts the deployment-ready web application, which features a modernized, glassmorphism-based user interface (UI) built on Streamlit, designed to streamline the workflow for preliminary medical image analysis.

### Technical Architecture

The core of the application is built upon a robust deep learning pipeline:

* **Model Architecture:** A specialized CNN trained on a curated dataset of MRI scans, optimized for feature extraction (edges, textures, and tumor boundaries) and classification.
* **Preprocessing Pipeline:** Automated image standardization including grayscale conversion, resizing (150x150px), and array normalization to ensure consistent model inference.
* **Inference Engine:** Real-time prediction capabilities powered by TensorFlow/Keras, delivering instant classification results with associated confidence metrics.

### Key Capabilities

* **Multi-Class Classification:** Accurately identifies four distinct classes: *Glioma*, *Meningioma*, *Pituitary Tumor*, and *No Tumor*.
* **Confidence Quantification:** Returns a precise confidence score (%) for every prediction, aiding in the assessment of model certainty.
* **DICOM/NIFTI Visual Support:** UI indicators for standard medical imaging formats.
* **Advanced UI/UX:** A custom-styled, responsive interface featuring glassmorphism design principles to enhance visual clarity and user engagement.

### Technology Stack

* **Deep Learning:** Keras
* **Frontend Deployment:** Streamlit
* **Data Processing:** NumPy, Pillow (PIL), Scikit-Image
* **Version Control:** Git

### Objective

The primary objective of this project is to demonstrate the efficacy of deep learning in reducing diagnostic turnaround times and serving as a reliable second-opinion tool for radiologists and neurologists.

---
The url for the test and training data : https://drive.google.com/drive/folders/16cWy7jHD6gwRMYqJy4inmC1xffHjVC02?usp=drive_link
