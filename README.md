🧠 Brain Tumor Classification — VGG16 Transfer Learning

One-line description: "Deep learning model to classify brain MRI scans into 4 tumor categories using fine-tuned VGG16."
Streamlit: https://brain-t-prediction.streamlit.app/

📌 Table of Contents

Demo
Overview
Dataset
Model Architecture
Improvements Over Baseline
Training
Results
Installation
Usage
Project Structure
Future Work
License


🎯 Demo

Screenshot or GIF of the Streamlit app
Sample prediction output


📖 Overview

Problem statement (1–2 lines)
4 classes: Glioma, Meningioma, No Tumor, Pituitary
Approach: Transfer learning on VGG16


📂 Dataset

Source / download link
Folder structure (Training / Testing split)
Class distribution table


🏗️ Model Architecture

Base: VGG16 (ImageNet pretrained)
Fine-tuning strategy (block5 unfrozen)
Classification head diagram or table
GAP vs Flatten comparison


🔧 Improvements Over Baseline

Table: Bug → Fix (all 9 fixes)


🎛️ Training

Hyperparameters table (lr, batch size, epochs)
Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
Augmentation strategy


📊 Results

Accuracy & Loss graphs
Final test accuracy & loss
Hardware note: trained on 4GB VRAM consumer GPU
