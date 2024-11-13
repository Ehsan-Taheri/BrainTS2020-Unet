import streamlit as st
import numpy as np
import nibabel as nib
import cv2
import os
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Constants
IMG_SIZE = 128
VOLUME_SLICES = 100
VOLUME_START_AT = 22
SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',
    2: 'EDEMA',
    3: 'ENHANCING'
}
COLORS = {
    1: "Oranges",
    2: "Blues",
    3: "Purples"
}

# Define custom metrics
# (use your previously defined metrics here)

# Functions
# (use your previously defined functions like load_model_from_gdrive, preprocess_image, predict here)

def plot_predictions(flair, t1ce, p, slice_num):
    core = p[:, :, :, 1]
    edema = p[:, :, :, 2]
    enhancing = p[:, :, :, 3]

    plt.figure(figsize=(24, 12))  # Larger figure for bigger images
    f, axarr = plt.subplots(1, 5, figsize=(24, 12))  # Adjust layout for 5 images

    # Original flair image
    axarr[0].imshow(cv2.resize(flair[:, :, slice_num + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")
    axarr[0].title.set_text('FLAIR Image')
    
    # T1CE image
    axarr[1].imshow(cv2.resize(t1ce[:, :, slice_num + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")
    axarr[1].title.set_text('T1CE Image')

    # All classes overlaid
    axarr[2].imshow(edema[slice_num, :, :], cmap=COLORS[2], interpolation='none', alpha=0.3)
    axarr[2].imshow(core[slice_num, :, :], cmap=COLORS[1], interpolation='none', alpha=0.3)
    axarr[2].imshow(enhancing[slice_num, :, :], cmap=COLORS[3], interpolation='none', alpha=0.3)
    axarr[2].title.set_text('All Classes')

    # Individual class predictions
    axarr[3].imshow(edema[slice_num, :, :], cmap=COLORS[2], interpolation='none', alpha=0.3)
    axarr[3].title.set_text(f'{SEGMENT_CLASSES[2]} Prediction')

    axarr[4].imshow(core[slice_num, :, :], cmap=COLORS[1], interpolation='none', alpha=0.3)
    axarr[4].title.set_text(f'{SEGMENT_CLASSES[1]} Prediction')

    return f

# Load Model
st.title("3D Medical Image Segmentation")
st.write("Upload your FLAIR and T1CE NIfTI files to predict tumor segments.")

# Google Drive URL for the model
model_url = 'https://drive.google.com/uc?id=1Hrgh_qnd4Ly1HvPH7d-2tluf3Y0lgCTV'
model_destination = 'trained_brain_mri_model.h5'

# Download model from Google Drive
# (use your previously defined code for downloading and loading the model here)

# Upload files
flair_file = st.file_uploader("Upload FLAIR NIfTI file", type=["nii"])
t1ce_file = st.file_uploader("Upload T1CE NIfTI file", type=["nii"])

if flair_file is not None and t1ce_file is not None:
    # Save uploaded files to a temporary location
    # (use your previously defined file handling code here)

    # Load NIfTI files
    try:
        flair = nib.load(flair_path).get_fdata()
        t1ce = nib.load(t1ce_path).get_fdata()
    except Exception as e:
        st.error(f"Failed to load NIfTI files: {e}")
        flair = None
        t1ce = None

    if flair is not None and t1ce is not None:
        # Preprocess images
        X = preprocess_image(flair, t1ce)
        
        # Make predictions
        try:
            p = predict(model, X)
        except Exception as e:
            st.error(f"Failed to make predictions: {e}")
            p = None

        if p is not None:
            # Select slice with a slider
            slice_num = st.slider("Select Slice", 0, VOLUME_SLICES - 1, 60)  # Default at 60

            # Plot predictions with modified layout
            fig = plot_predictions(flair, t1ce, p, slice_num)
            st.pyplot(fig)
