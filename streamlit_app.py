import streamlit as st
import tensorflow as tf
import numpy as np
import nibabel as nib
import cv2
import tempfile
import requests
import gdown  # Make sure to add gdown to your requirements.txt
import os

# Custom metric for model
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Function to download model from Google Drive
def load_model_from_drive(drive_url):
    try:
        model_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
        gdown.download(drive_url, model_file.name, quiet=False)
        model = tf.keras.models.load_model(model_file.name, custom_objects={'dice_coef': dice_coef})
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Google Drive link for the pre-trained model file (.h5)
google_drive_url = "https://drive.google.com/uc?id=1Hrgh_qnd4Ly1HvPH7d-2tluf3Y0lgCTV"  # Replace FILE_ID with your model's ID

# Load the model
st.write("Loading model from Google Drive...")
model = load_model_from_drive(google_drive_url)
if model is None:
    st.stop()
st.write("Model loaded successfully!")

# Title and instructions
st.title("Brain Tumor Segmentation with MRI Modalities")
st.write("Upload three MRI modalities: T1ce, T2, and FLAIR in NIfTI (.nii) format.")

# File uploads
uploaded_t1ce = st.file_uploader("Upload T1ce MRI file (.nii)", type="nii")
uploaded_t2 = st.file_uploader("Upload T2 MRI file (.nii)", type="nii")
uploaded_flair = st.file_uploader("Upload FLAIR MRI file (.nii)", type="nii")

# Process files if all are uploaded
if uploaded_t1ce and uploaded_t2 and uploaded_flair:
    # Save and load each NIfTI image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as temp_t1ce:
        temp_t1ce.write(uploaded_t1ce.getbuffer())
        t1ce_img = nib.load(temp_t1ce.name).get_fdata()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as temp_t2:
        temp_t2.write(uploaded_t2.getbuffer())
        t2_img = nib.load(temp_t2.name).get_fdata()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as temp_flair:
        temp_flair.write(uploaded_flair.getbuffer())
        flair_img = nib.load(temp_flair.name).get_fdata()

    # Select a slice to visualize
    slice_num = st.slider("Select MRI Slice", 0, t1ce_img.shape[2] - 1, t1ce_img.shape[2] // 2)
    t1ce_slice = t1ce_img[:, :, slice_num]
    t2_slice = t2_img[:, :, slice_num]
    flair_slice = flair_img[:, :, slice_num]

    # Resize each slice to match model input
    IMG_SIZE = model.input_shape[1] if model.input_shape[1] is not None else 128
    t1ce_slice_resized = cv2.resize(t1ce_slice, (IMG_SIZE, IMG_SIZE))
    t2_slice_resized = cv2.resize(t2_slice, (IMG_SIZE, IMG_SIZE))
    flair_slice_resized = cv2.resize(flair_slice, (IMG_SIZE, IMG_SIZE))

    # Stack the slices to create a 3-channel input
    input_img = np.stack([t1ce_slice_resized, t2_slice_resized, flair_slice_resized], axis=-1)
    input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension
    input_img = input_img / np.max(input_img)  # Normalize

    # Prediction
    st.write("Predicting segmentation mask...")
    prediction = model.predict(input_img)[0]

    # Convert prediction to color mask
    mask = np.argmax(prediction, axis=-1)
    color_map = {
        0: (0, 0, 0),        # Background
        1: (255, 0, 0),      # Tumor region 1
        2: (0, 255, 0),      # Tumor region 2
        3: (0, 0, 255)       # Tumor region 3
    }
    segmented_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        segmented_img[mask == class_id] = color

    # Display the images
    st.image([t1ce_slice, segmented_img], caption=["T1ce Slice", "Predicted Segmentation"], use_column_width=True)
else:
    st.warning("Please upload all three modalities: T1ce, T2, and FLAIR.")
