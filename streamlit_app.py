import streamlit as st
import tensorflow as tf
import numpy as np
import nibabel as nib
import tempfile
import gdown
import os
import cv2

# Custom metric for model
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Function to download model from Google Drive
def load_model_from_drive(drive_url):
    model_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
    gdown.download(drive_url, model_file.name, quiet=False)
    model = tf.keras.models.load_model(model_file.name, custom_objects={'dice_coef': dice_coef})
    return model

# Google Drive link for the pre-trained model file (.h5)
google_drive_url = "https://drive.google.com/uc?id=1Hrgh_qnd4Ly1HvPH7d-2tluf3Y0lgCTV"  # Replace FILE_ID with your model's ID

# Load the model
st.write("Loading model from Google Drive...")
model = load_model_from_drive(google_drive_url)
if model is None:
    st.stop()
st.write("Model loaded successfully!")

# Title and instructions
st.title("2D Brain Tumor Segmentation with MRI Modalities")
st.write("Upload three MRI modalities: T1ce, T2, and FLAIR in NIfTI (.nii) format.")

# File uploads
uploaded_t1ce = st.file_uploader("Upload T1ce MRI file (.nii)", type="nii")
uploaded_t2 = st.file_uploader("Upload T2 MRI file (.nii)", type="nii")
uploaded_flair = st.file_uploader("Upload FLAIR MRI file (.nii)", type="nii")

# Process files if all are uploaded
if uploaded_t1ce and uploaded_t2 and uploaded_flair:
    # Load each NIfTI image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as temp_t1ce:
        temp_t1ce.write(uploaded_t1ce.getbuffer())
        t1ce_img = nib.load(temp_t1ce.name).get_fdata()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as temp_t2:
        temp_t2.write(uploaded_t2.getbuffer())
        t2_img = nib.load(temp_t2.name).get_fdata()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as temp_flair:
        temp_flair.write(uploaded_flair.getbuffer())
        flair_img = nib.load(temp_flair.name).get_fdata()

    # Resize volumes to (128, 128) per slice if necessary
    IMG_SIZE = 128
    t1ce_img_resized = np.stack([cv2.resize(slice, (IMG_SIZE, IMG_SIZE)) for slice in t1ce_img], axis=2)
    t2_img_resized = np.stack([cv2.resize(slice, (IMG_SIZE, IMG_SIZE)) for slice in t2_img], axis=2)
    flair_img_resized = np.stack([cv2.resize(slice, (IMG_SIZE, IMG_SIZE)) for slice in flair_img], axis=2)

    # Stack modalities along the channel dimension for 2D input slices
    combined_slices = np.stack([t1ce_img_resized, t2_img_resized, flair_img_resized], axis=-1)

    # Create an empty volume to store the 2D predictions for each slice
    prediction_volume = np.zeros((IMG_SIZE, IMG_SIZE, combined_slices.shape[2], 4))

    # Slider to select slice index
    slice_index = st.slider("Select Slice Number", 0, combined_slices.shape[2] - 1, combined_slices.shape[2] // 2)

    # Process each slice independently for prediction
    for i in range(combined_slices.shape[2]):
        input_slice = combined_slices[:, :, i, :2]  # Select first two modalities per model's input expectation
        input_slice = np.expand_dims(input_slice, axis=0)  # Add batch dimension

        # Predict segmentation mask for the slice
        prediction_slice = model.predict(input_slice)[0]
        prediction_volume[:, :, i, :] = prediction_slice  # Store prediction

    # Show the selected slice for T1ce, T2, FLAIR, and prediction
    t1ce_slice = t1ce_img_resized[:, :, slice_index]
    t2_slice = t2_img_resized[:, :, slice_index]
    flair_slice = flair_img_resized[:, :, slice_index]
    mask = np.argmax(prediction_volume[:, :, slice_index, :], axis=-1)

    # Apply color mapping for each class in mask
    color_map = {
        0: (0, 0, 0),        # Background
        1: (255, 0, 0),      # Tumor region 1
        2: (0, 255, 0),      # Tumor region 2
        3: (0, 0, 255)       # Tumor region 3
    }
    segmented_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        segmented_img[mask == class_id] = color

    # Display each image
    st.image(t1ce_slice, caption="T1ce Slice", use_column_width=True)
    st.image(t2_slice, caption="T2 Slice", use_column_width=True)
    st.image(flair_slice, caption="FLAIR Slice", use_column_width=True)
    st.image(segmented_img, caption="Predicted Segmentation Mask", use_column_width=True)

else:
    st.warning("Please upload all three modalities: T1ce, T2, and FLAIR.")
