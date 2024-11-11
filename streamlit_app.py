import streamlit as st
import tensorflow as tf
import numpy as np
import nibabel as nib
import cv2
import keras.backend as K
import tempfile
import os

# Define custom metrics if needed
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Load model from uploaded file
uploaded_model = st.file_uploader("Upload the model file (.h5)", type="h5")
model = None
if uploaded_model is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_model_file:
        temp_model_file.write(uploaded_model.getbuffer())
        model = tf.keras.models.load_model(temp_model_file.name, custom_objects={'dice_coef': dice_coef})
    st.write("Model loaded successfully!")

st.title("Brain Tumor Segmentation with Multiple MRI Modalities")

# File upload for the three modalities
uploaded_t2 = st.file_uploader("Upload T2 MRI file (.nii)", type=["nii"])
uploaded_t1ce = st.file_uploader("Upload T1ce MRI file (.nii)", type=["nii"])
uploaded_flair = st.file_uploader("Upload FLAIR MRI file (.nii)", type=["nii"])

# Process files if all are uploaded
if uploaded_t2 and uploaded_t1ce and uploaded_flair and model:
    # Temporarily save the uploaded files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as temp_t2:
        temp_t2.write(uploaded_t2.getbuffer())
        t2_img = nib.load(temp_t2.name).get_fdata()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as temp_t1ce:
        temp_t1ce.write(uploaded_t1ce.getbuffer())
        t1ce_img = nib.load(temp_t1ce.name).get_fdata()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as temp_flair:
        temp_flair.write(uploaded_flair.getbuffer())
        flair_img = nib.load(temp_flair.name).get_fdata()

    # Select a slice to display
    slice_num = st.slider("Select MRI Slice", 0, t2_img.shape[2] - 1, t2_img.shape[2] // 2)
    t2_slice = t2_img[:, :, slice_num]
    t1ce_slice = t1ce_img[:, :, slice_num]
    flair_slice = flair_img[:, :, slice_num]

    # Resize and stack slices for model input
    IMG_SIZE = 128  # Update if needed
    t2_slice_resized = cv2.resize(t2_slice, (IMG_SIZE, IMG_SIZE))
    t1ce_slice_resized = cv2.resize(t1ce_slice, (IMG_SIZE, IMG_SIZE))
    flair_slice_resized = cv2.resize(flair_slice, (IMG_SIZE, IMG_SIZE))
    input_img = np.stack([t2_slice_resized, t1ce_slice_resized, flair_slice_resized], axis=-1)
    input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension

    # Normalize image
    input_img = input_img / np.max(input_img)

    # Predict segmentation
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

    # Display original and segmented images side-by-side
    st.image([t2_slice, segmented_img], caption=["Original T2 Slice", "Predicted Segmentation"], use_column_width=True)
