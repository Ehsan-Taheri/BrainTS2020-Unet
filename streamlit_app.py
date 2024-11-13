import streamlit as st
import numpy as np
import nibabel as nib
import tensorflow as tf
import cv2
import tempfile
from google_drive_downloader import GoogleDriveDownloader as gdd

# Constants
IMG_SIZE = 128
MODEL_URL = "https://drive.google.com/uc?id=1Hrgh_qnd4Ly1HvPH7d-2tluf3Y0lgCTV"  # Link to your model file

# Load model function
@st.cache_resource
def load_model():
    gdd.download_file_from_google_drive(file_id="1Hrgh_qnd4Ly1HvPH7d-2tluf3Y0lgCTV", dest_path="./model.h5", unzip=False)
    model = tf.keras.models.load_model("./model.h5", custom_objects={'dice_coef': dice_coef})
    return model

# Dice coefficient for evaluation
def dice_coef(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# Resize and normalize function
def preprocess_image(img):
    resized_img = np.array([cv2.resize(slice, (IMG_SIZE, IMG_SIZE)) for slice in img])  # Resize each slice
    normalized_img = (resized_img - np.min(resized_img)) / (np.max(resized_img) - np.min(resized_img))
    return normalized_img

# Function to load NIfTI files from uploaded files
def load_nii_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        nii_img = nib.load(tmp_file.name).get_fdata()
    return nii_img

# Main code
st.title("Brain Tumor Segmentation with MRI Images")

# Upload T1ce and T2 MRI files
t1ce_file = st.file_uploader("Upload T1ce MRI file", type=["nii", "nii.gz"])
t2_file = st.file_uploader("Upload T2 MRI file", type=["nii", "nii.gz"])

if t1ce_file and t2_file:
    # Load images
    t1ce_img = load_nii_file(t1ce_file)
    t2_img = load_nii_file(t2_file)
    
    # Preprocess images to required dimensions
    t1ce_img_resized = preprocess_image(t1ce_img)
    t2_img_resized = preprocess_image(t2_img)
    
    # Stack T1ce and T2 images along the last dimension to form input of shape (128, 128, num_slices, 2)
    combined_slices = np.stack([t1ce_img_resized, t2_img_resized], axis=-1)  # Shape: (128, 128, num_slices, 2)
    model = load_model()

    # Prediction for each slice
    prediction_volume = np.zeros((IMG_SIZE, IMG_SIZE, combined_slices.shape[2], 4))  # Output volume

    for i in range(combined_slices.shape[2]):
        input_slice = combined_slices[:, :, i, :]  # Shape: (128, 128, 2)
        input_slice = np.expand_dims(input_slice, axis=0)  # Add batch dimension to make (1, 128, 128, 2)
        
        prediction_slice = model.predict(input_slice)[0]  # Predict
        prediction_volume[:, :, i, :] = prediction_slice  # Store

    # Add a slider to navigate through slices
    slice_index = st.slider("Select Slice", 0, combined_slices.shape[2] - 1, 0)
    
    # Display images
    st.subheader("MRI Slices and Predicted Tumor Segmentation")

    col1, col2, col3 = st.columns(3)
    col1.image(t1ce_img_resized[:, :, slice_index], caption="T1ce Slice", use_column_width=True)
    col2.image(t2_img_resized[:, :, slice_index], caption="T2 Slice", use_column_width=True)
    col3.image(prediction_volume[:, :, slice_index, :], caption="Predicted Segmentation", use_column_width=True)
    
    st.success("Segmentation Completed!")
else:
    st.warning("Please upload both T1ce and T2 MRI images.")
