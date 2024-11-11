

import streamlit as st
import tensorflow as tf
import numpy as np
import nibabel as nib
import cv2
from PIL import Image

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# Load model
model = tf.keras.models.load_model("/Users/ciro/Downloads/trained_brain_mri_model.h5", 
                                   custom_objects={
                                       "dice_coef": dice_coef,
                                       "precision": precision,
                                       "sensitivity": sensitivity,
                                       "specificity": specificity
                                   })

# App title
st.title("Brain Tumor Segmentation")

# File upload
uploaded_file = st.file_uploader("Upload an MRI file (in .nii format)", type=["nii"])

# Process the uploaded file
if uploaded_file is not None:
    # Load the image
    img = nib.load(uploaded_file)
    img_data = img.get_fdata()
    
    # Select a slice (or allow user to choose)
    slice_num = st.slider("Select MRI Slice", 0, img_data.shape[2] - 1, img_data.shape[2] // 2)
    slice_img = img_data[:, :, slice_num]
    
    # Preprocess for model
    processed_img = cv2.resize(slice_img, (IMG_SIZE, IMG_SIZE))
    processed_img = np.expand_dims(processed_img, axis=-1)  # Add channel dimension
    processed_img = np.expand_dims(processed_img, axis=0)   # Add batch dimension

    # Predict
    prediction = model.predict(processed_img / np.max(processed_img))

    # Display original slice
    st.image(slice_img, caption="Original MRI Slice", use_column_width=True)
    
    # Display segmented slice
    st.image(prediction[0, :, :, 1], caption="Predicted Segmentation", use_column_width=True)

