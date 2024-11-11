import streamlit as st
import tensorflow as tf
import numpy as np
import nibabel as nib
import cv2
from PIL import Image
import keras.backend as K
import tempfile

# Dice Coefficient Metric
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Precision Metric
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_value = true_positives / (predicted_positives + K.epsilon())
    return precision_value

# Sensitivity Metric (also known as recall)
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    sensitivity_value = true_positives / (possible_positives + K.epsilon())
    return sensitivity_value

# Specificity Metric
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    specificity_value = true_negatives / (possible_negatives + K.epsilon())
    return specificity_value

# App title
st.title("Brain Tumor Segmentation")

# File upload for model
model_file = st.file_uploader("Upload the model file", type="h5")
model = None

if model_file is not None:
    # Save uploaded model to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_model_file:
        temp_model_file.write(model_file.getbuffer())
        temp_model_path = temp_model_file.name

    # Load model
    model = tf.keras.models.load_model(temp_model_path, custom_objects={
        'dice_coef': dice_coef,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
    })
    st.write("Model loaded successfully!")

# MRI file upload
uploaded_file = st.file_uploader("Upload an MRI file (in .nii format)", type=["nii"])

# Process the uploaded MRI file
if uploaded_file is not None and model is not None:
    # Load the MRI image
    img = nib.load(uploaded_file)
    img_data = img.get_fdata()
    
    # Select a slice (or allow user to choose)
