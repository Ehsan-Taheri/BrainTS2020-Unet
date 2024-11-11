import streamlit as st
import tensorflow as tf
import numpy as np
import nibabel as nib
import cv2
import keras.backend as K
import tempfile

# Custom metrics (same as before)
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_value = true_positives / (predicted_positives + K.epsilon())
    return precision_value

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    sensitivity_value = true_positives / (possible_positives + K.epsilon())
    return sensitivity_value

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
    # Save the uploaded MRI file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as temp_mri_file:
        temp_mri_file.write(uploaded_file.getbuffer())
        temp_mri_path = temp_mri_file.name

    # Load the MRI image using the temporary file path
    img = nib.load(temp_mri_path)
    img_data = img.get_fdata()
    
    # Select a slice (or allow user to choose)
    slice_num = st.slider("Select MRI Slice", 0, img_data.shape[2] - 1, img_data.shape[2] // 2)
    slice_img = img_data[:, :, slice_num]
    
    # Print model input shape for debugging
    st.write("Model input shape:", model.input_shape)

    # Define the image size for model input (use your model's expected size)
    IMG_SIZE = 128  # Update this based on your model's expected input size
    
    # Preprocess for model
    processed_img = cv2.resize(slice_img, (IMG_SIZE, IMG_SIZE))
    
    # Add channel dimensions as needed
    # Duplicate the grayscale data to create a 2-channel input
    processed_img = np.stack([processed_img, processed_img], axis=-1)  # Create 2 channels

    processed_img = np.expand_dims(processed_img, axis=0)  # Add batch dimension

    # Normalize and predict
    processed_img = processed_img / np.max(processed_img)  # Normalize the image
    prediction = model.predict(processed_img)

    # Display original slice
    st.image(slice_img, caption="Original MRI Slice", use_column_width=True)
    
    # Display segmented slice
    st.image(prediction[0, :, :, 1], caption="Predicted Segmentation", use_column_width=True)
