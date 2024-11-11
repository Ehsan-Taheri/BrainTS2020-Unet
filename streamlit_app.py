import streamlit as st
import tensorflow as tf
import numpy as np
import nibabel as nib
import cv2
from PIL import Image
import keras.backend as K

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

model_path = "/mount/src/trained_brain_mri_model.h5"


# Function to load the model from the uploaded file
def load_uploaded_model(uploaded_model):
    model = tf.keras.models.load_model(uploaded_model, custom_objects={
        'dice_coef': dice_coef,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
    })
    return model

# Streamlit file uploader for model
uploaded_model = st.file_uploader("Upload the model file", type="h5")

if uploaded_model is not None:
    # Load the model
    model = load_uploaded_model(uploaded_model)
    st.write("Model loaded successfully!")
    
# App title
st.title("Brain Tumor Segmentation")

# Upload the MRI image and ground truth image files
uploaded_mri_file = st.file_uploader("Upload MRI file (in .nii format)", type=["nii"])
uploaded_gt_file = st.file_uploader("Upload Ground Truth file (in .nii format)", type=["nii"])

# Process the uploaded files
if uploaded_mri_file is not None and uploaded_gt_file is not None:
    # Load the MRI image and ground truth segmentation
    mri_img = nib.load(uploaded_mri_file)
    mri_img_data = mri_img.get_fdata()

    gt_img = nib.load(uploaded_gt_file)
    gt_img_data = gt_img.get_fdata()

    # Select a slice (or allow user to choose)
    slice_num = st.slider("Select MRI Slice", 0, mri_img_data.shape[2] - 1, mri_img_data.shape[2] // 2)
    slice_img = mri_img_data[:, :, slice_num]
    gt_slice = gt_img_data[:, :, slice_num]

    # Preprocess for model
    IMG_SIZE = 128  # Set appropriate size for the model input
    processed_img = cv2.resize(slice_img, (IMG_SIZE, IMG_SIZE))
    processed_img = np.expand_dims(processed_img, axis=-1)  # Add channel dimension
    processed_img = np.expand_dims(processed_img, axis=0)   # Add batch dimension

    # Predict
    prediction = model.predict(processed_img / np.max(processed_img))

    # Create a color map for visualizing the classes
    num_classes = prediction.shape[-1]  # Assuming last axis is the number of classes
    color_map = np.random.randint(0, 255, size=(num_classes, 3))  # Random colors for each class

    # Convert prediction to class mask (for visualization)
    predicted_class_mask = np.argmax(prediction, axis=-1)[0]  # Get class with max probability

    # Create the combined image for the predicted segmentation (using colors for each class)
    combined_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for i in range(num_classes):
        class_mask = (predicted_class_mask == i).astype(np.uint8)
        combined_img[class_mask == 1] = color_map[i]  # Apply color to predicted mask

    # Create the ground truth mask (gt_slice is the ground truth corresponding to the slice)
    gt_combined_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for i in range(num_classes):
        gt_class_mask = (gt_slice == i).astype(np.uint8)
        gt_class_mask_resized = cv2.resize(gt_class_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        gt_combined_img[gt_class_mask_resized == 1] = color_map[i]

    # Display the images side by side
    st.image([slice_img, combined_img, gt_combined_img], caption=["Original MRI Slice", "Predicted Segmentation", "Ground Truth"], use_column_width=True)
