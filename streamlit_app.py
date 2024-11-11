import streamlit as st
import tensorflow as tf
import numpy as np
import nibabel as nib
import cv2
from PIL import Image
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

# Ground truth file upload
gt_file = st.file_uploader("Upload the ground truth segmentation file (in .nii format)", type=["nii"])

# Process the uploaded MRI and ground truth files
if uploaded_file is not None and gt_file is not None and model is not None:
    # Save the uploaded MRI and ground truth files to temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as temp_mri_file:
        temp_mri_file.write(uploaded_file.getbuffer())
        temp_mri_path = temp_mri_file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as temp_gt_file:
        temp_gt_file.write(gt_file.getbuffer())
        temp_gt_path = temp_gt_file.name

    # Load the MRI image using the temporary file path
    img = nib.load(temp_mri_path)
    img_data = img.get_fdata()
    
    # Load the ground truth image
    gt_img = nib.load(temp_gt_path)
    gt_data = gt_img.get_fdata()

    # Select a slice (or allow user to choose)
    slice_num = st.slider("Select MRI Slice", 0, img_data.shape[2] - 1, img_data.shape[2] // 2)
    slice_img = img_data[:, :, slice_num]
    gt_slice = gt_data[:, :, slice_num]

    # Print model input shape for debugging
    st.write("Model input shape:", model.input_shape)

    # Define the image size for model input (use your model's expected size)
    IMG_SIZE = 128  # Update this based on your model's expected input size
    
    # Preprocess the MRI image for model prediction
    processed_img = cv2.resize(slice_img, (IMG_SIZE, IMG_SIZE))
    
    # Add channel dimensions as needed
    # Duplicate the grayscale data to create a 2-channel input (or the number of channels your model needs)
    processed_img = np.stack([processed_img, processed_img], axis=-1)  # Create 2 channels

    processed_img = np.expand_dims(processed_img, axis=0)  # Add batch dimension

    # Normalize the image to [0, 1] range for display and prediction
    processed_img = processed_img / np.max(processed_img)  # Normalize the image

    # Predict segmentation (all classes)
    prediction = model.predict(processed_img)

    # Normalize the original slice and ground truth for display
    slice_img_display = slice_img / np.max(slice_img)  # Normalize the slice image
    gt_slice_display = gt_slice / np.max(gt_slice)  # Normalize the ground truth image

    # Convert the images to uint8 for Streamlit display
    slice_img_display = (slice_img_display * 255).astype(np.uint8)
    gt_slice_display = (gt_slice_display * 255).astype(np.uint8)

    # Generate a color map for each class (you can customize this with more colors if needed)
    num_classes = prediction.shape[-1]  # Number of classes (channels)
    color_map = [
        (0, 0, 0),       # Background: black
        (255, 0, 0),     # Class 1: red
        (0, 255, 0),     # Class 2: green
        (0, 0, 255),     # Class 3: blue
        (255, 255, 0),   # Class 4: yellow
    ]
    
slice_img_display = slice_img  # No processing needed if the slice is already in the correct format

# Post-process model prediction for segmentation visualization
predicted_class_mask = np.argmax(prediction, axis=-1)  # Get class with the highest probability per pixel

# Create the combined image for the predicted segmentation (using colors for each class)
combined_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
for i in range(num_classes):
    class_mask = (predicted_class_mask == i).astype(np.uint8)
    combined_img[class_mask == 1] = color_map[i]  # Apply color to predicted mask

# Create the combined ground truth image (apply colors as per the ground truth labels)
gt_combined_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
for i in range(num_classes):
    gt_class_mask = (gt_slice == i).astype(np.uint8)
    gt_class_mask_resized = cv2.resize(gt_class_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    gt_combined_img[gt_class_mask_resized == 1] = color_map[i]

# Display the original slice, predicted segmentation, and ground truth side by side
st.image([slice_img_display, combined_img, gt_combined_img], caption=["Original MRI Slice", "Predicted Segmentation", "Ground Truth"], use_column_width=True)
