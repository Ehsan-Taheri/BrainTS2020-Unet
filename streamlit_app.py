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
def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    dice_scores = []
    for i in range(class_num):
        y_true_f = tf.keras.backend.flatten(y_true[:, :, :, i])
        y_pred_f = tf.keras.backend.flatten(y_pred[:, :, :, i])
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
        dice_scores.append(dice)
    return tf.keras.backend.mean(tf.keras.backend.stack(dice_scores))

def precision(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def sensitivity(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + tf.keras.backend.epsilon())

def specificity(y_true, y_pred):
    true_negatives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + tf.keras.backend.epsilon())

def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true[:, :, :, 1] * y_pred[:, :, :, 1]))
    return (2. * intersection) / (tf.keras.backend.sum(tf.keras.backend.square(y_true[:, :, :, 1])) + tf.keras.backend.sum(tf.keras.backend.square(y_pred[:, :, :, 1])) + epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true[:, :, :, 2] * y_pred[:, :, :, 2]))
    return (2. * intersection) / (tf.keras.backend.sum(tf.keras.backend.square(y_true[:, :, :, 2])) + tf.keras.backend.sum(tf.keras.backend.square(y_pred[:, :, :, 2])) + epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true[:, :, :, 3] * y_pred[:, :, :, 3]))
    return (2. * intersection) / (tf.keras.backend.sum(tf.keras.backend.square(y_true[:, :, :, 3])) + tf.keras.backend.sum(tf.keras.backend.square(y_pred[:, :, :, 3])) + epsilon)


# Functions
def load_model_from_gdrive(url, destination):
    try:
        gdown.download(url, destination, quiet=False)
        st.success(f"Model downloaded successfully to {destination}")
    except Exception as e:
        st.error(f"Failed to download model: {e}")

def preprocess_image(flair, t1ce):
    X = np.zeros((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))
    for j in range(VOLUME_SLICES):
        X[j, :, :, 0] = cv2.resize(flair[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
        X[j, :, :, 1] = cv2.resize(t1ce[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
    return X

def predict(model, X):
    return model.predict(X / np.max(X), verbose=1)


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
if not os.path.exists(model_destination):
    st.info(f"Model not found. Downloading from Google Drive...")
    load_model_from_gdrive(model_url, model_destination)

# Load the model
if os.path.exists(model_destination):
    try:
        model = load_model(model_destination, custom_objects={
            "dice_coef": dice_coef,
            "precision": precision,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "dice_coef_necrotic": dice_coef_necrotic,
            "dice_coef_edema": dice_coef_edema,
            "dice_coef_enhancing": dice_coef_enhancing
        }, compile=False)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
else:
    st.error("Model not found. Please ensure the model is downloaded correctly.")


# Upload files
flair_file = st.file_uploader("Upload FLAIR NIfTI file", type=["nii"])
t1ce_file = st.file_uploader("Upload T1CE NIfTI file", type=["nii"])

if flair_file is not None and t1ce_file is not None:
    # Save uploaded files to a temporary location
    flair_path = os.path.join("temp", flair_file.name)
    t1ce_path = os.path.join("temp", t1ce_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(flair_path, "wb") as f:
        f.write(flair_file.getbuffer())
    with open(t1ce_path, "wb") as f:
        f.write(t1ce_file.getbuffer())
        
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
