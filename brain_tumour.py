# ============================================================
# Brain Tumor MRI Image Classification
# Single-file Streamlit App & Model Training
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import ResNet50, MobileNetV2
import streamlit as st
from PIL import Image

# ================== USER CONFIG ==================
DATASET_PATH = "C:/guvi_project/Tumour_project/brain"  # Replace with your dataset folder path
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
MODEL_SAVE_PATH = "best_brain_tumor_model.h5"
# ==================================================

# ================== STREAMLIT UI ==================
st.set_page_config(page_title="Brain Tumor MRI Classification", layout="wide")
st.title("🧠 Brain Tumor MRI Image Classification")
st.markdown("""
This application allows you to classify **brain MRI images** into tumor categories using deep learning.
You can either **train models** or **upload an image** for real-time prediction.
""")

# Sidebar for options
st.sidebar.title("Options")
app_mode = st.sidebar.selectbox("Choose the mode", ["Train Models", "Predict MRI Image"])

# ================== DATA PREPROCESSING ==================
def load_data(dataset_path, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_gen, val_gen

# ================== CUSTOM CNN MODEL ==================
def build_custom_cnn(input_shape=(224,224,3), num_classes=4):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ================== TRANSFER LEARNING MODEL ==================
def build_transfer_model(base_model_name="ResNet50", input_shape=(224,224,3), num_classes=4):
    if base_model_name == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    elif base_model_name == "MobileNetV2":
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    else:
        raise ValueError("Unsupported model")

    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        base_model,
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ================== TRAINING FUNCTION ==================
def train_model(model, train_gen, val_gen, epochs=EPOCHS, model_path=MODEL_SAVE_PATH):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
    ]
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )
    return history

# ================== VISUALIZATION ==================
def plot_history(history):
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    # Accuracy
    ax[0].plot(history.history['accuracy'], label='train_accuracy')
    ax[0].plot(history.history['val_accuracy'], label='val_accuracy')
    ax[0].set_title("Accuracy")
    ax[0].legend()
    # Loss
    ax[1].plot(history.history['loss'], label='train_loss')
    ax[1].plot(history.history['val_loss'], label='val_loss')
    ax[1].set_title("Loss")
    ax[1].legend()
    st.pyplot(fig)

def plot_confusion(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)

# ================== PREDICTION FUNCTION ==================
def predict_image(model, img, class_labels):
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0]
    class_idx = np.argmax(pred)
    return class_labels[class_idx], np.max(pred)

# ================== APP LOGIC ==================
if app_mode == "Train Models":
    st.header("Train Deep Learning Models")
    st.info("This will train a custom CNN and a pretrained transfer learning model.")
    if st.button("Start Training"):
        with st.spinner("Loading data..."):
            train_gen, val_gen = load_data(DATASET_PATH)
            class_labels = list(train_gen.class_indices.keys())
        
        st.success(f"Found classes: {class_labels}")

        st.info("Training Custom CNN...")
        cnn_model = build_custom_cnn(num_classes=len(class_labels))
        cnn_history = train_model(cnn_model, train_gen, val_gen)
        st.success("Custom CNN training complete!")
        plot_history(cnn_history)
        
        st.info("Training Transfer Learning Model (ResNet50)...")
        tl_model = build_transfer_model("ResNet50", num_classes=len(class_labels))
        tl_history = train_model(tl_model, train_gen, val_gen, model_path="best_tl_model.h5")
        st.success("Transfer Learning model training complete!")
        plot_history(tl_history)
        
        st.success("All models trained successfully!")

elif app_mode == "Predict MRI Image":
    st.header("Upload Brain MRI Image for Prediction")
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded MRI Image', use_column_width=True)
        st.write("")
        st.write("Predicting tumor type...")

        # Load pretrained model
        if os.path.exists(MODEL_SAVE_PATH):
            model = load_model(MODEL_SAVE_PATH)
            class_labels = os.listdir(os.path.join(DATASET_PATH))
            predicted_class, confidence = predict_image(model, img, class_labels)
            st.success(f"Predicted Tumor Type: **{predicted_class}**")
            st.info(f"Confidence: {confidence*100:.2f}%")
        else:
            st.error("Model not found. Please train the model first.")

# ================== END OF FILE ==================
