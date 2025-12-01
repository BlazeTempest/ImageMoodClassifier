"""
Train CNN model using Transfer Learning (MobileNetV2) - CORRECTED
Fixed preprocessing and layer freezing issues
"""
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm


def load_images_from_dataset(dataset_path='dataset', target_size=(224, 224)):
    """
    Load all images and preprocess them correctly for MobileNetV2
    """
    mood_classes = ['Calm', 'Energetic', 'Warm', 'Dark', 'Soft']
    
    images = []
    labels = []
    
    print("Loading images from dataset...")
    
    for class_idx, mood_class in enumerate(mood_classes):
        class_path = os.path.join(dataset_path, mood_class)
        
        if not os.path.exists(class_path):
            continue
        
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"Loading {len(image_files)} images from {mood_class}...")
        
        for img_file in tqdm(image_files, desc=mood_class):
            img_path = os.path.join(class_path, img_file)
            
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, target_size)
                
                # CRITICAL: Don't normalize here manually!
                # We will use MobileNetV2's preprocess_input later
                images.append(img)
                labels.append(class_idx)
                
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
    
    X = np.array(images, dtype='float32')
    y = np.array(labels)
    
    # Apply MobileNetV2 specific preprocessing (scales to [-1, 1])
    print("Applying MobileNetV2 preprocessing...")
    X = preprocess_input(X)
    
    return X, y, mood_classes


def build_improved_model(num_classes=5, input_shape=(224, 224, 3)):
    """
    Build improved Transfer Learning model
    """
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Unfreeze the last 40 layers immediately (better for small datasets)
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False
        
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),  # Increased dropout
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),  # Lower LR
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_cnn_model(dataset_path='dataset', epochs=30, batch_size=32):
    
    # Load images
    X, y, class_names = load_images_from_dataset(dataset_path)
    
    # Convert labels
    y_cat = to_categorical(y, num_classes=len(class_names))
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )
    
    # Augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Build model
    model = build_improved_model(num_classes=len(class_names))
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint('mood_classifier_cnn.h5', save_best_only=True, monitor='val_accuracy'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
    
    print("\nStarting training...")
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Save class names
    joblib.dump(class_names, 'cnn_class_names.pkl')
    
    return model, history

if __name__ == '__main__':
    train_cnn_model()
