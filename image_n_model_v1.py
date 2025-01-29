# pip install tensorflow==2.10.0
# pip install keras==2.10.0
# pip install pandas numpy scikit-learn pillow matplotlib

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Updated imports for image processing
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array  # Updated import
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

# Paths
dataset_dir = "dataset"
images_dir = os.path.join(dataset_dir, "Images")
train_labels_file = os.path.join(dataset_dir, "train.csv")
test_labels_file = os.path.join(dataset_dir, "test.csv")

# Load and clean the training data
df_train = pd.read_csv(train_labels_file)
df_train['Label'] = df_train['Label'].str.strip().str.lower()

# Create class indices
class_names = sorted(df_train['Label'].unique())
class_indices = {name: i for i, name in enumerate(class_names)}

# Convert labels to numeric values and ensure they're integers
df_train['label'] = df_train['Label'].map(class_indices)
df_train['label'] = df_train['label'].astype('int32')

# Prepare image paths
image_paths_train = df_train['img_IDS'].apply(lambda x: os.path.join(images_dir, x)).values

def load_and_preprocess_image(image_path, img_size=(224, 224)):
    if not image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path += '.jpg'
    
    if os.path.exists(image_path):
        try:
            # Using updated image loading functions
            img = load_img(image_path, target_size=img_size)
            img_array = img_to_array(img)
            return img_array / 255.0
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None
    else:
        print(f"Warning: {image_path} not found.")
        return None

# Load and preprocess images
print("Loading and preprocessing images...")
X_train_raw = [load_and_preprocess_image(img_path) for img_path in image_paths_train]
valid_indices = [i for i, x in enumerate(X_train_raw) if x is not None]
X_train = np.array([X_train_raw[i] for i in valid_indices])
labels_train = np.array(df_train['label'].values[valid_indices])

# Split the data
X_train, X_val, y_train, y_val = train_test_split(
    X_train, 
    labels_train,
    test_size=0.2, 
    random_state=42,
    stratify=labels_train
)

# Print shapes for debugging
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"Number of classes: {len(class_names)}")
print(f"Sample of y_train values: {y_train[:5]}")

# Define the model using explicit tensorflow.keras imports
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(class_names), activation="softmax")
])

# Compile with explicit loss function
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# Train the model
print("Starting model training...")
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_val, y_val),
    batch_size=32
)

# Save the model
model.save("gesture_model.h5")
print("Model saved as gesture_model.h5")

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()