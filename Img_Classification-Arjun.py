import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters
img_size = (64, 64)
batch_size = 32
epochs = 1000
train_dir = r"C:\ML_Project\train"
val_dir = r"C:\ML_Project\val"
val_labels_file = r"C:\ML_Project\val\labels.txt"

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,  
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Map Hindi labels to numeric indices
def create_label_map(labels_file):
    unique_labels = set()
    with open(labels_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            _, label = line.strip().split("\t")
            unique_labels.add(label)
    label_to_index = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    return label_to_index

label_to_index = create_label_map(val_labels_file)
index_to_label = {v: k for k, v in label_to_index.items()}

# Load validation data
def load_validation_data(val_dir, val_labels_file, img_size, label_to_index):
    val_images, val_labels = [], []
    with open(val_labels_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
    for line in lines:
        filename, label = line.strip().split("\t")
        if label in label_to_index:
            img_path = os.path.join(val_dir, filename)
            if os.path.exists(img_path):
                img = tf.keras.utils.load_img(img_path, target_size=img_size)
                img_array = tf.keras.utils.img_to_array(img)
                img_array = img_array / 255.0
                val_images.append(img_array)
                val_labels.append(tf.keras.utils.to_categorical(label_to_index[label], len(label_to_index)))
    val_images = np.array(val_images, dtype=np.float32)
    val_labels = np.array(val_labels, dtype=np.float32)
    return val_images, val_labels

val_images, val_labels = load_validation_data(val_dir, val_labels_file, img_size, label_to_index)

# CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.3),  

    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    MaxPooling2D((2, 2)),
    Dropout(0.5),

    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(len(label_to_index), activation='softmax')
])

cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Learning Rate Scheduler
def scheduler(epoch, lr):
    if epoch % 100 == 0 and epoch > 0:
        return lr * 0.1
    return lr

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Train the model
history = cnn_model.fit(
    train_generator,
    epochs=epochs,
    validation_data=(val_images, val_labels),
    callbacks=[lr_scheduler]
)

# Plot learning curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Evaluate model
val_preds = cnn_model.predict(val_images)
val_preds_classes = np.argmax(val_preds, axis=1)
val_labels_classes = np.argmax(val_labels, axis=1)

# Metrics calculation
accuracy = accuracy_score(val_labels_classes, val_preds_classes)
precision = precision_score(val_labels_classes, val_preds_classes, average='weighted')
recall = recall_score(val_labels_classes, val_preds_classes, average='weighted')
f1 = f1_score(val_labels_classes, val_preds_classes, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(val_labels_classes, val_preds_classes, target_names=list(index_to_label.values())))
