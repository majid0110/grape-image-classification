# Importing the required libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing import image
from PIL import Image

# Set the train and validation paths
train_path = '/content/dataset/grapes images/train'
val_path = '/content/dataset/grapes images/validation'
test_path = '/content/dataset/grapes images/test'

# Image preprocessing using ImageDataGenerator
image_generator = ImageDataGenerator(rescale=1./255)
train_set = image_generator.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_set = image_generator.flow_from_directory(
    val_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_set = image_generator.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Load the pre-trained VGG16 model
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Build the final model
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(train_set.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_set,
    steps_per_epoch=train_set.samples // train_set.batch_size,
    epochs=10,
    validation_data=val_set,
    validation_steps=val_set.samples // val_set.batch_size
)

# Save the model weights
model.save_weights('/content/drive/MyDrive/model_weights.h5')
print('Model weights saved.')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_set)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

# Make predictions on new images
new_image_paths = ['/content/dataset/grapes images/test/image1.jpg', '/content/dataset/grapes images/test/image2.jpg']
for image_path in new_image_paths:
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = val_set.class_indices[predicted_class_index]

    print(f'Image: {image_path}')
    print(f'Predicted Class: {predicted_class}\n')

# Plot the accuracy and loss curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Generate the classification report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print('Classification Report:')
print(report)

# Generate the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()