# animals_image_classification
import os
os.environ['KAGGLE_CONFIG_DIR'] = '/content'

!kaggle datasets download -d borhanitrash/animal-image-classification-dataset

!unzip animal-image-classification-dataset.zip

!ls /content/
# Change the directory to where you unzipped the dataset
data_dir = '/content/Animals' # Replace with the actual path
categories = os.listdir(data_dir)

# List the categories (labels) available in the dataset
print("Categories: ", categories)

# Check the number of images in each category
for category in categories:
    category_path = os.path.join(data_dir, category)
    print(f"Category: {category}, Number of images: {len(os.listdir(category_path))}")

import cv2
# Load a random image from the dataset
sample_image_path = os.path.join(data_dir, categories[0], os.listdir(os.path.join(data_dir, categories[0]))[0])

# Read the image using OpenCV
image = cv2.imread(sample_image_path)

# Check if it's grayscale or RGB by checking the number of channels
if len(image.shape) == 2:
    print("The image is grayscale.")
elif len(image.shape) == 3:
    print(f"The image is RGB with {image.shape[2]} channels.")

# Check the dimensions of a sample image
print(f"The image has dimensions: {image.shape[:2]} (Height x Width)")

num_classes = len(categories)
print(f"Number of classes in the dataset: {num_classes}")

import matplotlib.pyplot as plt

# Visualize a few samples from different categories
def visualize_samples(data_dir, categories, num_images=3):
    plt.figure(figsize=(10, 5))

    for i, category in enumerate(categories[:3]):  # Visualize samples from the first 3 categories
        category_path = os.path.join(data_dir, category)
        sample_images = os.listdir(category_path)[:num_images]  # Get a few sample images

        for j, img_name in enumerate(sample_images):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for display

            plt.subplot(len(categories), num_images, i * num_images + j + 1)
            plt.imshow(img)
            plt.title(f'{category}')
            plt.axis('off')

    plt.show()

# Visualize some sample images
visualize_samples(data_dir, categories)


# For TensorFlow 2.x and Keras (integrated within TensorFlow)
!pip install tensorflow

# Install PyTorch and torchvision
!pip install torch torchvision torchaudio

import tensorflow as tf
print(tf.__version__)

import torch
print(torch.__version__)

!pip install opencv-python

!pip install matplotlib

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Training Data Augmentation and Normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,                   # Normalize pixel values to the range [0, 1]
    rotation_range=40,                # Randomly rotate images by up to 40 degrees
    width_shift_range=0.2,            # Randomly shift images horizontally by up to 20%
    height_shift_range=0.2,           # Randomly shift images vertically by up to 20%
    shear_range=0.2,                  # Randomly apply shearing transformations
    zoom_range=0.2,                   # Randomly zoom into images
    horizontal_flip=True,             # Randomly flip images horizontally
    fill_mode='nearest'               # Strategy for filling in new pixels created by transformations
)

# Test Data Normalization (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values to the range [0, 1]

# Create the training data generator
train_generator = train_datagen.flow_from_directory(
    '/content/Split_Animals/train',  # Path to the training directory
    target_size=(150, 150),          # Resize images to 150x150 pixels
    batch_size=32,                   # Number of images to yield per batch
    class_mode='categorical'         # Labels are one-hot encoded
)

# Create the test data generator
test_generator = test_datagen.flow_from_directory(
    '/content/Split_Animals/test',   # Path to the testing directory
    target_size=(150, 150),          # Resize images to 150x150 pixels
    batch_size=32,                   # Number of images to yield per batch
    class_mode='categorical'         # Labels are one-hot encoded
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(num_classes, activation='softmax')  # num_classes is the number of classes in the dataset
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

loss_function = 'categorical_crossentropy'

loss_function = 'sparse_categorical_crossentropy'

metrics = ['accuracy']

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Optimizer
    loss='categorical_crossentropy',  # Loss function (for one-hot encoded labels)
    metrics=['accuracy']  # Metrics to monitor
)
model.summary()

batch_size = 32

epochs = 20

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
# Create a model to check the output shape
dummy_input = Input(shape=(150, 150, 3))
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(dummy_input)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Flatten()(x)
# Create the model
model = Model(inputs=dummy_input, outputs=x)
# Print the shape of the output from the Flatten layer
model.summary()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Adjust num_classes
])

# Recompile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Optimizer
    loss='categorical_crossentropy',  # Loss function for multi-class classification
    metrics=['accuracy']  # Metrics to monitor
)
# Train the model
history = model.fit(
    train_generator,                  # Training data generator
    steps_per_epoch=train_generator.samples // train_generator.batch_size,  # Number of batches per epoch
    epochs=10,                        # Number of epochs
    validation_data=test_generator,   # Validation data generator
    validation_steps=test_generator.samples // test_generator.batch_size, # Number of batches for validation
    verbose=1                         # Print progress bar
)
# Model evaluation
print("Training complete")

import matplotlib.pyplot as plt
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

import matplotlib.pyplot as plt
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
# Print the evaluation results
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
# Predict the classes for the test set
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
# Get the true class labels from the test generator
y_true = test_generator.classes
# Generate a classification report
print("Classification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))
# Optional: Generate a confusion matrix for deeper insights
print("Confusion Matrix:\n")
print(confusion_matrix(y_true, y_pred_classes))

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
def plot_accuracy_loss(history):
    plt.figure(figsize=(12, 5))
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
# Visualize the accuracy and loss curves
plot_accuracy_loss(history)

# Clear the session to reset any previous model configuration
tf.keras.backend.clear_session()
# Predict on the test data generator without manually setting the steps
Y_pred = model.predict(test_generator, steps=None)
# Convert predictions to class indices
y_pred = np.argmax(Y_pred, axis=1)
# Get true labels
y_true = test_generator.classes
# Now, compute the confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred)
# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices.keys())
disp.plot(cmap=plt.cm.Blues)
plt.show()
