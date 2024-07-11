import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models
import numpy as np
import os
import matplotlib.pyplot as plt

def load_image(filepath, target_size):
    # Load and resize the image to the target size
    img = load_img(filepath, color_mode='grayscale', target_size=target_size)
    img = img_to_array(img)  # Convert the image to a numpy array
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

def load_data_from_directory(base_dir, target_size):
    categories = ['meningioma', 'notumor']
    images = []
    labels = []
    
    for label, category in enumerate(categories):
        category_dir = os.path.join(base_dir, category)
        image_paths = sorted([os.path.join(category_dir, fname) for fname in os.listdir(category_dir) if fname.endswith('.jpg')])
        
        for image_path in image_paths:
            img = load_image(image_path, target_size)
            images.append(img)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Example directories
train_dir = r'D:\Data\Brain Tumors\archive\data_dir\Training'
val_dir = r'D:\Data\Brain Tumors\archive\data_dir\Validation'
test_dir = r'D:\Data\Brain Tumors\archive\data_dir\Testing'

# Load data with specified target size
target_size = (224, 224)  # Image size that the model expects
X_train, Y_train = load_data_from_directory(train_dir, target_size)
X_val, Y_val = load_data_from_directory(val_dir, target_size)
X_test, Y_test = load_data_from_directory(test_dir, target_size)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(32).shuffle(buffer_size=100)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(32)

def create_custom_cnn(input_shape=(224, 224, 1), num_classes=2):
    model = models.Sequential()
    
    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

model = create_custom_cnn()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test accuracy: {test_accuracy:.4f}')

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
