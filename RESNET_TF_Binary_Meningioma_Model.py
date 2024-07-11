import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# Example directories
train_dir = r"D:\Data\Brain Tumors\archive\data_dir\Training"
val_dir = r"D:\Data\Brain Tumors\archive\data_dir\Validation"
test_dir = r"D:\Data\Brain Tumors\archive\data_dir\Testing"

# Define image dimensions and batch size
img_height, img_width = 224, 224
batch_size = 32
num_classes = 2  # Adjust based on your dataset's number of classes

# Create data generators with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    color_mode='grayscale',  # Load images as grayscale
    batch_size=batch_size,
    class_mode='categorical',  # Automatically detects labels based on directory structure
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Load the pre-trained ResNet50 model with RGB input
pretrained_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_height, img_width, 3))

# Create a new ResNet50 model with grayscale input
new_model = ResNet50(weights=None, include_top=False, pooling='avg', input_shape=(img_height, img_width, 1))

# Copy the weights from the pre-trained model to the new model
layers = [l for l in new_model.layers]

for i in range(1, len(layers)):
    old_weights = pretrained_model.layers[i].get_weights()
    if len(old_weights) > 0:  # Only copy weights if the layer has weights
        if i == 2:  # The first convolutional layer (conv1_conv)
            new_weights = old_weights[0].mean(axis=2, keepdims=True)  # Average the RGB weights to get grayscale weights
            new_biases = old_weights[1]
            layers[i].set_weights([new_weights, new_biases])
        else:
            layers[i].set_weights(old_weights)

# Add new top layers for classification
x = new_model.output
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=new_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Calculate steps_per_epoch and validation_steps based on the number of samples
steps_per_epoch = train_generator.samples // batch_size
validation_steps = val_generator.samples // batch_size

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=5,
    validation_data=val_generator,
    validation_steps=validation_steps
)

# Evaluate the model on test data
test_generator = val_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_loss, test_accuracy = model.evaluate(test_generator)
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

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()
