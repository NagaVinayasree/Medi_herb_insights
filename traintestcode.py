import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import Xception
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

# Define constants
IMAGE_SIZE = (299, 299)
BATCH_SIZE = 32
NUM_CLASSES = len(os.listdir('/content/drive/MyDrive/imagedataset'))
DATA_DIR = '/content/drive/MyDrive/imagedataset'

# Define data generator for loading images from the data folder with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)  # Use 20% of data for validation

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training')  # Set to training data

validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation')  # Set to validation data

# Load the pre-trained Xception model without the top classification layer
base_model = Xception(weights='imagenet', include_top=False)

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add additional fully-connected layers for better representation
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)

# Add a classification layer with softmax activation
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# Model to be trained
model = Model(inputs=base_model.input, outputs=predictions)

# Fine-tune the last few layers of the base model
for layer in base_model.layers:
    layer.trainable = False
for layer in model.layers[-8:]:
    layer.trainable = True

# Use Adam optimizer with a lower learning rate
optimizer = Adam(learning_rate=0.0001)

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Set up callbacks for model training
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // BATCH_SIZE,
    epochs=50,
    batch_size=16,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // BATCH_SIZE,
    callbacks=[checkpoint, early_stopping, tensorboard])

# Evaluate the model on the entire validation set
val_loss, val_accuracy = model.evaluate(validation_generator, steps=validation_generator.n // BATCH_SIZE)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)