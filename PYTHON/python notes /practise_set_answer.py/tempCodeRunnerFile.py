# Step 1: Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import pandas as pd

# Step 2: Define dataset path
dataset_path = '/Users/anshulshukla/Desktop/Oral-Cancer-Detection/dataset/'

# Step 3: Image preprocessing
# Using ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Scaling pixel values and splitting dataset

# Load images from directories (cancer and non_cancer)
train_data = datagen.flow_from_directory(dataset_path, 
                                         target_size=(150, 150), 
                                         batch_size=32, 
                                         class_mode='binary', 
                                         subset='training')

# Step 4: Function to visualize images in the dataset
def display_dataset_images(data, num_images=10):
    class_labels = {v: k for k, v in data.class_indices.items()}  # Get class labels (cancer or non_cancer)
    
    # Loop over the batches of images
    images_shown = 0
    for images, labels in data:
        for i in range(len(images)):
            plt.figure(figsize=(2, 2))
            plt.imshow(images[i])
            plt.title(f'Label: {class_labels[int(labels[i])]}')
            plt.axis('off')
            plt.show()
            
            images_shown += 1
            if images_shown >= num_images:  # Display a limited number of images
                return

# Step 5: Display the images in the dataset
display_dataset_images(train_data, num_images=30)  # Adjust 'num_images' as per your need
