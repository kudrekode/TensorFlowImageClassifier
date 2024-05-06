#*MODULES:
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
pp=pprint.PrettyPrinter(indent=4)
from __future__ import absolute_import,division,print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt 
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    r'D:\Images For Archedentifier',  # Path to the folders containing the training images
    target_size=(256, 256),  # Resizing all images to 256x256 pixels
    batch_size=32,
    class_mode='categorical',  
    subset='training'  # Set as 'training' data
)

validation_generator = datagen.flow_from_directory(
    r'D:\Images For Archedentifier',  
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Set as 'validation' data
)

#*Not needed!
class_names = ['Art_Deco','Baroque','Gothic','Modernist','Victorian']
images,labels = next(train_generator)

# Showing 25 images to determine how model will interpret images.
images, labels = next(train_generator)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i], cmap=plt.cm.binary)  
    plt.xlabel(class_names[np.argmax(labels[i])])  
plt.show()

from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# The simple CNN model
model = keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')  # As only five classes specfied thus far
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',  
              metrics=['accuracy'])

model.fit(train_generator, epochs=10, validation_data=validation_generator)

#*Evaluating the model post epoch running:
print("Evaluate on validation data")
results = model.evaluate(validation_generator)
val_loss, val_acc = results
print("Validation loss:", val_loss)
print("Validation accuracy:", val_acc)

predictions = model.predict(validation_generator)
predicted_classes = np.argmax(predictions, axis=1)

true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

from sklearn.metrics import confusion_matrix, classification_report
conf_matrix = confusion_matrix(true_classes, predicted_classes)
class_report = classification_report(true_classes, predicted_classes, target_names=class_labels)

print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

#*Determining strength on one image:
predictions = model.predict(validation_generator)
predictions[0]

#*Function to look at accuracy of each individual image if called

import matplotlib.pyplot as plt
import numpy as np

def plot_image(i, predictions_array, images, true_labels):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_labels[predicted_label],
                                         100*np.max(predictions_array),
                                         class_labels[true_label]),
                                         color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks(range(5))
    plt.yticks([])
    thisplot = plt.bar(range(5), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

#*Calling function to look at 25 images and their corresponding accuracy in the model.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, images, np.argmax(labels, axis=1))
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, np.argmax(labels, axis=1))
plt.show()

