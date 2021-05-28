###  brian tumor dataset from kagel
###  download dataset : https://www.kaggle.com/preetviradiya/brian-tumor-dataset
## if output value close to 0 then it's tumor ,if close to 1 or 1 then healthy


import os
import cv2
import glob
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize
import tensorflow as tf
from keras.utils import normalize
from skimage.io import imread, imshow
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings(action="ignore")
import keras
from keras.models import Sequential
from tensorflow.keras.layers import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# define path for tumor and healthy dataset
tumor_dir='../input/brian-tumor-dataset/Brain Tumor Data Set/Brain Tumor Data Set/Brain Tumor/'
healthy_dir='../input/brian-tumor-dataset/Brain Tumor Data Set/Brain Tumor Data Set/Healthy/'

features = []
label = []
size = 242

tumor_images = os.listdir(tumor_dir)
healthy_images = os.listdir(healthy_dir)

# read images
for i, id_ in tqdm(enumerate(tumor_images),total = len(tumor_images)):
    image = cv2.imread(tumor_dir + id_)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((size, size))
    features.append(np.array(image))
    label.append(0)


for i, id_ in tqdm(enumerate(healthy_images),total = len(healthy_images)):
    image = cv2.imread(healthy_dir + id_)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((size, size))
    features.append(np.array(image))
    label.append(1)

features = np.array(features)
label = np.array(label)

# split into train test
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.20, random_state = 0,shuffle = True)

# Normalize data between 0 & 1
X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)

#input shape
INPUT_SHAPE = (size, size, 3)

#define model and layers
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',  metrics=['accuracy'])

print(model.summary())

history = model.fit(X_train, y_train, batch_size = 64, verbose = 1, epochs = 5, validation_data=(X_test,y_test),shuffle = False )

##plot loss and val loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot accuracy and val_accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#plot test image and prediction
for i in range(10):
    img = X_test[i]
    plt.imshow(img)
    plt.show()
    input_img = np.expand_dims(img, axis=0)
    print("The prediction for this image is: ", model.predict(input_img))
    print("The actual label for this image is: ", y_test[i])

# confusion matrix
mythreshold=0.908
y_pred = (model.predict(X_test)>= mythreshold).astype(int)
cm=confusion_matrix(y_test, y_pred)
print(cm)
