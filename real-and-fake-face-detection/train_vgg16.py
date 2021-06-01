import itertools
import os
import random

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix
import cv2

# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

## I don't know why but without running this cell the below code is shown an error. 
## Running all these imports again solved it.
## Will figure out soon.


### 0. Load fake and real images from dataset directory
real = "../input/real-and-fake-face-detection/real_and_fake_face/training_real/"
fake = "../input/real-and-fake-face-detection/real_and_fake_face/training_fake/"
datadir = "../input/real-and-fake-face-detection/real_and_fake_face/"

real_path = os.listdir(real)
fake_path = os.listdir(fake)


### 1. Real Face**

def load_img(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224))
    # print(labels)
    return image[...,::-1]

plt.imshow(load_img(real + real_path[2]), cmap='gray')
plt.imshow(load_img(real + real_path[20]), cmap='gray')


### 2. Fake Face
plt.imshow(load_img(fake + fake_path[20]), cmap='gray')
plt.imshow(load_img(fake + fake_path[80]), cmap='gray')

categories = ["training_real" , "training_fake"]

for category in categories:
    path = os.path.join(datadir, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap="gray")
        # plt.imshow(img_array, cmap= plt.cm.binary)
        plt.show()
        print(category)
        break
    break


### 3. Creating Training Data
training_data = []
IMG_SIZE = 224

## This means 0 will indicate Real facial Images and 1 to Fake facial Images.

categories = ["training_real" , "training_fake"]

def create_training_data():
    for category in categories:
        path = os.path.join(datadir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_UNCHANGED)
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except:
                pass
create_training_data()

training_data = np.array(training_data)
print(training_data.shape)


### 4. Building X(Features) and Y(Labels) for train test split.
np.random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y)

print(X.shape)
print(y.shape)

print(np.unique(y, return_counts = True))
## (array([0, 1]), array([1081,  960])) This should be the answer
print(y[1:10])


### 5. Normalization 
X = X/255.0 


### 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of test_x: ",X_train.shape)
print("Shape of train_y: ",y_train.shape)
print("Shape of test_x: ",X_test.shape)
print("Shape of test_y: ",y_test.shape)

print(y_test[1:10])
print(np.unique(y_train, return_counts = True))
print(np.unique(y_test, return_counts = True))

train_x = tf.keras.utils.normalize(X_train, axis=1)
test_x = tf.keras.utils.normalize(X_test, axis=1)


### 7. Load vgg model
vgg16_model = keras.applications.vgg16.VGG16()
vgg16_model.summary()

type(vgg16_model)
## This is not a sequential model.


### 8. Customizing our model
from keras.models import Sequential

model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)
# Now, we have replicated the entire vgg16_model
# (excluding the output layer) to a new Sequential model, which we've just given the name model

for layer in model.layers:
    layer.trainable = False
    
# Next, we’ll iterate over each of the layers in our new Sequential model and set them to
# be non-trainable. This freezes the weights and other trainable parameters 
# in each layer so that they will not be updated when we pass in our images of fake and real faces.
model.add(Dense(2, activation='softmax'))
model.summary()


### 9. Compile our model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit(X_train,y_train, batch_size=20, epochs = 50, validation_split=0.1)

epochs = 50
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(epochs)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

val_loss, val_acc = model.evaluate(X_test, y_test)
print(val_loss)
print(val_acc)

predictions = model.predict(X_test)

rounded_prediction = model.predict_classes(x = X_test, batch_size=10, verbose=0)
for i in rounded_predictions[:10]:
    print(i)

print(y_test[1:10])
print(np.unique(y_test, return_counts = True))

rounded_prediction = np.array(rounded_prediction)
print(np.unique(rounded_prediction, return_counts = True))

cm = confusion_matrix(y_test,rounded_prediction)
cm_plot_labels = ['Real', 'Fake']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')