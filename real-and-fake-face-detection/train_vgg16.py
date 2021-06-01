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
# from sklearn.metrics import confusion_matrix
import itertools
import os
import cv2
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

## I don't know why but without running this cell the below code is shown an error. 
## Running all these imports again solved it.
## Will figure out soon.

vgg16_model = keras.applications.vgg16.VGG16()
vgg16_model.summary()