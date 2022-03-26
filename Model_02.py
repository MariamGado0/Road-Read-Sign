import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
import cv2 as cv
#from imgaug import augmenters as iaa
import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        #gtReader.next() # skip header
        # loop over all images in current annotations file
        i=0
        for row in gtReader:
            if (i is not 0):
                images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
                labels.append(row[7]) # the 8th column is the label
            i+=1
        gtFile.close()
    return images, labels

def to_grayscale(img):
    """
    Converts an image in RGB format to grayscale
    """
    return cv.cvtColor(img, cv.COLOR_RGB2GRAY)

def imageResize(img):
    
    width = 30
    height = 30
    dim = (width, height)

    # resize image
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    return (resized)

def fcn(rootpath):
    
    data, labels = readTrafficSigns(rootpath)
    print("D",len(data)) #39209
    imgs=[]
    for indx in range(0,len(data)):
        img=imageResize(data[indx])
        img=to_grayscale(img)
        imgs.append(img)
    return imgs,labels 
   
def createModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(30,30,1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))
     
    return model

print("start")
data,labels=fcn("GTSRB\Final_Training\Images")
#(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.1, random_state=42)
print("reading data is donne")

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
data = data.reshape(data.shape[0], 30, 30 , 1).astype('float32')
labels = np_utils.to_categorical(labels)

model.add(Conv2D(40, kernel_size=5, padding="same",input_shape=(28, 28, 1), activation = 'relu'))
model.add(Conv2D(50, kernel_size=5, padding="valid", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(70, kernel_size=3, padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(100, kernel_size=3, padding="valid", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(100))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense())
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(data, labels, epochs= 32 , batch_size=200, validation_split = 0.2)
# scores = model.evaluate(X_test, y_test, verbose = 10 )
# print ( scores )

print("done")




