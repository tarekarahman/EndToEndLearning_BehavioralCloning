import os
import csv
from keras.models import Sequential, Model
from keras.layers import Flatten,Dense
from keras.layers import Lambda, Cropping2D
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import cv2
from scipy import ndimage
#import matplotlib.pyplot as plt


samples = []
with open('./../../../opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None) #this is necessary to skip the first record as it contains the headings
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './../../../opt/carnd_p3/data/IMG/'+batch_sample[0].split('/')[-1]
                #center_image = cv2.imread(name)
                center_image = ndimage.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
#model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# # trim image to only see section with road
model.add(Cropping2D(cropping=((70,25),(0,0))))   
model.add(Conv2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
# model.add(Dropout(0.5))
# model.add(Activation('relu'))
model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
            steps_per_epoch=np.ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=np.ceil(len(validation_samples)/batch_size), 
            epochs=5, verbose=1)

model.save('model.h5')

# keras method to print the model summary
model.summary()
