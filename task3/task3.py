import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization,Activation

import scipy.io

mat_train = scipy.io.loadmat('train_32x32.mat')
mat_test = scipy.io.loadmat('test_32x32.mat')


num_classes = 10
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)

def make_data(mat):
	X_train_raw = mat_train['X']
	y_train_raw = mat_train['y']

	X_train = X_train_raw.transpose((3, 0, 1, 2))

	y_train = []
	for i,x in enumerate(y_train_raw):
	    y_train.append(x[0])

	y_train = pd.Series(y_train).replace(10,0)
	y_train = np.asarray(y_train)

	X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
	X_train = X_train.astype('float32')
	X_train /= 255


	y_train = keras.utils.to_categorical(y_train, num_classes)

	return X_train,y_train

X_train,y_train = make_data(mat_train)
X_test,y_test = make_data(mat_test)


#CNN MODEL
cnn = Sequential()
cnn.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten())
cnn.add(Dense(64, activation='relu'))
cnn.add(Dense(num_classes, activation='softmax'))

cnn.summary()
cnn.compile("adam", "categorical_crossentropy", metrics=['accuracy'])

history_cnn = cnn.fit(X_train, y_train,
                      batch_size=128, epochs=20, verbose=1)

print('Base Model')
score_base = cnn.evaluate(X_test, y_test, verbose=0)
print("Test loss: {:.3f}".format(score_base[0]))
print("Test Accuracy: {:.3f}".format(score_base[1]))

#Model with batch normalization

cnn32_bn = Sequential()
cnn32_bn.add(Conv2D(64, kernel_size=(3, 3),
                 input_shape=input_shape))
cnn32_bn.add(Activation("relu"))
cnn32_bn.add(BatchNormalization())
cnn32_bn.add(MaxPooling2D(pool_size=(2, 2)))
cnn32_bn.add(Conv2D(64, (3, 3)))
cnn32_bn.add(Activation("relu"))
cnn32_bn.add(BatchNormalization())
cnn32_bn.add(MaxPooling2D(pool_size=(2, 2)))
cnn32_bn.add(Flatten())
cnn32_bn.add(Dense(64, activation='relu'))
cnn32_bn.add(Dense(num_classes, activation='softmax'))

cnn32_bn.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history_cnn_32_bn = cnn32_bn.fit(X_train, y_train,
                                 batch_size=128, epochs=10, verbose=1)

print('Batch Normalization Model')
score_batch = cnn32_bn.evaluate(X_test, y_test, verbose=0)
print("Test loss: {:.3f}".format(score_batch[0]))
print("Test Accuracy: {:.3f}".format(score_batch[1]))
