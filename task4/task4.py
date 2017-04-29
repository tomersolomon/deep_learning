import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale, StandardScaler

import tensorflow as tf


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation
import keras 

from keras.datasets import mnist

import matplotlib.image as mpimg
from keras.preprocessing import image

from keras import applications
from keras.applications.vgg16 import preprocess_input

import os

from sklearn.linear_model import LogisticRegressionCV

#create url list

url_complete = []

for file in os.listdir('./Images/'):
    if 'jpg' in file:
        url_complete.append(os.path.join('Images/',file))
       
#for running 
url_complete_local = url_complete[1:3]
url_complete_local

#create X matrix
images = [image.load_img(url, target_size=(224, 224))
                 for url in url_complete_local]

X = np.array([image.img_to_array(img) for img in images])

#drop numbers

num_classes = 37

dropped = []

#CHANGE URL COMPLETE LOCAL

#create y matrix
for url in url_complete_local:
    dropped.append('_'.join(map(str, url.rsplit('_')[:-1])) )
    
numbers = pd.Series(dropped,dtype='category').values.codes

y = keras.utils.to_categorical(numbers, num_classes)

#load in model
pet_model = applications.VGG16(include_top=False,
                           weights='imagenet')

pet_model.summary()

#preprocess

X_pre = preprocess_input(X)
features = pet_model.predict(X_pre)
features_ = features.reshape(X.shape[0], -1)

#run model
X_train, X_test, y_train, y_test = train_test_split(features_, y, stratify=y)
lr = LogisticRegressionCV().fit(X_train, y_train)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))


