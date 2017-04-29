import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout


from keras.datasets import mnist

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#to run locally
# X_train = X_train[0:100]
# y_train = y_train[0:100]

# X_test = X_test[0:20]
# y_test = y_test[0:20]

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

num_classes = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#vanilla model 
vanilla_model = Sequential([
    Dense(32, input_shape=(784,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax'),
])

vanilla_model.compile("adam", "categorical_crossentropy", metrics=['accuracy'])

print('vanilla model')
vanilla_model.summary()

#model fit
history_vanilla = vanilla_model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1)

#drop out model

dropout_model = Sequential([
    Dense(32, input_shape=(784,), activation='relu'),
    Dropout(.25),
    Dense(32, activation='relu'),
    Dropout(.25),
    Dense(10, activation='softmax'),
])

dropout_model.compile("adam", "categorical_crossentropy", metrics=['accuracy'])

print('drop out model')
dropout_model.summary()

#model fit
history_dropout = dropout_model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1)

#SCORES
print('Vanilla Scores')
score = vanilla_model.evaluate(X_test, y_test, verbose=0)
print("Test loss: {:.3f}".format(score[0]))
print("Test Accuracy: {:.3f}".format(score[1]))

print('Dropout Scores')
score = dropout_model.evaluate(X_test, y_test, verbose=0)
print("Test loss: {:.3f}".format(score[0]))
print("Test Accuracy: {:.3f}".format(score[1]))

#create graphs of accuracy and loss
def vizualize(model_fit,title):
    fig, ax = plt.subplots()
    df = pd.DataFrame(model_fit.history)
    acc = df['acc'].plot(label='accuracy')
    plt.ylabel("accuracy")
    
    ax2 = ax.twinx()
    loss = ax2.plot(df['loss'],linestyle='--',label='loss')
    #loss = df['loss'].plot(linestyle='--', ax=plt.twinx(),label='loss')
    plt.ylabel("loss")

    ax2.legend(loc=1)
    ax.legend(loc=2)

    plt.title(title)
    plt.savefig(str(title)+'.png')


vizualize(history_vanilla,'Vanilla')

vizualize(history_dropout,'Dropout')



