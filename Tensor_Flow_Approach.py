import keras
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
import keras.utils
from keras.utils import to_categorical

# Importing the dataset
train_data= pd.read_csv("train.csv")
X_test= pd.read_csv("test.csv")
train_data.shape, X_test.shape

y_train= train_data['label'].values
X_train = train_data.drop(labels=['label'], axis=1)
X_train.shape, X_test.shape

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

num_classes= 10
y_train = to_categorical(y_train, num_classes)

X_train /= 255
X_test /= 255

X_train = X_train.values.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.values.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, activation="relu", padding='same',input_shape=input_shape))
model.add(MaxPool2D())
model.add(Conv2D(filters=64, kernel_size=3, activation="relu", padding='same'))
model.add(MaxPool2D())
model.add(Conv2D(filters=128, kernel_size=3, activation="relu", padding='same'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(units=256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=num_classes, activation="softmax"))


model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=20,
    verbose=1,
)