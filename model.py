import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, BatchNormalization, Flatten, Activation, LayerNormalization, Reshape
from keras.optimizers import Adam

df = pd.read_csv('./sudoku.csv')

X = np.array(df.quizzes.map(lambda x: list(map(int, x))).to_list())
Y = np.array(df.solutions.map(lambda x: list(map(int, x))).to_list())


X = X.reshape(-1, 9, 9, 1)
Y = Y.reshape(-1, 9, 9) - 1

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

model = Sequential()

model.add(Conv2D(128, 3, activation='relu', padding='same', input_shape = (9, 9, 1)))
model.add(BatchNormalization())
model.add(Conv2D(128, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(512, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(1024, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(9, 3, activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(512))
model.add(Dense(81*9))
model.add(LayerNormalization(axis=-1))
model.add(Reshape((9, 9, 9)))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))

model.evaluate(x_test, y_test)

model.save('mindlock-model.h5')






