import matplotlib.pyplot as plt
# from keras import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ReLU, Softmax
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import math

warnings.filterwarnings("ignore")
input_shape = (5, 5, 1)
from sklearn import metrics

import tensorflow as tf

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# %matplotlib inline # Only use this if using iPython

# %%
df = pd.read_csv("./hybrid.csv")
x = df.iloc[0:10, 0:25]

min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)

y = df['Label'].astype('category')
y = y[0:10]

y = y.cat.codes
y = y.values
y = np.array(y)

# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)
x_train = x_train.reshape(-1, 5, 5, 1)  # (64,64,1)
x_test = x_test.reshape(-1, 5, 5, 1)  # (64,64,1)
print(y_train)
# y_train = y_train.reshape(-1,1)    #(64,64,1)
print("x_train shape : ", x_train.shape)
print("y_train shape : ", y_train.shape)
# %%
# Importing the required Keras modules containing model and layers


# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=ReLU()))
model.add(Dropout(0.2))
model.add(Dense(27, activation=Softmax()))
# %%
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
# %%
model.fit(x=x_train, y=y_train, epochs=10)
# %%
model.evaluate(x_test, y_test)

print("Ok")
