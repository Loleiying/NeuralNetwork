# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 20:04:27 2021

@author: ll8922

This script builds an ANN neural network that predicts the house pricing in California. This is a regression problem.
"""
#%% Import Packages
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# %% Import Data
housing = fetch_california_housing()
print(housing.feature_names)

#%% Train-Validation-Test
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

#%% Data Transformation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

#%% ANN model
np.random.seed(42)
tf.random.set_seed(42)

# multiple regression
model = keras.models.Sequential(
    [
     keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
     keras.layers.Dense(30, activation="relu"),
     keras.layers.Dense(1)
     ])
print(model.summary())

# Compile model
model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.SGD(lr=1e-3), # learning rate
              metrics=['mae'])

model_history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

mae_test = model.evaluate(X_test, y_test)

model_history.history

#%% plot performance
pd.DataFrame(model_history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)

plt.show()


#%% prediction
X_new = X_test[:3]
y_pred = model.predict(X_new)
print(y_pred)
print(y_test[:3])