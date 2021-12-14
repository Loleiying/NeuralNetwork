# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 17:25:19 2021

@author: ll8922

This script builds an Artificial Neural Network to build a image classifier using tensorflow and keras.
"""
# %% Import Dependencies
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pydot

# %% Import Data
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# %% Data Exploratory
# check a random image and its label
plt.imshow(X_train_full[1])  # view image
print(y_train_full[1])  # label

# create a data dictionary for the class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(class_names[y_train_full[1]])

# %% Data Normalization
# Since we are going to use Sigmoid activation fucntion that has a range from 0 to 1.
# The data needs to be normalized to be from 0.0 to 1.0
# Normalization: the range of the values are "normalized to be from 0.0 to 1.0".
# Standardization: the range of the values are "standardized" to measure how many standard deviation the value is from its mean.

X_train_n = X_train_full / 255.
X_test_n = X_test / 255.

# %% Train-Validation-Test Split
# Training data - used for training the model
# Validation data - used for tuning the hyperparameters and evaluate the models
# Test data - used to test the model after the model has gone through initial vetting by the validation set.

X_valid, X_train = X_train_n[:5000], X_train_n[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test_n

# %% Build a Network

# set seed
np.random.seed(42)
tf.random.set_seed(42)

# model
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))  # Input layer
model.add(keras.layers.Dense(300, activation="relu"))  # Hidden layer 1
model.add(keras.layers.Dense(100, activation="relu"))  # Hidden layer 2
model.add(keras.layers.Dense(10, activation="softmax"))  # Output layer

print(model.summary())

# need to install graphviz for the following command
# keras.utils.plot_model(model)

weights, biases = model.layers[1].get_weights()
print(weights.shape)
print(biases.shape)

# %% Compile model

model.compile(
    # y-variable is multi-class labels (i.e. 1, 2, 3, ...), use sparse_categorical_crossentropy
    # if y is probability, use categorical_crossentropy
    # if y is binary, use binary_crossentropy
    loss="sparse_categorical_crossentropy",
    optimizer="sgd",  # stochastic gradient descent
    metrics=["accuracy"])

model_history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

#%% view model
print(model_history.params)
print(model_history.history)

pd.DataFrame(model_history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

#%% evaluate model
model.evaluate(X_test, y_test)

# take the first three record from the test data as the "unseen" data
X_new = X_test[:3]

# predict the probability for each class
y_proba = model.predict(X_new)
y_proba.round(2)
print(y_proba)

# predict the class label directly
y_pred = model.predict(X_new)
y_pred_class = np.argmax(y_pred, axis=1)
print(y_pred_class) # class labels
print(np.array(class_names)[y_pred_class]) # label in words
# check the first three images
plt.imshow(X_test[0])
plt.imshow(X_test[1])
plt.imshow(X_test[2])