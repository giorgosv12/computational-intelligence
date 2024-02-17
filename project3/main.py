"""
Vellios Georgios Serafeim AEM:9471
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
from keras.layers import Flatten, Dense
import time
import matplotlib.pyplot as plt

from mlp import train_model_1
from mlpFineTuning import train_model2, fine_tune_model2


(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.astype("float32")/255
X_test = X_test.astype("float32")/255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

train_model_1(X_train, y_train, batch_size=256)

fine_tune_model2(X_train, y_train)

train_model2(X_train, y_train, X_test, y_test)

