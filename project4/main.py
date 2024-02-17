"""

Vellios Georgios Serafeim AEM:9471

"""
import numpy as np
from modelTraining import train_model_with_given_params
from modelFineTuning import fine_tune_model, train_model
from keras.datasets import boston_housing


# Downloading and normalizing data
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()
mean = X_train.mean(axis=0)
X_train -= mean
std = X_train.std(axis=0)
X_train /= std
X_test -= mean
X_test /= std


np.set_printoptions(suppress=True)


train_model_with_given_params(X_train, y_train, X_test, y_test, 0.1, batch_sz=6)
train_model_with_given_params(X_train, y_train, X_test, y_test, 0.5, batch_sz=6)
train_model_with_given_params(X_train, y_train, X_test, y_test, 0.9, batch_sz=6)

fine_tune_model(X_train,y_train)

train_model(X_train, y_train, X_test, y_test)






