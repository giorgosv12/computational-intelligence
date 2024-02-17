"""

Vellios Georgios Serafeim AEM:9471

"""
from keras_tuner import RandomSearch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from rbflayer import RBFLayer
import matplotlib.pyplot as plt
from initializers import InitCentersKMeans, InitBetas
from tensorflow.keras import backend as K
from keras.datasets import boston_housing

from utils import root_mean_squared_error, mean_squared_error, coeff_determination


def fine_tune_model(X_train, y_train):
    """
    Fine tuning the model.
    """

    # Create the tuner
    tuner = RandomSearch(test_model,
                         objective="val_loss",
                         max_trials=200,
                         executions_per_trial=1,
                         directory="/C/",
                         overwrite=True)

    tuner.search_space_summary()
    tuner.search(X_train, y_train, epochs=100, batch_size=6, validation_split=0.2, verbose=True)
    tuner.results_summary()


def test_model(hp):
    """
    Creation of the model with the given hyperparameters.
    """

    n_train = 379
    unitsRBF = hp.Choice("Units_RBF", values=[int(0.05 * n_train), int(0.15 * n_train), int(0.30 * n_train),
                                              int(0.50 * n_train)])
    units2 = hp.Choice("Units_2", values=[32, 64, 128, 256])
    drop = hp.Choice("Dropout_rate", values=[0.2, 0.35, 0.5])

    (X_train, y_train), (X_test, y_test) = boston_housing.load_data()

    mean = X_train.mean(axis=0)
    X_train -= mean
    std = X_train.std(axis=0)
    X_train /= std
    X_test -= mean
    X_test /= std

    model = Sequential()
    rbflayer = RBFLayer(unitsRBF,
                        initializer=InitCentersKMeans(X_train),
                        initializer_beta=InitBetas(X_train, units2),
                        input_shape=(13,))

    model.add(rbflayer)
    model.add(Dropout(drop))
    model.add(Dense(units2))
    model.add(Dense(1))

    opt = SGD(learning_rate=0.001)
    model.compile(optimizer=opt, loss=mean_squared_error, metrics=[root_mean_squared_error])
    return model



def train_model(X_train, y_train, X_test, y_test):
    """
     Training model with the fine-tuned hyperparameters and creating graphs

    :param X_train: featues for training set
    :param y_train: target vector for training set
    :param X_test:  features for testing set
    :param y_test:  target vector for testing set
    """

    model = create_model()

    history = model.fit(X_train, y_train,
                        batch_size=6,
                        epochs=100,
                        verbose=1,
                        validation_split=0.2)

    plt.figure(1)
    plt.title("Learning Curve for Best Model")
    plt.xlabel('Epochs')
    plt.ylabel("MSE")
    a, = plt.plot(history.history["loss"])
    b, = plt.plot(history.history["val_loss"])
    plt.legend([a, b], ['Training', 'Validation'])

    plt.figure(2)
    plt.title("R2 and epochs for Best Model ")
    plt.xlabel('Epochs')
    plt.ylabel("R2")
    a, = plt.plot(history.history["coeff_determination"])
    b, = plt.plot(history.history["val_coeff_determination"])
    plt.legend([a, b], ['Training', 'Validation'])


    print("###############")
    model.evaluate(X_test, y_test)
    plt.show()


def create_model():
    """
    Creating the fine tuned model
    """

    (X_train, y_train), (X_test, y_test) = boston_housing.load_data()

    # Standardization
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train -= mean
    X_train /= std
    X_test -= mean
    X_test /= std

    model = Sequential()
    rbflayer = RBFLayer(189,
                        initializer=InitCentersKMeans(X_train),
                        initializer_beta=InitBetas(X_train, 32),
                        input_shape=(13,))

    model.add(rbflayer)
    model.add(Dropout(0.35))
    model.add(Dense(32))
    model.add(Dense(1))

    opt = SGD(learning_rate=0.001)
    model.compile(optimizer=opt, loss=mean_squared_error, metrics=[root_mean_squared_error, coeff_determination])
    return model

