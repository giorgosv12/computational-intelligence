"""

Vellios Georgios Serafeim AEM:9471

"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from rbflayer import RBFLayer
import matplotlib.pyplot as plt
from initializers import InitCentersKMeans, InitBetas
from tensorflow.keras import backend as K

from utils import root_mean_squared_error, mean_squared_error, coeff_determination


def train_model_with_given_params(X_train, y_train, X_test, y_test, perc_nodes_rbf, batch_sz):
    """
    Training model and creating graphs

    :param X_train:        featues for training set
    :param y_train:        target vector for training set
    :param X_test:         features for testing set
    :param y_test:         target vector for testing set
    :param perc_nodes_rbf: number of nodes for rbf layer given by  percentage of training data size
    :param batch_sz:       batch size for training
    """


    out_dim = 128
    perc = perc_nodes_rbf
    n_nodes_rbf = int(perc * X_train.shape[0])

    # Creating the model
    model = Sequential()
    rbflayer = RBFLayer(n_nodes_rbf,
                        initializer=InitCentersKMeans(X_train),
                        initializer_beta=InitBetas(X_train, out_dim),
                        input_shape=(13,))

    model.add(rbflayer)
    model.add(Dense(out_dim))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error',
                  optimizer=SGD(learning_rate=0.001),
                  metrics=[root_mean_squared_error, coeff_determination])


    # training model
    history = model.fit(X_train, y_train,
                        batch_size=batch_sz,
                        epochs=100,
                        verbose=1,
                        validation_split=0.2)


    plt.figure(1)
    plt.title("Learning Curve for rbf n_nodes={}% of Training".format(100 * perc))
    plt.xlabel('Epochs')
    plt.ylabel("MSE")
    a, = plt.plot(history.history["loss"])
    b, = plt.plot(history.history["val_loss"])
    plt.legend([a, b], ['Training', 'Validation'])

    plt.figure(2)
    plt.title("R2 and epochs for rbf n_nodes={}% of Training".format(100 * perc))
    plt.xlabel('Epochs')
    plt.ylabel("R2")
    a, = plt.plot(history.history["coeff_determination"])
    b, = plt.plot(history.history["val_coeff_determination"])
    plt.legend([a, b], ['Training', 'Validation'])


    model.evaluate(X_test, y_test, batch_size=6)
    plt.show()