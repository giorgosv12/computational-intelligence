"""

Vellios Georgios Serafeim AEM:9471

"""

from tensorflow.keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    """
    RMSE calculation
    """

    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def mean_squared_error(y_true, y_pred):
    """
    MSE calculation
    """

    return K.mean(K.square(y_pred - y_true), axis=-1)


def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))

    return 1 - SS_res / (SS_tot + K.epsilon())