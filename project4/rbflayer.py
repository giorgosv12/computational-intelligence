
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomUniform,  Constant
import numpy as np


class RBFLayer(Layer):
    """ Layer of Gaussian RBF units.

    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the
                    layer)
        initializer: instance of initiliazer to initialize centers
        initializer_beta: instance of initializer to initialize betas

    """

    def __init__(self, output_dim,initializer_beta, initializer=None,  **kwargs):
        self.output_dim = output_dim
        self.initializer_beta = initializer_beta
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer

        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=False)



        self.betas = self.add_weight(name='betas',  #beta = 1/(2σ^2)
                                     shape=(self.output_dim,),
                                     initializer=self.initializer_beta,
                                     trainable=False)

        super(RBFLayer, self).build(input_shape)

    def call(self, x):

        C = K.expand_dims(self.centers)
        H = K.transpose(C-K.transpose(x))
        return K.exp(-self.betas*K.sum(H**2, axis=1))


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
