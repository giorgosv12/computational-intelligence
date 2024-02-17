"""

Vellios Georgios Serafeim AEM:9471

"""

from tensorflow.keras.initializers import Initializer
from sklearn.cluster import KMeans
import numpy as np
import math


class InitCentersKMeans(Initializer):
    """ Initializer for initialization of centers of RBF network
        by clustering the given data set.
    # Arguments
        X: matrix, dataset
    """

    def __init__(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]

        n_centers = shape[0]
        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)

        return km.cluster_centers_


class InitBetas(Initializer):
    """
    # Arguments
        X: matrix, dataset
        out_dim: dimension of second hidden Layer
    """

    def __init__(self, X, out_dim):
        self.X= X
        self.out_dim = out_dim


    def __call__(self, shape, dtype=None):
        n_centers = shape[0]
        km = KMeans(n_clusters=n_centers, max_iter=100, verbose=0)
        km.fit(self.X)
        centers = km.cluster_centers_
        dmax = []
        for i in range(0, centers.shape[0]):
            max_dist = 0
            for j in range(0, centers.shape[1] - 1):
                for k in range(j + 1, centers.shape[1]):
                    dist = np.linalg.norm(centers[i, j] - centers[i, k])
                    if dist > max_dist:
                        max_dist = dist
            dmax.append(max_dist)
        s = np.array(dmax) / math.sqrt(2 * self.out_dim)
        self.b= 1/(2*s*s)
        return self.b


