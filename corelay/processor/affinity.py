"""Affinity (similarity) processors.

"""
import logging

import numpy as np
from scipy import sparse as sp

from .base import Processor, Param

LOGGER = logging.getLogger(__name__)


class Affinity(Processor):
    """Base class for Affinity (Similarity) classes.

    Each subclass implements a __call__ function to compute its corresponding affinity matrix of some data.

    """


class SparseKNN(Affinity):
    """Sparse K-Nearest-Neighbors affinity

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors to consider.
    symmetrix : bool
        If `True`, Affinity matrix is set to the mean of itself and itself transposed.

    """
    n_neighbors = Param(int, 10, identifier=True)
    symmetric = Param(bool, True, identifier=True)

    def function(self, data):
        """Compute Sparse K-Nearest-Neighbors affinity matrix.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            Distance matrix used to compute affinity matrix.

        Returns
        ------
        :obj:`sp.csr_matrix`
            Sparse CSR representation of KNN affinity matrix

        """
        k = self.n_neighbors
        # number of samples
        n = data.shape[0]

        # silently use maximum number of neighbors if k is larger
        k = min(k, n - 1)

        # set up indices for sparse representation of nearest neighbors
        cols = data.argsort(1)[:, 1:k + 1]
        rows = np.mgrid[:n, :k][0]
        # existing edges are denoted with ones
        vals = np.ones((n, k), dtype=data.dtype)
        affinity = sp.csr_matrix((vals.flat, (rows.flat, cols.flat)), shape=(n, n))

        # make the affinity matrix symmetric
        if self.symmetric:
            affinity = (affinity + affinity.T) / 2.
        return affinity


class RadialBasisFunction(Affinity):
    """Radial Basis Function affinity

    Parameters
    ----------
    sigma : float
        RBF scale

    """
    sigma = Param(float, 1.0, identifier=True)

    def function(self, data):
        """Compute Radial Basis Function affinity matrix.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            Distance matrix used to compute affinity matrix.

        Returns
        ------
        :obj:`np.ndarray`
            Dense RBF affinity matrix

        """
        sigma = self.sigma
        affinity = np.exp(-data / (2 * sigma ** 2))
        return affinity
