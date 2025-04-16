"""Graph Laplacian Processors (mainly for spectral embedding)

"""
import logging

import numpy as np
from scipy import sparse as sp

from .base import Processor

LOGGER = logging.getLogger(__name__)


def a1ifmat(x):
    """Return flat representation of x if x is a :obj:`numpy.matrix`

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`numpy.matrix`
        Object to convert if necessary

    Returns
    -------
    :obj:`numpy.ndarray`
        Matrix as flat :obj:`numpy.ndarray` if `x` was a :obj:`numpy.matrix`, else `x`

    """
    return x.A1 if isinstance(x, np.matrix) else x


class Laplacian(Processor):
    """Graph Laplacian Processor

    """


class SymmetricNormalLaplacian(Laplacian):
    """ Normal Symmetric Graph Laplacian

    """
    def function(self, data):
        """Normalized Symmetric Graph Laplacian

        Parameters
        ----------
        data : :obj:`sp.csr_matrix` or :obj:`np.ndarray`
            Graph affinity/similarity matrix.

        Returns
        -------
        :obj:`sp.csr_matrix`
            Sparse representation of a symmetric graph laplacian matrix

        """
        deg = sp.diags(a1ifmat(data.sum(1))**-.5, 0)
        lap = deg @ data @ deg
        return lap


class RandomWalkNormalLaplacian(Laplacian):
    """ Normal Random Walk Graph Laplacian

    """
    def function(self, data):
        """Normalized Random Walk Graph Laplacian

        Parameters
        ----------
        affinity : :obj:`sp.csr_matrix` or :obj:`np.ndarray`
            Graph affinity/similarity matrix.

        Returns
        -------
        :obj:`sp.csr_matrix`
            Sparse representation of a random walk graph laplacian matrix

        """
        deg = sp.diags(a1ifmat(data.sum(1))**-1., 0)
        lap = deg @ data
        return lap
