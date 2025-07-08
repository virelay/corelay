"""A module that contains processors for computing affinity, i.e., similarity, matrices for sets of measurements."""

import typing
from typing import Annotated

import numpy
import scipy.sparse

from corelay.base import Param
from corelay.processor.base import Processor


class Affinity(Processor):
    """The abstract base class for processors that compute affinity (i.e., similarity) matrices.

    Note:
        Each sub-class has to implement a :py:meth:`Processor.__call__ <corelay.processor.base.Processor.__call__>` method to compute its
        corresponding affinity matrix of some data.

    Args:
        is_output (bool): A value indicating whether this :py:class:`Affinity` processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
    """


class SparseKNN(Affinity):
    """A processor for computing an affinity matrix using the sparse k-nearest neighbors (KNN) method.

    Args:
        is_output (bool): A value indicating whether this :py:class:`SparseKNN` affinity processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        n_neighbors (int): Number of neighbors to consider. Defaults to 10.
        symmetric (bool): If :py:obj:`True`, the affinity matrix is set to the mean of itself and its transpose. Defaults to :py:obj:`True`.
    """

    n_neighbors: Annotated[int, Param(int, 10, identifier=True)]
    """A parameter for the number of neighbors to consider. Defaults to 10."""

    symmetric: Annotated[bool, Param(bool, True, identifier=True)]
    """A parameter for whether to make the affinity matrix symmetric. Defaults to :py:obj:`True`."""

    def function(self, data: typing.Any) -> typing.Any:
        """Compute Sparse K-Nearest-Neighbors affinity matrix.

        Args:
            data (typing.Any): A NumPy array :py:class:`~numpy.ndarray` containing the pairwise distances between samples, which is used to compute
                the affinity matrix.

        Returns:
            typing.Any: Returns a sparse CSR representation :py:class:`~scipy.sparse.csr_matrix` of the KNN affinity matrix.
        """

        number_of_neighbors = self.n_neighbors
        number_of_samples = data.shape[0]

        # Silently uses the maximum number of neighbors if the specified number of neighbors is larger than the number of available samples
        number_of_neighbors = min(number_of_neighbors, number_of_samples - 1)

        # Sets up indices for a sparse representation of the nearest neighbors
        columns = data.argsort(1)[:, 1:number_of_neighbors + 1]
        rows = numpy.mgrid[:number_of_samples, :number_of_neighbors][0]

        # Denotes the existing edges with ones
        values = numpy.ones((number_of_samples, number_of_neighbors), dtype=data.dtype)
        affinity: typing.Any = scipy.sparse.csr_matrix((values.flat, (rows.flat, columns.flat)), shape=(number_of_samples, number_of_samples))

        # Makes the affinity matrix symmetric
        if self.symmetric:
            affinity = (affinity + affinity.T) / 2.0

        return affinity


class RadialBasisFunction(Affinity):
    """A processor for computing an affinity matrix using the Radial Basis Function (RBF) kernel.

    Args:
        is_output (bool): A value indicating whether this :py:class:`RadialBasisFunction` affinity processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        sigma (float): The scale of the RBF kernel. Defaults to 1.0.
    """

    sigma: Annotated[float, Param(float, 1.0, identifier=True)]
    """A parameter for the scale of the RBF kernel. Defaults to 1.0."""

    def function(self, data: typing.Any) -> typing.Any:
        """Compute Radial Basis Function affinity matrix.

        Args:
            data (typing.Any): A NumPy array :py:class:`~numpy.ndarray` containing the pairwise distances between samples, which is used to compute
                the affinity matrix. The data is expected to be a square matrix of shape (number_of_samples, number_of_samples).

        Returns:
            typing.Any: Returns a NumPy array :py:class:`~numpy.ndarray` containing the RBF affinity matrix.
        """

        affinity = numpy.exp(-data**2 / (2 * self.sigma**2))
        return affinity
