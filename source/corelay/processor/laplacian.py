"""A module that contains processors for computing the graph Laplacian, which is mainly used for spectral embeddings."""

import typing

import numpy
import scipy.sparse

from corelay.processor.base import Processor


def a1ifmat(matrix: numpy.ndarray[typing.Any, typing.Any] | numpy.matrix[typing.Any, typing.Any]) -> numpy.ndarray[typing.Any, typing.Any]:
    """Converts the specified matrix to a flat representation. If the matrix is already a flat NumPy array, it is returned as is.

    Args:
        matrix (numpy.ndarray[typing.Any, typing.Any] | numpy.matrix[typing.Any, typing.Any]): The input matrix to be converted.

    Returns:
        numpy.ndarray[typing.Any, typing.Any]: Returns the converted matrix. If the input was a NumPy matrix, it is returned as a flat NumPy array.
        Otherwise, the input is returned as is.
    """

    return matrix.A1 if isinstance(matrix, numpy.matrix) else matrix


class Laplacian(Processor):
    """The abstract base class for processors that compute a graph Laplacian.

    Args:
        is_output (bool): A value indicating whether this :py:class:`Laplacian` processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
    """


class SymmetricNormalLaplacian(Laplacian):
    """A :py:class:`~corelay.processor.base.Processor` that computes the normal symmetric graph Laplacian.

    Args:
        is_output (bool): A value indicating whether this :py:class:`SymmetricNormalLaplacian` Laplacian processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
    """

    def function(self, data: typing.Any) -> typing.Any:
        """Computes the symmetric normal graph Laplacian.

        Args:
            data (typing.Any): The graph affinity/similarity matrix. This can be a NumPy array or a sparse matrix.

        Returns:
            typing.Any: Returns the symmetric normal graph Laplacian, which is a sparse representation of the symmetric graph Laplacian matrix.
        """

        input_data: scipy.sparse.csr_matrix | numpy.ndarray[typing.Any, typing.Any] = data
        degree = scipy.sparse.diags(a1ifmat(input_data.sum(1))**-0.5, 0)
        return degree @ input_data @ degree


class RandomWalkNormalLaplacian(Laplacian):
    """A :py:class:`~corelay.processor.base.Processor` that computes the normal random walk graph Laplacian.

    Args:
        is_output (bool): A value indicating whether this :py:class:`RandomWalkNormalLaplacian` Laplacian processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
    """

    def function(self, data: typing.Any) -> typing.Any:
        """Computes the random walk normal graph Laplacian.

        Args:
            data (typing.Any): The graph affinity/similarity matrix. This can be a NumPy array or a sparse matrix.

        Returns:
            typing.Any: Returns the random walk normal graph Laplacian, which is a sparse representation of the random walk graph Laplacian matrix.
        """

        degree = scipy.sparse.diags(a1ifmat(data.sum(1))**-1., 0)
        return degree @ data
