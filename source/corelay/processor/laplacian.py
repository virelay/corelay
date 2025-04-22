"""A module that contains processors for computing the graph Laplacian, which is mainly used for spectral embeddings."""

from typing import Any

import numpy
import scipy.sparse
from numpy.typing import NDArray

from corelay.processor.base import Processor


def a1ifmat(matrix: NDArray[Any] | numpy.matrix[Any, Any]) -> NDArray[Any]:
    """Converts the specified matrix to a flat representation. If the matrix is already a flat NumPy array, it is returned as is.

    Args:
        matrix (NDArray[Any] | numpy.matrix[Any, Any]): The input matrix to be converted.

    Returns:
        NDArray[Any]: The converted matrix. If the input was a NumPy matrix, it is returned as a flat NumPy array. Otherwise, the input is returned as
            is.
    """

    return matrix.A1 if isinstance(matrix, numpy.matrix) else matrix


class Laplacian(Processor):
    """The abstract base class for processors that compute a graph Laplacian.

    Args:
        is_output (bool, optional): A value indicating whether this ``Laplacian`` processor is the output of a ``Pipeline``. Defaults to `False`.
        is_checkpoint (bool | None, optional): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to `False`.
        io (Storable | None, optional): An IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this
            run or in subsequent runs of the ``Pipeline``. Defaults to an instance of ``NoStorage``.
    """


class SymmetricNormalLaplacian(Laplacian):
    """A ``Processor`` that computes the normal symmetric graph Laplacian.

    Args:
        is_output (bool, optional): A value indicating whether this ``SymmetricNormalLaplacian`` Laplacian processor is the output of a ``Pipeline``.
            Defaults to `False`.
        is_checkpoint (bool | None, optional): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to `False`.
        io (Storable | None, optional): An IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this
            run or in subsequent runs of the ``Pipeline``. Defaults to an instance of ``NoStorage``.
    """

    def function(self, data: Any) -> Any:
        """Computes the symmetric normal graph Laplacian.

        Args:
            data (Any): The graph affinity/similarity matrix. This can be a NumPy array or a sparse matrix.

        Returns:
            Any: Returns the symmetric normal graph Laplacian, which is a sparse representation of the symmetric graph Laplacian matrix.
        """

        input_data: scipy.sparse.csr_matrix | NDArray[Any] = data
        degree = scipy.sparse.diags(a1ifmat(input_data.sum(1))**-0.5, 0)
        return degree @ input_data @ degree


class RandomWalkNormalLaplacian(Laplacian):
    """A ``Processor`` that computes the normal random walk graph Laplacian.

    Args:
        is_output (bool, optional): A value indicating whether this ``RandomWalkNormalLaplacian`` Laplacian processor is the output of a ``Pipeline``.
            Defaults to `False`.
        is_checkpoint (bool | None, optional): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to `False`.
        io (Storable | None, optional): An IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this
            run or in subsequent runs of the ``Pipeline``. Defaults to an instance of ``NoStorage``.
    """

    def function(self, data: Any) -> Any:
        """Computes the random walk normal graph Laplacian.

        Args:
            data (Any): The graph affinity/similarity matrix. This can be a NumPy array or a sparse matrix.

        Returns:
            Any: Returns the random walk normal graph Laplacian, which is a sparse representation of the random walk graph Laplacian matrix.
        """

        degree = scipy.sparse.diags(a1ifmat(data.sum(1))**-1., 0)
        return degree @ data
