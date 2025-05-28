"""A module that contains unit tests for the :py:mod:`corelay.processor.laplacian` module."""

import numpy
import pytest
import scipy
from sklearn.datasets import make_blobs

from corelay.processor.affinity import SparseKNN
from corelay.processor.distance import SciPyPDist
from corelay.processor.laplacian import RandomWalkNormalLaplacian, SymmetricNormalLaplacian, a1ifmat


@pytest.mark.filterwarnings('ignore:the matrix subclass is not the recommended way to represent matrices')
def test_a1ifmat() -> None:
    """Tests the :py:func:`corelay.processor.laplacian.a1ifmat` function."""

    array = numpy.array([[1, 2], [3, 4]])
    matrix = numpy.matrix(array)

    assert numpy.array_equal(array, a1ifmat(array))
    assert numpy.array_equal(array.ravel(), a1ifmat(matrix))


def test_symmetric_normal_laplacian() -> None:
    """Tests the symmetric normal laplacian processor."""

    samples, _ = make_blobs(n_samples=100, n_features=2, centers=5)  # pylint: disable=unbalanced-tuple-unpacking

    euclidean_distance = SciPyPDist(metric='euclidean')
    distance_matrix = euclidean_distance(samples)

    sparse_knn = SparseKNN(n_neighbors=5, symmetric=True)
    affinity_matrix = sparse_knn(distance_matrix)

    manual_degree = scipy.sparse.diags(affinity_matrix.sum(axis=1).A1**-0.5, 0)
    manual_laplacian = manual_degree @ affinity_matrix @ manual_degree

    symmetric_normal_laplacian = SymmetricNormalLaplacian()
    laplacian = symmetric_normal_laplacian(affinity_matrix)

    numpy.testing.assert_array_equal(
        numpy.array(manual_laplacian.todense()),
        numpy.array(laplacian.todense())
    )


def test_symmetric_random_walk_laplacian() -> None:
    """Tests the symmetric random walk laplacian processor."""

    samples, _ = make_blobs(n_samples=100, n_features=2, centers=5)  # pylint: disable=unbalanced-tuple-unpacking

    euclidean_distance = SciPyPDist(metric='euclidean')
    distance_matrix = euclidean_distance(samples)

    sparse_knn = SparseKNN(n_neighbors=5, symmetric=True)
    affinity_matrix = sparse_knn(distance_matrix)

    manual_degree = scipy.sparse.diags(affinity_matrix.sum(axis=1).A1**-1.0, 0)
    manual_laplacian = manual_degree @ affinity_matrix

    symmetric_random_walk_laplacian = RandomWalkNormalLaplacian()
    laplacian = symmetric_random_walk_laplacian(affinity_matrix)

    numpy.testing.assert_array_equal(
        numpy.array(manual_laplacian.todense()),
        numpy.array(laplacian.todense())
    )
