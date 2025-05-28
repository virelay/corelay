"""A module that contains unit tests for the :py:mod:`corelay.processor.affinity` module."""

import typing

import numpy
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform

from corelay.processor.affinity import RadialBasisFunction, SparseKNN


class TestSparseKNN:
    """Contains unit tests for the :py:class:`corelay.processor.affinity.SparseKNN` class."""

    @staticmethod
    def test_sparse_knn_affinity() -> None:
        """Tests the :py:class:`corelay.processor.affinity.SparseKNN` affinity class with symmetric k-Nearest-Neighbor (kNN)."""

        affinity = SparseKNN(n_neighbors=1)
        assert affinity.n_neighbors == 1
        assert affinity.symmetric

        data_points = numpy.array([[1, 1], [1, 2], [1, 3]])
        distance_matrix: numpy.ndarray[typing.Any, typing.Any] = squareform(pdist(data_points, metric='euclidean'))
        numpy.testing.assert_array_equal(distance_matrix, numpy.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]))

        affinity_matrix: csr_matrix = affinity(distance_matrix)
        numpy.testing.assert_array_equal(affinity_matrix.toarray(), numpy.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.5], [0.0, 0.5, 0.0]]))

    @staticmethod
    def test_asymmetric_sparse_knn_affinity() -> None:
        """Tests the :py:class:`corelay.processor.affinity.SparseKNN` affinity class with asymmetric k-Nearest-Neighbor (kNN)."""

        affinity = SparseKNN(n_neighbors=1, symmetric=False)
        assert affinity.n_neighbors == 1
        assert not affinity.symmetric

        data_points = numpy.array([[1, 1], [1, 2], [1, 3]])
        distance_matrix: numpy.ndarray[typing.Any, typing.Any] = squareform(pdist(data_points, metric='euclidean'))
        numpy.testing.assert_array_equal(distance_matrix, numpy.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]))

        affinity_matrix: csr_matrix = affinity(distance_matrix)
        numpy.testing.assert_array_equal(affinity_matrix.toarray(), numpy.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))


class TestRadialBasisFunction:
    """Contains unit tests for the :py:class:`corelay.processor.affinity.RadialBasisFunction` class."""

    @staticmethod
    def test_radial_basis_function_affinity() -> None:
        """Tests the :py:class:`corelay.processor.affinity.RadialBasisFunction` affinity class."""

        affinity = RadialBasisFunction(sigma=1.0)
        assert affinity.sigma == 1.0

        data_points = numpy.array([[1, 1], [1, 2], [1, 3]])
        distance_matrix: numpy.ndarray[typing.Any, typing.Any] = squareform(pdist(data_points, metric='euclidean'))
        numpy.testing.assert_array_equal(distance_matrix, numpy.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]))

        affinity_matrix: numpy.ndarray[typing.Any, typing.Any] = affinity(distance_matrix)
        expected_affinity = numpy.exp(-distance_matrix**2 / (2 * affinity.sigma**2))
        numpy.testing.assert_array_almost_equal(affinity_matrix, expected_affinity)
