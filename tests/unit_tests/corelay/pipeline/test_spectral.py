"""A module that contains unit tests for the :py:mod:`corelay.io.spectral` module."""

import os
import typing

import numpy
import pytest
from matplotlib import pyplot

from corelay.pipeline.spectral import SpectralEmbedding, SpectralClustering
from corelay.processor.affinity import SparseKNN
from corelay.processor.clustering import KMeans
from corelay.processor.embedding import EigenDecomposition
from corelay.processor.laplacian import SymmetricNormalLaplacian


@pytest.fixture(name='spiral_data', scope='module')
def get_spiral_data_fixture(number_of_samples_per_class: int = 150) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
    """A fixture that produces data points that has the shape of a spiral.

    Args:
        number_of_samples_per_class (int): Number of samples per class. Defaults to 150.

    Returns:
        numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: Returns a NumPy array of shape (300, 2) containing the spiral data.
    """

    # Fixes the seed for the random number generator to ensure reproducibility (100 random numbers are sampled uniformly as "part of the seed"; I do
    # not know why the original author of this unit test did this, but I assume it is to remove some bad initial data points)
    numpy.random.seed(1345123)
    _ = numpy.random.uniform(size=(2, 100)).T

    # Generates double-spiral data in 2D with N data points
    theta = numpy.sqrt(numpy.random.rand(number_of_samples_per_class)) * 2 * numpy.pi

    r_a = 2 * theta + numpy.pi
    data_a = numpy.array([numpy.cos(theta) * r_a, numpy.sin(theta) * r_a]).T
    x_a = data_a + numpy.random.randn(number_of_samples_per_class, 2) * 0.5

    r_b = -2 * theta - numpy.pi
    data_b = numpy.array([numpy.cos(theta) * r_b, numpy.sin(theta) * r_b]).T
    x_b = data_b + numpy.random.randn(number_of_samples_per_class, 2) * 0.5

    return numpy.append(x_a, x_b, axis=0)


@pytest.fixture(name='number_of_neighbors', scope='module')
def get_number_of_neighbors_fixture(spiral_data: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> int:
    """A fixture that choose a suitable number of neighbors for the k-nearest neighbors algorithm.

    Args:
        spiral_data (numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]): The spiral test data.

    Returns:
        int: Returns a suitable number of neighbors for the k-nearest neighbors algorithm.
    """

    return int(numpy.log(spiral_data.shape[0]))


@pytest.fixture(name='number_of_eigenvalues', scope='module')
def get_number_of_eigenvalues_fixture() -> int:
    """A fixtures that chooses a suitable number of eigenvalues.

    Returns:
        int: Returns a suitable number of eigenvalues.
    """

    return 8


@pytest.fixture(name='number_of_clusters', scope='module')
def get_number_of_clusters_fixture() -> int:
    """A fixture that chooses a suitable number of clusters.

    Returns:
        int: Returns a suitable number of clusters.
    """

    return 4


class TestSpectralEmbeddingAndSpectralClustering:
    """Contains unit tests for the :py:class:`~corelay.pipeline.spectral.SpectralEmbedding` and
    :py:class:`~corelay.pipeline.spectral.SpectralClustering` classes.
    """

    @staticmethod
    def test_spectral_embedding_instantiation() -> None:
        """Tests whether we can instantiate a spectral embedding instance successfully."""

        SpectralEmbedding()

    @staticmethod
    def test_spectral_clustering_instantiation() -> None:
        """Tests whether we can instantiate a spectral clustering instance successfully."""

        SpectralClustering()

    @staticmethod
    def test_data_generation(spiral_data: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> None:
        """Performs a sanity check to make sure the data looks as expected (from the outside).

        Args:
            spiral_data (numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]): The spiral test data.
        """

        assert isinstance(spiral_data, numpy.ndarray), f'Expected NumPy arrays, got {type(spiral_data)}.'
        assert spiral_data.shape == (300, 2), f'Expected the spiral test data to be of shape (300, 2), got {spiral_data.shape}.'

    @staticmethod
    def test_spectral_embedding_default_params(spiral_data: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> None:
        """Tests whether the :py:class:`~corelay.pipeline.spectral.SpectralEmbedding` operates on data all the way through, using its default
        parameters.

        Args:
            spiral_data (numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]): The spiral test data.
        """

        pipeline = SpectralEmbedding()
        output = pipeline(spiral_data)
        assert isinstance(output, tuple), f'Expected the output to be of type tuple, got {type(output)}.'
        assert len(output) == 2, f'Expected an output length of 2, got {len(output)}.'

        eigenvalues, eigenvectors = output
        assert isinstance(eigenvalues, numpy.ndarray), f'Expected the eigenvalues to be NumPy arrays, but got {type(eigenvalues)}.'
        assert isinstance(eigenvectors, numpy.ndarray), f'Expected the eigenvectors to be NumPy arrays, but got {type(eigenvectors)}.'
        assert eigenvectors.shape[0] == spiral_data.shape[0], (
            f'Expected the number of eigenvectors to be identical to number of samples ({spiral_data.shape[0]}), but got {eigenvectors.shape[0]}'
        )
        assert eigenvectors.shape[1] == eigenvalues.size, (
            f'Expected the dimensionality of the eigenvectors {eigenvectors.shape[1]} to be identical to the number of reported eigenvalues '
            f'{eigenvalues.size}.'
        )

    @staticmethod
    def test_spectral_clustering_default_params(spiral_data: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> None:
        """Tests whether the :py:class:`~corelay.pipeline.spectral.SpectralClustering` operates on data all the way through, using its default
        parameters.

        Args:
            spiral_data (numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]): The spiral test data.
        """

        pipeline = SpectralClustering()
        output = pipeline(spiral_data)
        assert isinstance(output, tuple), f'Expected the output to be of type tuple, got {type(output)}.'
        assert len(output) == 2, f'Expected an output length of 2, got {len(output)}.'

        eigenvalues_and_vectors, labels = output
        assert isinstance(eigenvalues_and_vectors, tuple), f'Expected the output to be of type tuple, got {type(eigenvalues_and_vectors)}.'
        assert len(eigenvalues_and_vectors) == 2, f'Expected the output tuple to be of length 2, got {len(eigenvalues_and_vectors)}.'

        eigenvalues, eigenvectors = eigenvalues_and_vectors
        assert isinstance(eigenvalues, numpy.ndarray), f'Expected the eigenvalues to be NumPy arrays, but got {type(eigenvalues)}.'
        assert isinstance(eigenvectors, numpy.ndarray), f'Expected the eigenvectors to be NumPy arrays, but got {type(eigenvectors)}.'
        assert eigenvectors.shape[0] == spiral_data.shape[0], (
            f'Expected the number of eigenvectors to be identical to the number of samples ({spiral_data.shape[0]}), but got {eigenvectors.shape[0]}.'
        )
        assert eigenvectors.shape[1] == eigenvalues.size, (
            f'Expected the dimensionality of the eigenvectors {eigenvectors.shape[1]} be be identical to the number of reported eigenvalues '
            f'{eigenvalues.size}.'
        )

        assert isinstance(labels, numpy.ndarray), f'Expected labels to be NumPy arrays, but got {type(labels)}'
        assert labels.size == spiral_data.shape[0], (
            'Expected the number of labels to be identical to the number of samples in the spiral test data ({spiral_data.shape[0]}), but got '
            f'{labels.size}.'
        )

        assert labels.ndim == 1, f'Expected the labels to be a flat array, but was shaped {labels.shape}.'

    @staticmethod
    def test_spectral_clustering_step_by_step_custom_params(
        spiral_data: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]],
        number_of_neighbors: int,
        number_of_eigenvalues: int,
        number_of_clusters: int,
    ) -> None:
        """This test manually compares the pipelines working order against manually executed steps with custom parameters attuned to the spiral test
        data. First, the processors are prepared separately in the same way they are set up in the custom pipeline. The parameters of the processors
        are set to the values passed to the test function. All processors are then executed one by one, and the results are later compared to the
        pipeline output.

        Args:
            spiral_data (numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]): The spiral test data.
            number_of_neighbors (int): The number of neighbors for the k-nearest neighbors algorithm.
            number_of_eigenvalues (int): The number of eigenvalues to compute.
            number_of_clusters (int): The number of clusters for the k-means algorithm.
        """

        sparse_knn = SparseKNN(n_neighbors=number_of_neighbors, symmetric=True, is_output=True)
        symmetric_normal_laplacian = SymmetricNormalLaplacian(is_output=True)

        # Generates a random fixed initialization vector for the eigenvalue decomposition
        initialization_vector = numpy.random.rand(spiral_data.shape[0])
        initialization_vector /= numpy.linalg.norm(initialization_vector, 1)
        eigen_decomposition = EigenDecomposition(
            n_eigval=number_of_eigenvalues,
            is_output=True,
            kwargs={'v0': initialization_vector}
        )

        random_state = 0
        k_means = KMeans(
            n_clusters=number_of_clusters,
            is_output=True,
            kwargs={'random_state': random_state}
        )

        pipeline = SpectralClustering(
            affinity=sparse_knn,
            laplacian=symmetric_normal_laplacian,
            embedding=eigen_decomposition,
            clustering=k_means
        )

        pipeline_output = pipeline(spiral_data)
        assert len(pipeline_output) == 4, (
            f'The length of the output was expected to be 4 (affinity, laplacian, eigenvalues, and labels), but is {len(pipeline_output)}.'
        )

        # Unpacks the pipeline results
        pipeline_affinity, pipeline_laplacian, pipeline_eigenvalues, pipeline_labels = pipeline_output

        # Produces the results manually and compares them to the pipeline output
        manual_distance = pipeline.pairwise_distance(spiral_data)
        manual_affinity = sparse_knn(manual_distance)
        numpy.testing.assert_array_equal(
            numpy.array(manual_affinity.todense()),
            numpy.array(pipeline_affinity.todense()),
            'The affinity matrix produces manually differs from the one produced by the pipeline.'
        )

        manual_laplacian = symmetric_normal_laplacian(manual_affinity)
        numpy.testing.assert_array_equal(
            numpy.array(manual_laplacian.todense()),
            numpy.array(pipeline_laplacian.todense()),
            'The laplacian produced manually differs from the one produced by the pipeline.'
        )

        manual_eigenvalues = eigen_decomposition(manual_laplacian)
        numpy.testing.assert_array_equal(
            manual_eigenvalues[0],
            pipeline_eigenvalues[0],
            'The eigenvalues produced manually differ from the one produced by the pipeline.'
        )
        numpy.testing.assert_array_equal(
            manual_eigenvalues[1],
            pipeline_eigenvalues[1],
            'The eigenvectors produced manually differ from the one produced by the pipeline.'
        )

        manual_labels = k_means(manual_eigenvalues[1])
        numpy.testing.assert_array_equal(
            manual_labels,
            pipeline_labels,
            'The labels produced manually differ from the one produced by the pipeline.'
        )

        path = '/tmp/spectral_eigenvalues.pdf'
        pyplot.figure()
        pyplot.plot(pipeline_eigenvalues[0][::-1], '.')
        pyplot.xticks([])
        pyplot.yticks([])
        pyplot.savefig(path)
        os.remove(path)

        path = '/tmp/spectral_labelling.pdf'
        pyplot.figure()
        pyplot.scatter(x=spiral_data[:, 0], y=spiral_data[:, 1], c=pipeline_labels)
        pyplot.xticks([])
        pyplot.yticks([])
        pyplot.savefig(path)
        os.remove(path)
