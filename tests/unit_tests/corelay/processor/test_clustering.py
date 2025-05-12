"""A module that contains unit tests for the :py:mod:`corelay.processor.clustering` module."""

import os
import typing
from importlib import import_module

import numpy
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances

from corelay.base import Param
from corelay.processor import clustering

EXTRA_PROCESSORS = []
"""Contains a list of extra clustering processors that are to be tested. The list of extra clustering processors depends on the availability of the
respective libraries, which can optionally be installed by the user. If the library is not installed, the processor will not be available and can
therefore not be tested.
"""

try:
    import_module('hdbscan')
    EXTRA_PROCESSORS = [clustering.HDBSCAN]
except ImportError:
    pass


@pytest.fixture(name='data', scope='module')
def get_data_fixture() -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
    """A fixture that produces tests data with 1000 elements that are split into 5 blobs.

    Returns:
        numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: Returns the data, which is a 2D array of shape `(1000, 2)`, i.e., 1000 samples with 2
        features.
    """

    blobs = make_blobs(1000, centers=5, random_state=100)
    blob: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] = blobs[0]
    return blob


@pytest.fixture(name='tiny_data', scope='module')
def get_tiny_data_fixture() -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
    """A fixtures that produces tiny test data with 50 elements that are split into 5 blobs.

    Returns:
        numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: Returns the data, which is a 2D array of shape `(50, 2)`, i.e., 50 samples with 2
        features.
    """

    blobs = make_blobs(50, centers=5, random_state=100)
    blob: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] = blobs[0]
    return blob


@pytest.fixture(name='distances', scope='module')
def get_distances_fixture(data: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
    """A fixtures that computes the euclidean distances between each pair of data points.

    Args:
        data (numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]): The data to compute the distances for.

    Returns:
        numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: Returns a distance matrix, which, given input data of shape
        `(<number_of_samples>, <number_of_features>)`, is a 2D array of shape `(<number_of_samples>, <number_of_samples>)`, i.e., the distances
        between each pair of samples.
    """

    distance_matrix: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] = euclidean_distances(data)
    return distance_matrix


@pytest.mark.parametrize('processor_type', [clustering.AgglomerativeClustering, clustering.KMeans])
def test_clustering(processor_type: type[clustering.Clustering], data: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> None:
    """Tests the clustering processors by checking that the 5 found clusters are approximately of similar size.

    Args:
        processor_type (type[clustering.Clustering]): The clustering :py:class:`~corelay.processor.base.Processor` that is to be used in the test.
        data (numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]): The data that is to be clustered.
    """

    processor = processor_type(n_clusters=5)
    computed_clustering = processor(data)

    # Checks that the number of unique cluster labels is equal to the number of clusters
    assert len(numpy.unique(computed_clustering)) == 5

    # Since the blobs are of equal size the cluster sizes should also be approximately the same size
    assert (numpy.unique(computed_clustering, return_counts=True)[1] / 1000).std() < 0.05


@pytest.mark.parametrize('processor_type', EXTRA_PROCESSORS + [clustering.DBSCAN])
def test_embedding_on_distances(
    processor_type: type[clustering.Clustering],
    distances: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]
) -> None:
    """Tests the clustering processors by checking that the 5 found clusters that were determined from pre-computed distances are approximately of
    similar size.

    Args:
        processor_type (type[clustering.Clustering]): The clustering :py:class:`~corelay.processor.base.Processor` that is to be used in the test.
        distances (numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]): The pre-computed distances of the data points that are used to cluster the
            data.
    """

    params: dict[str, typing.Any] = {'eps': 0.9} if 'eps' in processor_type.collect(Param) else {}
    processor = processor_type(metric='precomputed', **params)
    computed_clustering = processor(distances)

    assert len(numpy.unique(computed_clustering)) == 5
    assert (numpy.unique(computed_clustering, return_counts=True)[1] / 1000).std() < 0.13


def test_embedding_on_distances_using_agglomerative_clustering(distances: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> None:
    """Tests that the 5 clusters found using the :py:class:`~corelay.clustering.AgglomerativeClustering` class that were determined from pre-computed
    distances are approximately of similar size.

    Args:
        distances (numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]): The pre-computed distances of the data points that are used to cluster the
            data.
    """

    computed_clustering = clustering.AgglomerativeClustering(n_clusters=5, metric='precomputed', linkage='average')(distances)
    assert (numpy.unique(computed_clustering, return_counts=True)[1] / 1000).std() < 0.05


def test_dendrogram_creation_with_file_path(tiny_data: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> None:
    """Tests the creation of a dendrogram for the specified data, where the dendrogram image file is specified as a path :py:class:`str`.

    Args:
        tiny_data (numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]): The data to create the dendrogram for.
    """

    output_path = '/tmp/dendrogram.png'
    dendrogram_processor = clustering.Dendrogram(output_file=output_path)
    output_data = dendrogram_processor(tiny_data)

    numpy.testing.assert_equal(tiny_data, output_data)
    assert os.path.exists(output_path)

    os.remove(output_path)


def test_dendrogram_creation_with_file_object(tiny_data: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> None:
    """Tests the creation of a dendrogram for the specified data, where the dendrogram image file is specified as a file object.

    Args:
        tiny_data (numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]): The data to create the dendrogram for.
    """

    output_path = '/tmp/dendrogram.png'
    with open(output_path, 'wb') as image_file:
        dendrogram_processor = clustering.Dendrogram(output_file=image_file)
        output_data = dendrogram_processor(tiny_data)

        numpy.testing.assert_equal(tiny_data, output_data)
        assert os.path.exists(output_path)

    os.remove(output_path)
