"""A module that contains unit tests for the :py:mod:`corelay.processor.embedding` module."""

import typing
from importlib import import_module

import numpy
import pytest
from sklearn.datasets import load_digits
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import Bunch

from corelay.processor import embedding
from corelay.processor.base import Processor

EMBEDDING_PROCESSORS: list[type[embedding.Embedding]] = [embedding.TSNEEmbedding, embedding.LLEEmbedding, embedding.PCAEmbedding]
"""Contains a list of embedding processors that are to be tested."""


EXTRA_PROCESSORS: list[type[embedding.Embedding]] = []
"""Contains a list of extra embedding processors that are to be tested. The list of extra embedding processors depends on the availability of the
respective libraries, which can optionally be installed by the user. If the library is not installed, the processor will not be available and can
therefore not be tested.
"""


try:
    import_module('umap')
    EXTRA_PROCESSORS = [embedding.UMAPEmbedding]
except ImportError:
    pass


@pytest.fixture(name='data', scope='module')
def get_data_fixture() -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
    """A test fixture, which loads a digit dataset that comprises images of two kinds of digits with a shape of (360, 64).

    Returns:
        numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: Returns the data that was loaded from the dataset.
    """

    digits: Bunch = load_digits(n_class=2)
    digits_data: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] = digits['data']
    return digits_data


@pytest.fixture(name='distances', scope='module')
def get_distances_fixture(data: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
    """A test fixture, which takes the digit dataset loaded in the ``data`` fixture and computes the euclidean distances on it.

    Args:
        data (numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]): The input data that is to be used for the distance computation. This data comes
            from the ``data`` fixture.

    Returns:
        numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: Returns the euclidean distances that were computed on the input data.
    """

    distance_matrix: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] = euclidean_distances(data)
    return distance_matrix


@pytest.mark.parametrize('processor_type', EMBEDDING_PROCESSORS + EXTRA_PROCESSORS)
def test_embedding(processor_type: type[Processor], data: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> None:
    """Tests the embedding processors on the specified data and checks the dimensions of the result.

    Args:
        processor_type (type[Processor]): The type of embedding processor that is to be tested.
        data (numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]): The input data that is to be used for the embedding. This data comes from the
            ``data`` fixture.
    """

    processor = processor_type()
    output_embedding: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] = processor(data)

    assert output_embedding.shape == (360, 2)


def test_eigen_decomposition(distances: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> None:
    """Tests the eigen-decomposition of the distances and check the dimensions.

    Args:
        distances (numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]): The input data that is to be used for the eigen-decomposition. This data
            comes from the ``distances`` fixture.
    """

    eigen_decomposition = embedding.EigenDecomposition(n_eigval=32)
    eigenvalues, eigenvectors = eigen_decomposition(distances)

    assert eigenvalues.shape == (32, )
    assert eigenvectors.shape == (360, 32)
    numpy.testing.assert_array_almost_equal(numpy.linalg.norm(eigenvectors, axis=1, keepdims=True), numpy.ones((360, 1)))

    eigen_decomposition = embedding.EigenDecomposition(n_eigval=32, normalize=False)
    eigenvalues, eigenvectors = eigen_decomposition(distances)

    assert eigenvalues.shape == (32, )
    assert eigenvectors.shape == (360, 32)
    assert numpy.sum(numpy.abs(numpy.linalg.norm(eigenvectors, axis=1, keepdims=True) - numpy.ones((360, 1)))).item() != 0.0


def test_eigen_decomposition_invalid_metric() -> None:
    """Tests the eigen-decomposition processor with an invalid metric."""

    eigen_decomposition = embedding.EigenDecomposition(which='invalid_metric')

    with pytest.raises(ValueError):
        eigen_decomposition(numpy.random.rand(360, 360))


@pytest.mark.parametrize('processor_type', EXTRA_PROCESSORS + [embedding.TSNEEmbedding])
@pytest.mark.filterwarnings('ignore:using precomputed metric; inverse_transform will be unavailable')
def test_embedding_on_distances(processor_type: type[Processor], distances: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> None:
    """Tests the embedding processors on pre-computed distances and checks the dimensions.

    Args:
        processor_type (type[Processor]): The type of embedding processor that is to be tested.
        distances (numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]): The input data that is to be used for the embedding. This data comes from
            the ``distances`` fixture.
    """

    processor = processor_type(metric='precomputed')
    output_embedding = processor(distances)

    assert output_embedding.shape == (360, 2)

def test_eigen_decomposition_output_representation() -> None:
    """Tests the output representation of the eigen-decomposition processor."""

    eigen_decomposition = embedding.EigenDecomposition()
    assert repr(eigen_decomposition).endswith('(eigval: numpy.ndarray, eigvec: numpy.ndarray)')
