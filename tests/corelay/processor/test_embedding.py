"""Test embedding processors.

"""

import pytest
from sklearn.datasets import load_digits
from sklearn.metrics.pairwise import euclidean_distances

from corelay.processor import embedding

EMBEDDING_PROCESSORS = [embedding.TSNEEmbedding, embedding.LLEEmbedding, embedding.PCAEmbedding]

try:
    import umap  # pylint: disable=unused-import; # noqa: F401
    EXTRA_PROCESSORS = [embedding.UMAPEmbedding]
except ImportError:
    EXTRA_PROCESSORS = []


@pytest.fixture
def data():
    """Return data with images of 2 kind of digits with shape (360, 64).

    """
    digits = load_digits(n_class=2)  # shape 360 x 64
    # pylint: disable=no-member
    return digits.data


@pytest.fixture
def distances(data):
    """Return euclidean distances on data.

    """
    return euclidean_distances(data)


@pytest.mark.parametrize('processor', EMBEDDING_PROCESSORS + EXTRA_PROCESSORS)
def test_embedding(processor, data):
    """Test embedding processors on data and check the dimensions.

    """
    emb = processor()(data)
    assert emb.shape == (360, 2)


def test_eigen_decomposition(distances):
    """Test eigen decomposition and check the dimensions.

    """
    eigval, eigvec = embedding.EigenDecomposition(n_eigval=32)(distances)
    assert eigval.shape == (32, )
    assert eigvec.shape == (360, 32)


@pytest.mark.parametrize('processor', [embedding.TSNEEmbedding] + EXTRA_PROCESSORS)
def test_embedding_on_distances(processor, distances):
    """Test embedding processors on precomputed distances and check the dimensions.

    """
    emb = processor(metric='precomputed')(distances)
    assert emb.shape == (360, 2)
