"""Test spectral pipeline and processors.

"""

import os

import pytest
import numpy as np
from numpy import pi


import matplotlib.pyplot as plt

from corelay.pipeline.spectral import SpectralEmbedding, SpectralClustering
from corelay.processor.affinity import SparseKNN
from corelay.processor.laplacian import SymmetricNormalLaplacian
from corelay.processor.embedding import EigenDecomposition
from corelay.processor.clustering import KMeans


@pytest.fixture(scope='module')
def spiral_data(n=150):
    """Sample some double spiral data points

    Parameters
    ----------
    n : int
        samples per class of the two classes

    """
    np.random.seed(1345123)  # fix seed for data
    _ = np.random.uniform(size=(2, 100)).T  # "part of the seed"

    # generates double-spiral data in 2-d with N data points
    theta = np.sqrt(np.random.rand(n)) * 2 * pi

    r_a = 2 * theta + pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + np.random.randn(n, 2) * .5

    r_b = -2 * theta - pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    x_b = data_b + np.random.randn(n, 2) * .5

    return np.append(x_a, x_b, axis=0)


@pytest.fixture(scope='module')
def k_knn(spiral_data):
    """Choose k for KNN"""
    return int(np.log(spiral_data.shape[0]))


@pytest.fixture(scope='module')
def k_eig():
    """Number for eigen values"""
    return 8


@pytest.fixture(scope='module')
def k_clusters():
    """Number of clusters"""
    return 4


class TestSpectral:
    """Test class for SpectralClustering"""
    @staticmethod
    def test_spectral_embedding_instatiation():
        """test whether we can instantiate a spectral embedding instance successfully"""
        SpectralEmbedding()

    @staticmethod
    def test_spectral_clustering_instatiation():
        """test whether we can instantiate a spectral clustering instance successfully"""
        SpectralClustering()

    @staticmethod
    def test_data_generation(spiral_data):
        """sanity check. make sure the data looks as expected (from the outside)"""
        assert isinstance(spiral_data, np.ndarray), f'Expected numpy.ndarray type, got {type(spiral_data)}'
        assert spiral_data.shape == (300, 2), f'Expected spiral_data shape (300, 2), got {spiral_data.shape}'

    @staticmethod
    def test_spectral_embedding_default_params(spiral_data):
        """test wheter the SE operates on data all the way through, using its default parameters."""
        pipeline = SpectralEmbedding()
        output = pipeline(spiral_data)
        assert isinstance(output, tuple), f'Expected tuple type output, got {type(output)}'
        assert len(output) == 2, f'Expected output length of 2, got {len(output)}'

        eigval, eigvec = output
        assert isinstance(eigval, np.ndarray), f'Expected eigval to be numpy.ndarray, but got {type(eigval)}'
        assert isinstance(eigvec, np.ndarray), f'Expected eigvec to be numpy.ndarray, but got {type(eigvec)}'
        assert eigvec.shape[0] == spiral_data.shape[0], (
            'Expected number of eigenvectors to be identical to number of samples '
            f'({spiral_data.shape[0]}), but got {eigvec.shape[0]}'
        )
        assert eigvec.shape[1] == eigval.size, (
            f'Expected dim of eigenvectors {eigvec.shape[1]} be be identical '
            f'to the number of reported eigenvalues {eigval.size}'
        )

    @staticmethod
    def test_spectral_clustering_default_params(spiral_data):
        """test wheter the SC operates on data all the way through, using its default parameters."""
        pipeline = SpectralClustering()
        output = pipeline(spiral_data)
        assert isinstance(output, tuple), f'Expected tuple type output, got {type(output)}'
        assert len(output) == 2, f'Expected output lenght of 2, got {len(output)}'

        eigenstuff, labels = output
        assert isinstance(eigenstuff, tuple), (
            f'Expected tuple type output for eigenstuff, got {type(eigenstuff)}'
        )
        assert len(eigenstuff) == 2, (
            f'Expected eigenstuff length of 2, got {len(eigenstuff)}'
        )

        eigval, eigvec = eigenstuff
        assert isinstance(eigval, np.ndarray), f'Expected eigval to be numpy.ndarray, but got {type(eigval)}'
        assert isinstance(eigvec, np.ndarray), f'Expected eigvec to be numpy.ndarray, but got {type(eigvec)}'
        assert eigvec.shape[0] == spiral_data.shape[0], (
            'Expected number of eigenvectors to be identical to number of samples '
            f'({spiral_data.shape[0]}), but got {eigvec.shape[0]}'
        )
        assert eigvec.shape[1] == eigval.size, (
            f'Expected dim of eigenvectors {eigvec.shape[1]} be be identical to the number of '
            f'reported eigenvalues {eigval.size}'
        )

        assert isinstance(labels, np.ndarray), f'Expected labels to be numpy.ndarray, but got {type(labels)}'
        assert labels.size == spiral_data.shape[0], (
            'Expected number of labels to be identical to number of samples in spiral_data '
            f'({spiral_data.shape[0]}), but got {labels.size}'
        )

        assert labels.ndim == 1, f'Expected labels to be flat array, but was shaped {labels.shape}'

    @staticmethod
    def test_spectral_clustering_step_by_step_custom_params(spiral_data, k_knn, k_eig, k_clusters):
        """this test manually compares the pipelines working order against manually
        executed steps with custom parameters attuned to the spiral data

        first, separately prepare processors for customized pipeline
        customizations:
          1) parameters, as passed to the test function
          2) all processors (affected by changed parameters) produce outputs this time for later comparison

        """

        knn = SparseKNN(n_neighbors=k_knn, symmetric=True, is_output=True)
        lap = SymmetricNormalLaplacian(is_output=True)

        v0 = np.random.rand(spiral_data.shape[0])  # some fixed init vector
        v0 /= np.linalg.norm(v0, 1)
        eig = EigenDecomposition(n_eigval=k_eig, is_output=True,
                                 kwargs={'v0': v0})

        random_state = 0
        kmn = KMeans(n_clusters=k_clusters, is_output=True,
                     kwargs={'random_state': random_state})

        pipeline = SpectralClustering(
            affinity=knn,
            laplacian=lap,
            embedding=eig,
            clustering=kmn
        )

        output_pipeline = pipeline(spiral_data)
        assert len(output_pipeline) == 4, (
            f'length of output expected to be 4 (affinity, laplacian, embedding, labels), but is {len(output_pipeline)}'
        )

        # unpack pipeline results
        aff_pipe, lap_pipe, eig_pipe, label_pipe = output_pipeline

        # produce results manually and compare to pipeline output
        dist_man = pipeline.pairwise_distance(spiral_data)
        aff_man = knn(dist_man)
        np.testing.assert_array_equal(np.array(aff_man.todense()),
                                      np.array(aff_pipe.todense()),
                                      'Affinity matrices are not equal!')

        lap_man = lap(aff_man)
        np.testing.assert_array_equal(np.array(lap_man.todense()),
                                      np.array(lap_pipe.todense()),
                                      'Laplacians are not equal!')

        eig_man = eig(lap_man)
        np.testing.assert_array_equal(eig_man[0],
                                      eig_pipe[0],
                                      'Eigenvalues not equal!')
        np.testing.assert_array_equal(eig_man[1],
                                      eig_pipe[1],
                                      'Eigenvectors not equal enough!')

        label_man = kmn(eig_man[1])
        np.testing.assert_array_equal(label_man,
                                      label_pipe,
                                      'Label vectors not equal!')

        path = '/tmp/spectral_eigenvalues.pdf'
        plt.figure()
        plt.plot(eig_pipe[0][::-1], '.')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(path)
        os.remove(path)

        path = '/tmp/spectral_labelling.pdf'
        plt.figure()
        plt.scatter(x=spiral_data[:, 0], y=spiral_data[:, 1], c=label_pipe)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(path)
        os.remove(path)
