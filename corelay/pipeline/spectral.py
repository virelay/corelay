"""SprAy-specific spectral clustering pipelines.

"""
import logging

from ..processor.distance import SciPyPDist
from ..processor.affinity import SparseKNN
from ..processor.laplacian import SymmetricNormalLaplacian
from ..processor.embedding import EigenDecomposition
from ..processor.clustering import KMeans
from .base import Pipeline, Task

LOGGER = logging.getLogger(__name__)


class SpectralEmbedding(Pipeline):
    """Spectral Embedding with custom pipeline

    Parameters
    ----------
    preprocessing : callable, optional
        data pre-processing function of signature: (data : :obj:`numpy.ndarray`,) -> :obj:`numpy.ndarray`
    pairwise_distance : callable, optional
        pairwise distance function of signature: (data : :obj:`numpy.ndarray`,) -> :obj:`numpy.ndarray`
    affinity : callable, optional
        affinity function of signature: (distance : :obj:`numpy.ndarray`,) -> :obj:`numpy.ndarray`
    laplacian : callable, optional
        laplacian function of signature: (distance : :obj:`numpy.ndarray`,) -> :obj:`numpy.ndarray`

    Notes
    -----
    Pre-computed distance matrices can be supplied by passing ``pairwise_distance=(lambda x: x)``.
    Pre-computed affinity matrices can be supplied by additionally passing ``affinity=(lambda x: x)``.
    Pre-computed graph laplacian matrices can be supplied by further passing ``laplacian=(lambda x: x)``.

    """
    preprocessing = Task(default=(lambda x: x))
    pairwise_distance = Task(default=SciPyPDist(metric='euclidean'))
    affinity = Task(default=SparseKNN(n_neighbors=10, symmetric=True))
    laplacian = Task(default=SymmetricNormalLaplacian())
    embedding = Task(default=EigenDecomposition(n_eigval=32), is_output=True)


class SpectralClustering(SpectralEmbedding):
    """Clustering on a spectral embedding

    Parameters
    ----------
    clustering_fn : callable, optional
        label-returning clustering function of signature:
        (embedding : :obj:`numpy.ndarray`,) -> :obj:`numpy.ndarray`
    **kwargs
        Keyword arguments for `SpectralEmbedding`

    Returns
    -------
    :obj:`numpy.ndarray`
        Eigenvalues for spectral embedding
    :obj:`numpy.ndarray`
        Spectral embedding (eigenvectors)
    :obj:`numpy.ndarray`
        Labels of clustering on spectral embedding

    """
    select_eigenvector = Task(default=lambda x: x[1])
    clustering = Task(default=KMeans(n_clusters=2), is_output=True)
