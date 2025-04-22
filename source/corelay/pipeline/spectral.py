"""A module that contains SprAy-specific spectral clustering pipelines."""

from corelay.pipeline.base import Pipeline, Task
from corelay.processor.affinity import SparseKNN
from corelay.processor.clustering import KMeans
from corelay.processor.distance import SciPyPDist
from corelay.processor.embedding import EigenDecomposition
from corelay.processor.laplacian import SymmetricNormalLaplacian


class SpectralEmbedding(Pipeline):
    """A pipeline for spectral embeddings, which is customizable with different pre-processing, pairwise distance, affinity, laplacian, and embedding
    functions. When an instance of the pipeline is called, it will return eigenvalues and eigenvectors of the spectral embedding, as NumPy arrays.


    Args:
        preprocessing (Callable[[NDArray[Any]], Any], optional): A custom pre-processing function to be applied to the data before computing the
            pairwise distance. Defaults to the identity function.
        pairwise_distance (Callable[[NDArray[Any]], Any], optional): A custom pairwise distance function to be applied to the data. Defaults to the
            euclidean distance.
        affinity (Callable[[NDArray[Any]], Any], optional): A custom affinity function to be applied to the pairwise distance matrix. Defaults to a
            sparse k-nearest neighbors graph with 10 neighbors.
        laplacian (Callable[[NDArray[Any]], Any], optional): A custom graph laplacian function to be applied to the pairwise distance matrix. Defaults
            to a symmetric normal laplacian.
        embedding (Callable[[NDArray[Any]], Any], optional): A custom embedding function to be applied to the graph laplacian. Defaults to an
            eigen decomposition with 32 eigenvalues.

    Notes:
        Pre-computed distance matrices can be supplied by passing `pairwise_distance=lambda x: x`.
        Pre-computed affinity matrices can be supplied by additionally passing `affinity=lambda x: x`.
        Pre-computed graph laplacian matrices can be supplied by further passing `laplacian=lambda x: x`.
    """

    preprocessing = Task(default=lambda x: x)
    """A pre-processing task to be applied to the data before computing the pairwise distance task. Defaults to the identity function."""

    pairwise_distance = Task(default=SciPyPDist(metric='euclidean'))
    """A pairwise distance task to be applied to the data. Defaults to the euclidean distance."""

    affinity = Task(default=SparseKNN(n_neighbors=10, symmetric=True))
    """An affinity task to be applied to the pairwise distance matrix.  Defaults to a sparse k-nearest neighbors graph with 10 neighbors."""

    laplacian = Task(default=SymmetricNormalLaplacian())
    """A graph laplacian task to be applied to the pairwise distance matrix. Defaults to a symmetric normal laplacian."""

    embedding = Task(default=EigenDecomposition(n_eigval=32), is_output=True)
    """An embedding task to be applied to the graph laplacian matrix. Defaults to an eigen decomposition with 32 eigenvalues."""


class SpectralClustering(SpectralEmbedding):
    """A pipeline for spectral clustering a spectral embedding, which is customizable with a custom clustering function. When an instance of the
    pipeline is called, it will return eigenvalues and eigenvectors of the spectral embedding, and the labels of the spectral clustering, as NumPy
    arrays.

    Args:
        select_eigenvector (Callable[[NDArray[Any]], Any], optional): A custom task to select the eigenvector from the spectral embedding. Defaults to
            the second output of the spectral embedding.
        clustering (Callable[[NDArray[Any]], Any], optional): A custom clustering function to be applied to the spectral embedding. Defaults to a
            k-Means clustering with 2 clusters.
        preprocessing (Callable[[NDArray[Any]], Any], optional): A custom pre-processing function to be applied to the data before computing the
            pairwise distance. Defaults to the identity function.
        pairwise_distance (Callable[[NDArray[Any]], Any], optional): A custom pairwise distance function to be applied to the data. Defaults to the
            euclidean distance.
        affinity (Callable[[NDArray[Any]], Any], optional): A custom affinity function to be applied to the pairwise distance matrix. Defaults to a
            sparse k-nearest neighbors graph with 10 neighbors.
        laplacian (Callable[[NDArray[Any]], Any], optional): A custom graph laplacian function to be applied to the pairwise distance matrix. Defaults
            to a symmetric normal laplacian.
        embedding (Callable[[NDArray[Any]], Any], optional): A custom embedding function to be applied to the graph laplacian. Defaults to an
            eigen decomposition with 32 eigenvalues.
    """

    select_eigenvector = Task(default=lambda x: x[1])
    """A task to select the eigenvector from the spectral embedding. Defaults to the second output of the spectral embedding."""

    clustering = Task(default=KMeans(n_clusters=2), is_output=True)
    """A clustering task to be applied to the spectral embedding. Defaults to a k-Means clustering with 2 clusters."""
