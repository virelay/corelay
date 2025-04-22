"""A module that contains processors for clustering algorithms."""

import io
import os
from collections.abc import Callable
from typing import Annotated, Any, Literal, SupportsIndex

import scipy.cluster.hierarchy as shc
import sklearn.cluster
from numpy.typing import NDArray
from matplotlib import pyplot as plt
from sklearn.base import ClusterMixin

from corelay.base import Param
from corelay.processor.base import Processor
from corelay.utils import import_or_stub


hdbscan: Callable[..., ClusterMixin] = import_or_stub('hdbscan', 'HDBSCAN')
"""Performs the HDBSCAN clustering algorithm on a vector or distance matrix.

Returns:
    ClusterMixin: Returns an HDBSCAN cluster estimator, which can be used to fit the data.

Note:
    Since the HDBSCAN library is an optional dependency of CoRelAy, it is imported using the ``import_or_stub`` function, which tries to import the
    module/type/function specified. If the import fails, it returns a stub instead, which will raise an exception when used. The exception message
    will tell users how to install the missing dependencies for the functionality to work.
"""


class Clustering(Processor):
    """The abstract base class for ``Processor`` that performs clustering.

    Args:
        is_output (bool, optional): A value indicating whether this ``Clustering`` processor is the output of a ``Pipeline``. Defaults to `False`.
        is_checkpoint (bool | None, optional): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to `False`.
        io (Storable | None, optional): An IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this
            run or in subsequent runs of the ``Pipeline``. Defaults to an instance of ``NoStorage``.
        kwargs (dict[str, Any], optional): Additional keyword arguments for the clustering algorithm. Defaults to an empty dictionary.
    """

    kwargs: Annotated[dict[str, Any], Param(dict, default={})]
    """A dictionary of additional keyword arguments for the clustering algorithm. Defaults to an empty dictionary."""


class KMeans(Clustering):
    """A clustering ``Processor`` that performs the k-Means clustering algorithm.

    Args:
        is_output (bool, optional): A value indicating whether this ``KMeans`` clustering processor is the output of a ``Pipeline``. Defaults to
            `False`.
        is_checkpoint (bool | None, optional): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to `False`.
        io (Storable | None, optional): An IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this
            run or in subsequent runs of the ``Pipeline``. Defaults to an instance of ``NoStorage``.
        kwargs (dict[str, Any], optional): Additional keyword arguments for the k-Means clustering algorithm. Defaults to an empty dictionary.
        n_clusters (int, optional): The number of clusters to form. Defaults to 2.
        index (tuple[int | slice], optional): The indices of the data to be clustered. Defaults to an empty slice.

    See Also:
        :obj:`sklearn.cluster.KMeans`
    """

    n_clusters: Annotated[int, Param(int, 2, identifier=True)]
    """The number of clusters to form. Defaults to 2."""

    index: Annotated[SupportsIndex | tuple[SupportsIndex, ...], Param((SupportsIndex, tuple), (slice(None),))]
    """The indices of the data to be clustered. Defaults to an empty slice."""

    def function(self, data: Any) -> Any:
        """Performs k-Means clustering on the data.

        Args:
            data (Any): The data to be clustered. The data should be a NumPy array or a sparse matrix.

        Returns:
            Any: Returns the a NumPy array of shape `(number_of_samples,)`, that contains the cluster labels, where each label corresponds to a
                cluster assigned to the data point.
        """

        input_data: NDArray[Any] = data
        return sklearn.cluster.KMeans(n_clusters=self.n_clusters, **self.kwargs).fit_predict(input_data[self.index])


HDBSCANDistanceMeasure = Literal[
    'arccos',
    'braycurtis',
    'canberra',
    'chebyshev',
    'cityblock',
    'cosine',
    'dice',
    'euclidean',
    'hamming',
    'haversine',
    'infinity',
    'jaccard',
    'kulsinski',
    'l1',
    'l2',
    'mahalanobis',
    'manhattan',
    'matching',
    'minkowski',
    'p',
    'rogerstanimoto',
    'russellrao',
    'seuclidean',
    'sokalmichener',
    'sokalsneath',
    'wminkowski'
]
"""An enumeration of the distance measures supported by HDBSCAN."""


class HDBSCAN(Clustering):
    """A clustering ``Processor`` that performs the HDBSCAN clustering algorithm.

    Args:
        is_output (bool, optional): A value indicating whether this ``HDBSCAN`` clustering processor is the output of a ``Pipeline``. Defaults to
            `False`.
        is_checkpoint (bool | None, optional): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to `False`.
        io (Storable | None, optional): An IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this
            run or in subsequent runs of the ``Pipeline``. Defaults to an instance of ``NoStorage``.
        kwargs (dict[str, Any], optional): Additional keyword arguments for the HDBSCAN clustering algorithm. Defaults to an empty dictionary.
        n_clusters (int, optional): The number of clusters to form. Defaults to 2.
        metric (HDBSCANDistanceMeasure, optional): The distance metric to use. This can be one of "euclidean", "l2" (equivalent to "euclidean"),
            "minkowski", "p" (equivalent to "minkowski"), "manhattan", "cityblock" (equivalent to "manhattan"), "l1" (equivalent to "manhattan"),
            "chebyshev", "infinity" (equivalent to "chebyshev"), "seuclidean", "mahalanobis", "wminkowski", "hamming", "canberra", "braycurtis",
            "matching", "jaccard", "dice", "kulsinski", "rogerstanimoto", "russellrao", "sokalmichener", "sokalsneath", "haversine", "cosine" (since
            the cosine distance is not a true distance measure, it is not supported; using "cosine" will use the "arccos" distance instead), and
            "arccos". Defaults to "euclidean".

    See Also:
        :obj:`hdbscan.HDBSCAN`

    Notes:
        GitHub repository including documentation for HDBSCAN: https://github.com/scikit-learn-contrib/hdbscan.
    """

    n_clusters: Annotated[int, Param(int, 5, identifier=True)]
    """The number of clusters to form. Defaults to 5."""

    metric: Annotated[HDBSCANDistanceMeasure, Param(str, 'euclidean', identifier=True)]
    """The distance metric to use. Defaults to "euclidean"."""

    def function(self, data: Any) -> Any:
        """Performs the HDBSCAN clustering algorithm on the data.

        Args:
            data (Any): The data to be clustered. The data should be a NumPy array of shape `(number_of_samples, number_of_features)` or a sparse
                matrix.

        Returns:
            Any: Returns a NumPy array of shape `(number_of_samples,)`, that contains the cluster labels, where each label corresponds to a cluster
                assigned to the data point.
        """

        clustering = hdbscan(min_cluster_size=self.n_clusters, metric=self.metric, **self.kwargs)
        return clustering.fit_predict(data)


DBSCANDistanceMeasure = Literal[
    'cityblock',
    'cosine',
    'euclidean',
    'haversine',
    'l1',
    'l2',
    'manhattan',
    'nan_euclidean'
    'precomputed',
]
"""An enumeration of the distance measures supported by DBSCAN."""


class DBSCAN(Clustering):
    """A clustering ``Processor`` that performs the DBSCAN clustering algorithm.

    Args:
        is_output (bool, optional): A value indicating whether this ``DBSCAN`` clustering processor is the output of a ``Pipeline``. Defaults to
            `False`.
        is_checkpoint (bool | None, optional): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to `False`.
        io (Storable | None, optional): An IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this
            run or in subsequent runs of the ``Pipeline``. Defaults to an instance of ``NoStorage``.
        kwargs (dict[str, Any], optional): Additional keyword arguments for the DBSCAN clustering algorithm. Defaults to an empty dictionary.
        metric (str, optional): The distance metric to use. Defaults to "euclidean".
        eps (float, optional): The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a
            maximum bound on the distances of points within a cluster. This is the most important DBSCAN parameter to choose appropriately for your
            data set and distance function. Defaults to 0.5.
        min_samples (int, optional): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This
            includes the point itself. If ``min_samples`` is set to a higher value, DBSCAN will find denser clusters, whereas if it is set to a lower
            value, the found clusters will be more sparse. Defaults to 5.

    See Also:
        :obj:`sklearn.cluster.DBSCAN`
    """

    metric: Annotated[DBSCANDistanceMeasure, Param(str, 'euclidean', identifier=True)]
    """The distance metric to use. Defaults to "euclidean"."""

    eps: Annotated[float, Param(float, 0.5, identifier=True)]
    """The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the
    distances of points within a cluster. This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
    Defaults to 0.5.
    """

    min_samples: Annotated[int, Param(int, 5, identifier=True)]
    """The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself. If
    ``min_samples`` is set to a higher value, DBSCAN will find denser clusters, whereas if it is set to a lower value, the found clusters will be more
    sparse. Defaults to 5.
    """

    def function(self, data: Any) -> Any:
        """Performs the DBSCAN clustering algorithm on the data.

        Args:
            data (Any): The data to be clustered. The data should be a NumPy array of shape `(number_of_samples, number_of_features)` or a sparse
                matrix.

        Returns:
            Any: Returns a NumPy array of shape `(number_of_samples,)`, that contains the cluster labels, where each label corresponds to a cluster
                assigned to the data point. Noisy points will be labeled as -1.
        """

        clustering = sklearn.cluster.DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            **self.kwargs
        )
        return clustering.fit_predict(data)


AgglomerativeClusteringDistanceMeasure = Literal[
    'euclidean',
    'l1',
    'l2',
    'manhattan',
    'cosine',
    'precomputed'
]
"""An enumeration of the distance measures supported by the Agglomerative Clustering algorithm."""


AgglomerativeClusteringLinkageMethod = Literal['ward', 'complete', 'average', 'single']
"""An enumeration of the linkage method supported by the Agglomerative Clustering algorithm. The linkage method is used to determine the distance
between two clusters when performing hierarchical clustering. The Agglomerative Clustering algorithm will merge the pairs of clusters that minimize
this method. The following linkage methods are supported:

- "ward" minimizes the variance of the clusters being merged.
- "average" uses the average of the distances of each observation of the two clusters.
- "complete" uses the maximum distances between all observations of the two clusters.
- "single" uses the minimum of the distances between all observations of the two clusters.
"""


class AgglomerativeClustering(Clustering):
    """A clustering ``Processor`` that performs the Agglomerative Clustering algorithm.

    Args:
        is_output (bool, optional): A value indicating whether this ``AgglomerativeClustering`` clustering processor is the output of a ``Pipeline``.
            Defaults to `False`.
        is_checkpoint (bool | None, optional): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to `False`.
        io (Storable | None, optional): An IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this
            run or in subsequent runs of the ``Pipeline``. Defaults to an instance of ``NoStorage``.
        kwargs (dict[str, Any], optional): Additional keyword arguments for the agglomerative clustering algorithm. Defaults to an empty dictionary.
        n_clusters (int, optional): The number of clusters to form. Defaults to 5.
        metric (AgglomerativeClusteringDistanceMeasure, optional): The distance metric to use. Defaults to "euclidean".
        linkage (AgglomerativeClusteringLinkageMethod, optional): The linkage method to use. This determines which distance to use between two newly
            formed clusters. The algorithm will merge the pairs of clusters that minimize this method. Defaults to "ward".

    See Also:
        :obj:`sklearn.cluster.AgglomerativeClustering`
    """

    n_clusters: Annotated[int, Param(int, 5, identifier=True)]
    """The number of clusters to form. Defaults to 5."""

    metric: Annotated[AgglomerativeClusteringDistanceMeasure, Param(str, 'euclidean', identifier=True)]
    """The distance metric to use. Defaults to "euclidean"."""

    linkage: Annotated[AgglomerativeClusteringLinkageMethod, Param(str, 'ward', identifier=True)]
    """The linkage method to use. This determines which distance to use between two newly formed clusters. The algorithm will merge the pairs of
    clusters that minimize this method. Defaults to "ward".
    """

    def function(self, data: Any) -> Any:
        """Performs the Agglomerative Clustering algorithm on the data.

        Args:
            data (Any): The data to be clustered. The data should be a NumPy array of shape `(number_of_samples, number_of_features)` or
                `(number_of_samples, number_of_samples)`, or a sparse matrix.

        Returns:
            Any: _description_
        """

        clustering = sklearn.cluster.AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric=self.metric,
            linkage=self.linkage,
            **self.kwargs
        )
        return clustering.fit_predict(data)


DendrogramDistanceMeasure = Literal[
    'braycurtis',
    'canberra',
    'chebyshev',
    'cityblock',
    'correlation',
    'cosine',
    'dice',
    'euclidean',
    'hamming',
    'jaccard',
    'jensenshannon',
    'kulczynski1',
    'mahalanobis',
    'minkowski',
    'rogerstanimoto',
    'russellrao',
    'seuclidean',
    'sokalmichener',
    'sokalsneath',
    'sqeuclidean',
    'yule'
]
"""An enumeration of the distance measures supported by the dendrogram clustering ``Processor``."""


DendrogramLinkageMethod = Literal['ward', 'average', 'complete', 'single', 'centroid', 'median', 'weighted']
"""An enumeration of the linkage method supported by the hierarchical clustering algorithm used by the Dendrogram ``Processor``. The linkage method is
used to determine the distance between two newly formed clusters when performing hierarchical clustering. The hierarchical clustering algorithm used
by the Dendrogram ``Processor`` will merge the pairs of clusters that minimize this method. The following linkage methods are supported:

- "ward" minimizes the variance of the clusters being merged.
- "average" uses the average of the distances of each observation of the two clusters.
- "complete" uses the maximum distances between all observations of the two clusters.
- "single" uses the minimum of the distances between all observations of the two clusters.
- "centroid" the centroid of the new cluster that would be formed by merging the two clusters.
- "median" uses the median of the centroids of the two clusters.
- "weighted" assigns the weighted distance between the two original clusters and a third remaining cluster to the new cluster.
"""


class Dendrogram(Clustering):
    """A clustering ``Processor`` that generates plots the hierarchical clustering as a dendrogram.

    Args:
        is_output (bool, optional): A value indicating whether this ``Dendrogram`` clustering processor is the output of a ``Pipeline``. Defaults to
            `False`.
        is_checkpoint (bool | None, optional): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to `False`.
        io (Storable | None, optional): An IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this
            run or in subsequent runs of the ``Pipeline``. Defaults to an instance of ``NoStorage``.
        kwargs (dict[str, Any], optional): Additional keyword arguments for the hierarchical clustering algorithm. Defaults to an empty dictionary.
        output_file (str | bytes | os.PathLike[Any] | io.IOBase): The path to a file or a file descriptor to save the dendrogram plot to.
        metric (DendrogramDistanceMeasure, optional): The distance metric to use for the clustering. Defaults to "euclidean".
        linkage (DendrogramLinkageMethod, optional): The linkage criterion to use. This determines which distance to use between sets of observation.
            Defaults to "ward".
    """

    output_file: Annotated[str | bytes | os.PathLike[Any] | io.IOBase, Param((str, bytes, os.PathLike, io.IOBase), mandatory=True)]
    """The path to a file or a file descriptor to save the dendrogram plot to."""

    metric: Annotated[DendrogramDistanceMeasure, Param(str, 'euclidean')]
    """The distance metric to use for the clustering. This can be one of "braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine",
    "dice", "euclidean", "hamming", "jaccard", "jensenshannon", "kulczynski1", "mahalanobis", "minkowski", "rogerstanimoto", "russellrao",
    "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", or "yule". Defaults to "euclidean".
    """

    linkage: Annotated[DendrogramLinkageMethod, Param(str, 'ward')]
    """The linkage criterion to use. This determines which distance to use between sets of observation. Defaults to "ward"."""

    def function(self, data: Any) -> Any:
        """Performs the hierarchical clustering algorithm on the data and generates a dendrogram plot.

        Args:
            data (Any): The data to be clustered. The data should be a NumPy array that contains a condensed distance matrix. A condensed distance
                matrix is a flat array containing the upper triangular of the distance matrix. Alternatively, an array of shape
                `(number_of_observations, number_of_dimensions)` may be passed in.

        Returns:
            Any: Returns the data that was passed in. The data is not modified.
        """

        if isinstance(self.output_file, (str, bytes, os.PathLike)):
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        plt.figure(figsize=(10, 7))
        shc.dendrogram(shc.linkage(data, method=self.linkage))
        plt.savefig(self.output_file)

        return data
