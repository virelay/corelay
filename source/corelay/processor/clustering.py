"""A module that contains processors for clustering algorithms."""

import io
import os
import typing
from collections.abc import Callable
from typing import Annotated, Literal, SupportsIndex, TypeAlias, TypeGuard, get_args

import numpy
import scipy.cluster.hierarchy
import sklearn.base
import sklearn.cluster
from matplotlib import pyplot

import corelay.utils
from corelay.base import Param
from corelay.processor.base import Processor


hdbscan: Callable[..., sklearn.base.ClusterMixin] = corelay.utils.import_or_stub('hdbscan', 'HDBSCAN')
"""Performs the HDBSCAN clustering algorithm on a vector or distance matrix.

Note:
    Since the HDBSCAN library is an optional dependency of CoRelAy, it is imported using the :py:func:`~corelay.utils.import_or_stub` function, which
    tries to import the module/type/function specified. If the import fails, it returns a stub instead, which will raise an exception when used. The
    exception message will tell users how to install the missing dependencies for the functionality to work.

Returns:
    sklearn.base.ClusterMixin: Returns an HDBSCAN cluster estimator, which can be used to fit the data.
"""


class Clustering(Processor):
    """The abstract base class for :py:class:`~corelay.processor.base.Processor` that performs clustering.

    Args:
        is_output (bool): A value indicating whether this :py:class:`Clustering` processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        kwargs (dict[str, typing.Any]): Additional keyword arguments for the clustering algorithm. Defaults to an empty :py:class:`dict`.
    """

    kwargs: Annotated[dict[str, typing.Any], Param(dict, default={})]
    """A :py:class:`dict` of additional keyword arguments for the clustering algorithm. Defaults to an empty :py:class:`dict`."""


class KMeans(Clustering):
    """A clustering :py:class:`~corelay.processor.base.Processor` that performs the k-Means clustering algorithm.

    Args:
        is_output (bool): A value indicating whether this :py:class:`KMeans` clustering processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        kwargs (dict[str, typing.Any]): Additional keyword arguments for the k-Means clustering algorithm. Defaults to an empty :py:class:`dict`.
        n_clusters (int): The number of clusters to form. Defaults to 2.
        index (tuple[int | slice]): The indices of the data to be clustered. Defaults to an empty slice.

    See Also:
        * :py:class:`sklearn.cluster.KMeans`
    """

    n_clusters: Annotated[int, Param(int, 2, identifier=True)]
    """The number of clusters to form. Defaults to 2."""

    index: Annotated[SupportsIndex | tuple[SupportsIndex, ...], Param((SupportsIndex, tuple), (slice(None),))]
    """The indices of the data to be clustered. Defaults to an empty slice."""

    def function(self, data: typing.Any) -> typing.Any:
        """Performs k-Means clustering on the data.

        Args:
            data (typing.Any): The data to be clustered. The data should be a NumPy array or a sparse matrix.

        Returns:
            typing.Any: Returns the a NumPy array of shape `(number_of_samples,)`, that contains the cluster labels, where each label corresponds to a
            cluster assigned to the data point.
        """

        input_data: numpy.ndarray[typing.Any, typing.Any] = data
        return sklearn.cluster.KMeans(n_clusters=self.n_clusters, **self.kwargs).fit_predict(input_data[self.index])


class HDBSCAN(Clustering):
    """A clustering :py:class:`~corelay.processor.base.Processor` that performs the HDBSCAN clustering algorithm.

    Args:
        is_output (bool): A value indicating whether this :py:class:`HDBSCAN` clustering processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        kwargs (dict[str, typing.Any]): Additional keyword arguments for the HDBSCAN clustering algorithm. Defaults to an empty :py:class:`dict`.
        n_clusters (int): The number of clusters to form. Defaults to 2.
        metric (str): The distance metric to use. This can be one of "euclidean", "l2" (equivalent to "euclidean"), "minkowski", "p" (equivalent to
            "minkowski"), "manhattan", "cityblock" (equivalent to "manhattan"), "l1" (equivalent to "manhattan"), "chebyshev", "infinity" (equivalent
            to "chebyshev"), "seuclidean", "mahalanobis", "wminkowski", "hamming", "canberra", "braycurtis", "matching", "jaccard", "dice",
            "kulsinski", "rogerstanimoto", "russellrao", "sokalmichener", "sokalsneath", "haversine", "cosine" (since the cosine distance is not a
            true distance measure, it is not supported; using "cosine" will use the "arccos" distance instead), and "arccos". Defaults to "euclidean".

    See Also:
        * :py:class:`hdbscan.hdbscan_.HDBSCAN`

    Notes:
        GitHub repository including documentation for HDBSCAN: https://github.com/scikit-learn-contrib/hdbscan.
    """

    n_clusters: Annotated[int, Param(int, 5, identifier=True)]
    """The number of clusters to form. Defaults to 5."""

    metric: Annotated[str, Param(str, 'euclidean', identifier=True)]
    """The distance metric to use. Defaults to "euclidean"."""

    def function(self, data: typing.Any) -> typing.Any:
        """Performs the HDBSCAN clustering algorithm on the data.

        Args:
            data (typing.Any): The data to be clustered. The data should be a NumPy array of shape `(number_of_samples, number_of_features)` or a
                sparse matrix.

        Returns:
            typing.Any: Returns a NumPy array of shape `(number_of_samples,)`, that contains the cluster labels, where each label corresponds to a
            cluster assigned to the data point.
        """

        clustering = hdbscan(min_cluster_size=self.n_clusters, metric=self.metric, **self.kwargs)
        return clustering.fit_predict(data)


class DBSCAN(Clustering):
    """A clustering :py:class:`~corelay.processor.base.Processor` that performs the DBSCAN clustering algorithm.

    Args:
        is_output (bool): A value indicating whether this :py:class:`DBSCAN` clustering processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        kwargs (dict[str, typing.Any]): Additional keyword arguments for the DBSCAN clustering algorithm. Defaults to an empty :py:class:`dict`.
        metric (str): The distance metric to use. Defaults to "euclidean".
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum
            bound on the distances of points within a cluster. This is the most important DBSCAN parameter to choose appropriately for your data set
            and distance function. Defaults to 0.5.
        min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the
            point itself. If ``min_samples`` is set to a higher value, DBSCAN will find denser clusters, whereas if it is set to a lower value, the
            found clusters will be more sparse. Defaults to 5.

    See Also:
        * :py:class:`sklearn.cluster.DBSCAN`
    """

    metric: Annotated[str, Param(str, 'euclidean', identifier=True)]
    """The distance metric to use. Can be one of

    * "cityblock"
    * "cosine"
    * "euclidean"
    * "haversine"
    * "l1"
    * "l2"
    * "manhattan"
    * "nan_euclidean"
    * "precomputed".

    Defaults to "euclidean".
    """

    eps: Annotated[float, Param(float, 0.5, identifier=True)]
    """The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the
    distances of points within a cluster. This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
    Defaults to 0.5.
    """

    min_samples: Annotated[int, Param(int, 5, identifier=True)]
    """The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself. If
    :py:attr:`DBSCAN.min_samples` is set to a higher value, DBSCAN will find denser clusters, whereas if it is set to a lower value, the found
    clusters will be more sparse. Defaults to 5.
    """

    def function(self, data: typing.Any) -> typing.Any:
        """Performs the DBSCAN clustering algorithm on the data.

        Args:
            data (typing.Any): The data to be clustered. The data should be a NumPy array of shape `(number_of_samples, number_of_features)` or a
                sparse matrix.

        Returns:
            typing.Any: Returns a NumPy array of shape `(number_of_samples,)`, that contains the cluster labels, where each label corresponds to a
            cluster assigned to the data point. Noisy points will be labeled as -1.
        """

        clustering = sklearn.cluster.DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            **self.kwargs
        )
        return clustering.fit_predict(data)


class AgglomerativeClustering(Clustering):
    """A clustering :py:class:`~corelay.processor.base.Processor` that performs the Agglomerative Clustering algorithm.

    Args:
        is_output (bool): A value indicating whether this :py:class:`AgglomerativeClustering` clustering processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        kwargs (dict[str, typing.Any]): Additional keyword arguments for the agglomerative clustering algorithm. Defaults to an empty
            :py:class:`dict`.
        n_clusters (int): The number of clusters to form. Defaults to 5.
        metric (str): The distance metric to use. Defaults to "euclidean".
        linkage (str): The linkage method to use. This determines which distance to use between two newly formed clusters. The algorithm will merge
            the pairs of clusters that minimize this method. Defaults to "ward".

    See Also:
        * :py:class:`sklearn.cluster.AgglomerativeClustering`
    """

    n_clusters: Annotated[int, Param(int, 5, identifier=True)]
    """The number of clusters to form. Defaults to 5."""

    metric: Annotated[str, Param(str, 'euclidean', identifier=True)]
    """The distance metric to use. Can be one of

    * "euclidean"
    * "l1"
    * "l2"
    * "manhattan"
    * "cosine"
    * "precomputed".

    Defaults to "euclidean".
    """

    linkage: Annotated[str, Param(str, 'ward', identifier=True)]
    """The linkage method to use. This determines which distance to use between two newly formed clusters. The algorithm will merge the pairs of
    clusters that minimize this method. Can be one of

    * "ward" minimizes the variance of the clusters being merged.
    * "average" uses the average of the distances of each observation of the two clusters.
    * "complete" uses the maximum distances between all observations of the two clusters.
    * "single" uses the minimum of the distances between all observations of the two clusters.

    Defaults to "ward".
    """

    def function(self, data: typing.Any) -> typing.Any:
        """Performs the Agglomerative Clustering algorithm on the data.

        Args:
            data (typing.Any): The data to be clustered. The data should be a NumPy array of shape `(number_of_samples, number_of_features)` or
                `(number_of_samples, number_of_samples)`, or a sparse matrix.

        Returns:
            typing.Any: Returns a NumPy array of shape `(number_of_samples,)`, that contains the cluster labels, where each label corresponds to a
            cluster assigned to the data point.
        """

        clustering = sklearn.cluster.AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric=self.metric,
            linkage=self.linkage,
            **self.kwargs
        )
        return clustering.fit_predict(data)


class Dendrogram(Clustering):
    """A clustering :py:class:`~corelay.processor.base.Processor` that generates plots the hierarchical clustering as a dendrogram.

    Args:
        is_output (bool): A value indicating whether this :py:class:`Dendrogram` clustering processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        kwargs (dict[str, typing.Any]): Additional keyword arguments for the hierarchical clustering algorithm. Defaults to an empty :py:class:`dict`.
        output_file (str | io.IOBase): The path to a file or a file descriptor to save the dendrogram plot to.
        metric (str): The distance metric to use for the clustering. Defaults to "euclidean".
        linkage (str): The linkage criterion to use. This determines which distance to use between sets of observation. Defaults to "ward".
    """

    output_file: Annotated[str | io.IOBase, Param((str, io.IOBase), mandatory=True)]
    """The path to a file or a file descriptor to save the dendrogram plot to."""

    metric: Annotated[str, Param(str, 'euclidean')]
    """The distance metric to use for the clustering. This can be one of "braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine",
    "dice", "euclidean", "hamming", "jaccard", "jensenshannon", "kulczynski1", "mahalanobis", "minkowski", "rogerstanimoto", "russellrao",
    "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", or "yule". Defaults to "euclidean".
    """

    linkage: Annotated[str, Param(str, 'ward')]
    """The linkage method used by the Dendrogram :py:class:`~corelay.processor.base.Processor`. The linkage method is used to determine the distance
    between two newly formed clusters when performing hierarchical clustering. The hierarchical clustering algorithm used by the Dendrogram
    :py:class:`~corelay.processor.base.Processor` will merge the pairs of clusters that minimize this method. The following linkage methods
    are supported:

    * "ward" minimizes the variance of the clusters being merged.
    * "average" uses the average of the distances of each observation of the two clusters.
    * "complete" uses the maximum distances between all observations of the two clusters.
    * "single" uses the minimum of the distances between all observations of the two clusters.
    * "centroid" the centroid of the new cluster that would be formed by merging the two clusters.
    * "median" uses the median of the centroids of the two clusters.
    * "weighted" assigns the weighted distance between the two original clusters and a third remaining cluster to the new cluster.

    Defaults to "ward".
    """

    def function(self, data: typing.Any) -> typing.Any:
        """Performs the hierarchical clustering algorithm on the data and generates a dendrogram plot.

        Args:
            data (typing.Any): The data to be clustered. The data should be a NumPy array that contains a condensed distance matrix. A condensed
                distance matrix is a flat array containing the upper triangular of the distance matrix. Alternatively, an array of shape
                `(number_of_observations, number_of_dimensions)` may be passed in.

        Raises:
            ValueError: The linkage method is invalid.

        Returns:
            typing.Any: Returns the data that was passed in. The data is not modified.
        """

        # This is necessary to ensure that MyPy does not complain that the linkage method is not valid; ideally, we would use literals ourselves, but
        # unfortunately, Sphinx AutoDoc cannot handle type aliases correctly unless we use Postponed Evaluation of Annotations (PEP 563), which in
        # turn breaks our usage of typing.Annotated for slots
        LinkageMethod: TypeAlias = Literal['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']
        linkage_methods = list(get_args(LinkageMethod))

        def check_if_linkage_method_is_valid(linkage: str) -> TypeGuard[LinkageMethod]:
            return linkage in linkage_methods

        if not check_if_linkage_method_is_valid(self.linkage):
            raise ValueError(f'Invalid linkage method: {self.linkage}.')

        if isinstance(self.output_file, str):
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        pyplot.figure(figsize=(10, 7))
        scipy.cluster.hierarchy.dendrogram(scipy.cluster.hierarchy.linkage(data, method=self.linkage))
        pyplot.savefig(self.output_file)

        return data
