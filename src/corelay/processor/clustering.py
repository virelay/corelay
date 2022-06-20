"""Clustering Processors

"""
import io
import os
import logging

import sklearn.cluster
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as shc

from .base import Processor, Param
from ..utils import import_or_stub
hdbscan = import_or_stub('hdbscan')  # pylint: disable=invalid-name

LOGGER = logging.getLogger(__name__)


class Clustering(Processor):
    """Clustering Processor

    """
    kwargs = Param(dict, default={})


class KMeans(Clustering):
    """KMeans Clustering

    Parameters
    ----------
    n_clusters: int
        Default: 2
    index: tuple
        Default: empty slice
    kwargs: dict
        See also: :obj:`sklearn.cluster.KMeans`

    See Also
    --------
    :obj:`sklearn.cluster.KMeans`

    """
    n_clusters = Param(int, 2, identifier=True)
    index = Param(tuple, (slice(None),))

    def function(self, data):
        # pylint: disable=not-a-mapping
        return sklearn.cluster.KMeans(n_clusters=self.n_clusters, **self.kwargs).fit_predict(data[self.index])


class HDBSCAN(Clustering):
    """HDBSCAN clustering

    Parameters
    ----------
    n_clusters: int
        Default: 2
    metric: str
        Default: euclidean
    kwargs: dict
        See also: :obj:`hdbscan.HDBSCAN`

    See Also
    --------
    :obj:`hdbscan.HDBSCAN`

    Notes
    -----
    https://github.com/scikit-learn-contrib/hdbscan

    """
    n_clusters = Param(int, 5, identifier=True)
    metric = Param(str, 'euclidean', identifier=True)

    def function(self, data):
        # pylint: disable=not-a-mapping
        clustering = hdbscan.HDBSCAN(min_cluster_size=self.n_clusters, metric=self.metric, **self.kwargs)
        return clustering.fit_predict(data)


class DBSCAN(Clustering):
    """DBSCAN clustering

    Parameters
    ----------
    metric: str
        Default: euclidean
    eps: float
        Default: 0.5
    min_samples: int
        Default: 5
    kwargs: dict
        See also: :obj:`sklearn.cluster.DBSCAN`

    See Also
    --------
    :obj:`sklearn.cluster.DBSCAN`

    """
    metric = Param(str, 'euclidean', identifier=True)
    eps = Param(float, 0.5, identifier=True)
    min_samples = Param(int, 5, identifier=True)

    def function(self, data):
        # pylint: disable=not-a-mapping
        clustering = sklearn.cluster.DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric,
                                            **self.kwargs)
        return clustering.fit_predict(data)


class AgglomerativeClustering(Clustering):
    """Agglomerative clustering

    Parameters
    ----------
    n_clusters: int
        Default: 5
    metric: str
        Options: "euclidean", "l1", "l2", "manhattan", "cosine", or 'precomputed'. Default: "euclidean"
    linkage: str
        Options: "ward", "complete", "average", "single". Default: "ward"
    kwargs: dict
        See also: :obj:`sklearn.cluster.AgglomerativeClustering`

    See Also
    --------
    :obj:`sklearn.cluster.AgglomerativeClustering`

    """
    n_clusters = Param(int, 5, identifier=True)
    metric = Param(str, 'euclidean', identifier=True)
    linkage = Param(str, 'ward', identifier=True)

    def function(self, data):
        # pylint: disable=not-a-mapping
        clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=self.n_clusters, affinity=self.metric,
                                                             linkage=self.linkage, **self.kwargs)
        return clustering.fit_predict(data)


class Dendrogram(Clustering):
    """Dendrogram.

    Parameters
    ----------
    output_path: str
        Path to where the dendrogram is saved,
    metric: str
        Options: "euclidean", "l1", "l2", "manhattan", "cosine", or 'precomputed'. Default: "euclidean"
    linkage: str
        Options: "ward", "complete", "average", "single". Default: "ward"

    """
    output_file = Param((str, io.IOBase), mandatory=True)
    metric = Param(str, 'euclidean')
    linkage = Param(str, 'ward')

    def function(self, data):
        """Saves Dendrogram by default to /tmp/dendrogram.png and returns the input data.

        """
        if isinstance(self.output_file, (str, bytes, os.PathLike)):
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        plt.figure(figsize=(10, 7))
        shc.dendrogram(shc.linkage(data, method=self.linkage))
        plt.savefig(self.output_file)
        return data
