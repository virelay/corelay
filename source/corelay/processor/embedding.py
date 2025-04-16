"""Embedding Processors

"""
import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.decomposition import PCA

from ..utils import import_or_stub
from .base import Processor, Param
UMAP = import_or_stub('umap', 'UMAP')


class Embedding(Processor):
    """Embedding Processor base class

    """
    kwargs = Param(dict, {})


class EigenDecomposition(Embedding):
    """Eigenvalue Decomposition

    """
    n_eigval = Param(int, 32, identifier=True)
    which = Param(str, 'LM')
    normalize = Param(bool, True, identifier=True)

    @property
    def _output_repr(self):
        return '(eigval:np.ndarray, eigvec:np.ndarray)'

    def function(self, data):
        """Compute spectral embedding of `data`

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            data with samples in rows

        Returns
        -------
        :obj:`numpy.ndarray`
            Eigenvalues for spectral embedding
        :obj:`numpy.ndarray`
            Spectral embedding (eigenvectors)

        Note
        ----
        We use the fact that (I-A)v = (1-λ)v and thus compute the largest eigenvalues of the identity minus the
        data and return one minus the eigenvalue.

        """
        # pylint: disable=not-a-mapping
        eigval, eigvec = eigsh(data, k=self.n_eigval, which=self.which, **self.kwargs)
        eigval = 1. - eigval

        if self.normalize:
            eigvec /= np.linalg.norm(eigvec, axis=1, keepdims=True)
        return eigval, eigvec


class TSNEEmbedding(Embedding):
    """TSNE Embedding

    """
    n_components = Param(int, default=2, identifier=True)
    metric = Param(str, default='euclidean', identifier=True)
    perplexity = Param(float, default=30., identifier=True)
    early_exaggeration = Param(float, default=12., identifier=True)

    def function(self, data):
        # pylint: disable=not-a-mapping
        tsne = TSNE(n_components=self.n_components,
                    metric=self.metric,
                    perplexity=self.perplexity,
                    early_exaggeration=self.early_exaggeration,
                    **self.kwargs)
        emb = tsne.fit_transform(data)
        return emb


class PCAEmbedding(Embedding):
    """PCA Embedding

    """
    n_components = Param(int, default=2, identifier=True)
    whiten = Param(bool, default=False, identifier=True)

    def function(self, data):
        # pylint: disable=not-a-mapping
        pca = PCA(n_components=self.n_components, whiten=self.whiten, **self.kwargs)
        emb = pca.fit_transform(data)
        return emb


class LLEEmbedding(Embedding):
    """LocallyLinearEmbedding

    """
    n_components = Param(int, default=2, identifier=True)
    n_neighbors = Param(int, default=5, identifier=True)

    def function(self, data):
        # pylint: disable=not-a-mapping
        lle = LocallyLinearEmbedding(n_neighbors=self.n_neighbors, n_components=self.n_components,
                                     **self.kwargs)
        emb = lle.fit_transform(data)
        return emb


class UMAPEmbedding(Embedding):
    """UMAPEmbedding: https://umap-learn.readthedocs.io/en/latest/index.html

    """
    n_neighbors = Param(int, default=15, identifier=True)
    min_dist = Param(float, default=0.1, identifier=True)
    metric = Param(str, default='correlation', identifier=True)

    def function(self, data):
        umap = UMAP(n_neighbors=self.n_neighbors, min_dist=self.min_dist, metric=self.metric)
        emb = umap.fit_transform(data)
        return emb
