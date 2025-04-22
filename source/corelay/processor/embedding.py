"""A module that contains processors for embedding algorithms."""

from collections.abc import Callable
from typing import Annotated, Any, Literal

import numpy
from numpy.typing import NDArray
from scipy.sparse.linalg import eigsh
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding

from corelay.base import Param
from corelay.processor.base import Processor
from corelay.processor.distance import PairWiseDistanceMeasure
from corelay.utils import import_or_stub


UMAP: Callable[..., TransformerMixin] = import_or_stub('umap', 'UMAP')
"""Performs the Uniform Manifold Approximation and Projection (UMAP) dimensionality reduction algorithm, which will find a low dimensional embedding
of the data that approximates an underlying manifold.

Returns:
    TransformerMixin: Returns a UMAP cluster estimator, which can be used to fit the data.

Note:
    Since the UMAP library is an optional dependency of CoRelAy, it is imported using the ``import_or_stub`` function, which tries to import the
    module/type/function specified. If the import fails, it returns a stub instead, which will raise an exception when used. The exception message
    will tell users how to install the missing dependencies for the functionality to work.
"""


class Embedding(Processor):
    """The abstract base class for embedding processors.

    Args:
        is_output (bool, optional): A value indicating whether this ``Embedding`` processor is the output of a ``Pipeline``. Defaults to `False`.
        is_checkpoint (bool | None, optional): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to `False`.
        io (Storable | None, optional): An IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this
            run or in subsequent runs of the ``Pipeline``. Defaults to an instance of ``NoStorage``.
        kwargs (dict[str, Any], optional): Additional keyword arguments for the embedding algorithm. Defaults to an empty dictionary.
    """

    kwargs: Annotated[dict[str, Any], Param(dict, {})]
    """Additional keyword arguments to pass to the embedding function."""


EigenvalueType = Literal['LM', 'SM', 'LA', 'SA', 'BE']
"""The type of eigenvalues and eigenvectors to compute. The options are:

- "LM": Largest (in magnitude) eigenvalues.
- "SM": Smallest (in magnitude) eigenvalues.
- "LA": Largest (algebraic) eigenvalues.
- "SA": Smallest (algebraic) eigenvalues.
- "BE": Half (k/2) from each end of the spectrum.

Note:
    If the input is a complex Hermitian matrix, 'BE' is invalid.
"""


class EigenDecomposition(Embedding):
    """A spectral embedding ``Processor`` that performs eigenvalue decomposition.

    Args:
        is_output (bool, optional): A value indicating whether this ``EigenDecomposition`` embedding processor is the output of a ``Pipeline``.
            Defaults to `False`.
        is_checkpoint (bool | None, optional): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to `False`.
        io (Storable | None, optional): An IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this
            run or in subsequent runs of the ``Pipeline``. Defaults to an instance of ``NoStorage``.
        kwargs (dict[str, Any], optional): Additional keyword arguments for the eigenvalue decomposition embedding algorithm. Defaults to an empty
        n_eigval (int, optional): The number of eigenvalues and eigenvectors to compute. Defaults to 32.
        which (str, optional): The type of eigenvalues and eigenvectors to compute. Defaults to "LM" (largest in magnitude).
        normalize (bool, optional): A value indicating whether to normalize the eigenvectors. Defaults to `True`.
            dictionary.
    """

    n_eigval: Annotated[int, Param(int, 32, identifier=True)]
    """The number of eigenvalues and eigenvectors to compute. Defaults to 32."""

    which: Annotated[EigenvalueType, Param(str, 'LM')]
    """The type of eigenvalues and eigenvectors to compute. Defaults to "LM" (largest in magnitude)."""

    normalize: Annotated[bool, Param(bool, True, identifier=True)]
    """A value indicating whether to normalize the eigenvectors. Defaults to True."""

    @property
    def _output_repr(self) -> str:
        """Gets a string representation of the output of the function.

        Returns:
            str: Returns a string representation of the output of the function.
        """

        return '(eigval: numpy.ndarray, eigvec: numpy.ndarray)'

    def function(self, data: Any) -> Any:
        """Computes the spectral embedding of the input data using eigenvalue decomposition.

        Args:
            data (Any): The input data to be embedded. The data should be a NumPy array of shape `(number_of_samples, number_of_features)`.

        Returns:
            Any: Returns a tuple containing the eigenvalues and eigenvectors of the input data.

        Note:
            We use the fact that (I-A)v = (1-λ)v and thus compute the largest eigenvalues of the identity minus the data and return one minus the
            eigenvalue.
        """

        input_data: NDArray[Any] = data
        eigenvalues: NDArray[numpy.float64]
        eigenvectors: NDArray[numpy.float64]
        eigenvalues, eigenvectors = eigsh(input_data, k=self.n_eigval, which=self.which, **self.kwargs)  # type: ignore[arg-type]
        eigenvalues = 1.0 - eigenvalues

        if self.normalize:
            eigenvectors /= numpy.linalg.norm(eigenvectors, axis=1, keepdims=True)

        return eigenvalues, eigenvectors


class TSNEEmbedding(Embedding):
    """An embedding ``Processor`` that uses the t-SNE algorithm to reduce the dimensionality of the input data.

    Args:
        is_output (bool, optional): A value indicating whether this ``TSNEEmbedding`` embedding processor is the output of a ``Pipeline``. Defaults to
            `False`.
        is_checkpoint (bool | None, optional): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to `False`.
        io (Storable | None, optional): An IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this
            run or in subsequent runs of the ``Pipeline``. Defaults to an instance of ``NoStorage``.
        kwargs (dict[str, Any], optional): Additional keyword arguments for the t-SNE embedding algorithm. Defaults to an empty dictionary.
        n_components (int, optional): The number of dimensions to reduce the data to. Defaults to 2.
        metric (str, optional): The distance metric to use. Defaults to "euclidean".
        perplexity (float, optional): The perplexity parameter for the t-SNE algorithm. Defaults to 30.
        early_exaggeration (float, optional): The early exaggeration parameter for the t-SNE algorithm. Defaults to 12.
    """

    n_components: Annotated[int, Param(int, default=2, identifier=True)]
    """The number of dimensions to reduce the data to. Defaults to 2."""

    metric: Annotated[PairWiseDistanceMeasure, Param(str, default='euclidean', identifier=True)]
    """The distance metric to use. Defaults to "euclidean"."""

    perplexity: Annotated[float, Param(float, default=30.0, identifier=True)]
    """The perplexity parameter for the t-SNE algorithm. Defaults to 30."""

    early_exaggeration: Annotated[float, Param(float, default=12.0, identifier=True)]
    """The early exaggeration parameter for the t-SNE algorithm. Defaults to 12."""

    def function(self, data: Any) -> Any:
        """Computes the t-SNE embedding of the input data.

        Args:
            data (Any): The input data to be embedded. The data should be a NumPy array of shape `(number_of_samples, number_of_features)` or
                `(number_of_samples, number_of_samples)`.

        Returns:
            Any: Returns the t-SNE embedding of the input data as a NumPy array of shape `(number_of_samples, number_of_components)`.
        """

        tsne = TSNE(
            n_components=self.n_components,
            init='random',
            metric=self.metric,
            perplexity=self.perplexity,
            early_exaggeration=self.early_exaggeration,
            **self.kwargs
        )
        return tsne.fit_transform(data)


class PCAEmbedding(Embedding):
    """An embedding ``Processor`` that uses the principal component analysis (PCA) algorithm to reduce the dimensionality of the input data.

    Args:
        is_output (bool, optional): A value indicating whether this ``PCAEmbedding`` embedding processor is the output of a ``Pipeline``. Defaults to
            `False`.
        is_checkpoint (bool | None, optional): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to `False`.
        io (Storable | None, optional): An IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this
            run or in subsequent runs of the ``Pipeline``. Defaults to an instance of ``NoStorage``.
        kwargs (dict[str, Any], optional): Additional keyword arguments for the PCA embedding algorithm. Defaults to an empty dictionary.
        n_components (int, optional): The number of dimensions to reduce the data to. Defaults to 2.
        whiten (bool, optional): A value indicating whether to whiten the data. Defaults to `False`.
    """

    n_components: Annotated[int, Param(int, default=2, identifier=True)]
    """The number of dimensions to reduce the data to. Defaults to 2."""

    whiten: Annotated[bool, Param(bool, default=False, identifier=True)]
    """A value indicating whether to whiten the data. Defaults to `False`."""

    def function(self, data: Any) -> Any:
        """Computes the PCA embedding of the input data.

        Args:
            data (Any): The input data to be embedded. The data should be a NumPy array of shape `(number_of_samples, number_of_features)`.

        Returns:
            Any: Returns the PCA embedding of the input data as a NumPy array of shape `(number_of_samples, number_of_components)`.
        """

        pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            **self.kwargs
        )
        return pca.fit_transform(data)


class LLEEmbedding(Embedding):
    """An embedding ``Processor`` that uses the locally linear embedding (LLE) algorithm to reduce the dimensionality of the input data.

    Args:
        is_output (bool, optional): A value indicating whether this ``LLEEmbedding`` embedding processor is the output of a ``Pipeline``. Defaults to
            `False`.
        is_checkpoint (bool | None, optional): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to `False`.
        io (Storable | None, optional): An IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this
            run or in subsequent runs of the ``Pipeline``. Defaults to an instance of ``NoStorage``.
        kwargs (dict[str, Any], optional): Additional keyword arguments for the LLE embedding algorithm. Defaults to an empty dictionary.
        n_components (int, optional): The number of dimensions to reduce the data to. Defaults to 2.
        n_neighbors (int, optional): The number of neighbors to use for the LLE algorithm. Defaults to 5.
    """

    n_components: Annotated[int, Param(int, default=2, identifier=True)]
    """The number of dimensions to reduce the data to. Defaults to 2."""

    n_neighbors: Annotated[int, Param(int, default=5, identifier=True)]
    """The number of neighbors to use for the LLE algorithm. Defaults to 5."""

    def function(self, data: Any) -> Any:
        """Computes the LLE embedding of the input data.

        Args:
            data (Any): The input data to be embedded. The data should be a NumPy array of shape `(number_of_samples, number_of_features)`.

        Returns:
            Any: Returns the LLE embedding of the input data as a NumPy array of shape `(number_of_samples, number_of_components)`.
        """

        lle = LocallyLinearEmbedding(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            **self.kwargs
        )
        return lle.fit_transform(data)


UMAPDistanceMetric = Literal[
    'euclidean',
    'manhattan',
    'chebyshev',
    'minkowski',
    'canberra',
    'braycurtis',
    'mahalanobis',
    'wminkowski',
    'seuclidean',
    'cosine',
    'correlation',
    'haversine',
    'hamming',
    'jaccard',
    'dice',
    'russelrao',
    'kulsinski',
    'll_dirichlet',
    'hellinger',
    'rogerstanimoto',
    'sokalmichener',
    'sokalsneath',
    'yule'
]
"""An enumeration of the distance measures supported by ``umap.UMAP``."""


class UMAPEmbedding(Embedding):
    """An embedding ``Processor`` that uses the Uniform Manifold Approximation and Projection (UMAP) algorithm to reduce the dimensionality of the
    input data.

    Args:
        is_output (bool, optional): A value indicating whether this ``UMAPEmbedding`` embedding processor is the output of a ``Pipeline``. Defaults to
            `False`.
        is_checkpoint (bool | None, optional): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to `False`.
        io (Storable | None, optional): An IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this
            run or in subsequent runs of the ``Pipeline``. Defaults to an instance of ``NoStorage``.
        kwargs (dict[str, Any], optional): Additional keyword arguments for the UMAP embedding algorithm. Defaults to an empty dictionary.
        n_neighbors (int, optional): The number of neighbors to use for the UMAP algorithm. Defaults to 15.
        min_dist (float, optional): The minimum distance between points in the UMAP algorithm. Defaults to 0.1.
        metric (str, optional): The distance metric to use for the UMAP algorithm. Defaults to "correlation".
    """

    n_neighbors: Annotated[int, Param(int, default=15, identifier=True)]
    """The number of neighbors to use for the UMAP algorithm. Defaults to 15."""

    min_dist: Annotated[float, Param(float, default=0.1, identifier=True)]
    """The minimum distance between points in the UMAP algorithm. Defaults to 0.1."""

    metric: Annotated[UMAPDistanceMetric, Param(str, default='correlation', identifier=True)]
    """The distance metric to use for the UMAP algorithm. Defaults to "correlation"."""

    def function(self, data: Any) -> Any:
        """Computes the UMAP embedding of the input data.

        Args:
            data (Any): The input data to be embedded. The data should be a NumPy array of shape `(number_of_samples, number_of_features)`.

        Returns:
            Any: Returns the UMAP embedding of the input data as a NumPy array of shape `(number_of_samples, number_of_new_features)`.

        Note:
            For information on the UMAP algorithm, see the `UMAP documentation <https://umap-learn.readthedocs.io/en/latest/index.html>`_.
        """

        umap = UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric
        )
        return umap.fit_transform(data)
