"""A module that contains processors for embedding algorithms."""

import typing
from collections.abc import Callable
from typing import Annotated, Literal, TypeAlias, TypeGuard, get_args

import numpy
import sklearn.base
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding

import corelay.utils
from corelay.base import Param
from corelay.processor.base import Processor


UMAP: Callable[..., sklearn.base.TransformerMixin] = corelay.utils.import_or_stub('umap', 'UMAP')
"""Performs the Uniform Manifold Approximation and Projection (UMAP) dimensionality reduction algorithm, which will find a low dimensional embedding
of the data that approximates an underlying manifold.

Note:
    Since the UMAP library is an optional dependency of CoRelAy, it is imported using the :py:func:`corelay.utils.import_or_stub` function, which
    tries to import the module/type/function specified. If the import fails, it returns a stub instead, which will raise an exception when used. The
    exception message will tell users how to install the missing dependencies for the functionality to work.

Returns:
    sklearn.base.TransformerMixin: Returns a UMAP cluster estimator, which can be used to fit the data.
"""


class Embedding(Processor):
    """The abstract base class for embedding processors.

    Args:
        is_output (bool): A value indicating whether this :py:class:`Embedding` processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        kwargs (dict[str, typing.Any]): Additional keyword arguments for the embedding algorithm. Defaults to an empty :py:class:`dict`.
    """

    kwargs: Annotated[dict[str, typing.Any], Param(dict, {})]
    """Additional keyword arguments to pass to the embedding function."""


class EigenDecomposition(Embedding):
    """A spectral embedding :py:class:`~corelay.processor.base.Processor` that performs eigenvalue decomposition.

    Args:
        is_output (bool): A value indicating whether this :py:class:`EigenDecomposition` embedding processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        kwargs (dict[str, typing.Any]): Additional keyword arguments for the eigenvalue decomposition embedding algorithm. Defaults to an empty
        n_eigval (int): The number of eigenvalues and eigenvectors to compute. Defaults to 32.
        which (str): The type of eigenvalues and eigenvectors to compute. Defaults to "LM" (largest in magnitude).
        normalize (bool): A value indicating whether to normalize the eigenvectors. Defaults to :py:obj:`True`.
    """

    n_eigval: Annotated[int, Param(int, 32, identifier=True)]
    """The number of eigenvalues and eigenvectors to compute. Defaults to 32."""

    which: Annotated[str, Param(str, 'LM')]
    """The type of eigenvalues and eigenvectors to compute. The options are:

    * "LM": Largest (in magnitude) eigenvalues.
    * "SM": Smallest (in magnitude) eigenvalues.
    * "LA": Largest (algebraic) eigenvalues.
    * "SA": Smallest (algebraic) eigenvalues.
    * "BE": Half (k/2) from each end of the spectrum.

    Defaults to "LM" (largest in magnitude).

    Note:
        If the input is a complex Hermitian matrix, 'BE' is invalid.
    """

    normalize: Annotated[bool, Param(bool, True, identifier=True)]
    """A value indicating whether to normalize the eigenvectors. Defaults to True."""

    @property
    def _output_repr(self) -> str:
        """Gets a :py:class:`str` representation of the output of the function.

        Returns:
            str: Returns a :py:class:`str` representation of the output of the function.
        """

        return '(eigval: numpy.ndarray, eigvec: numpy.ndarray)'

    def function(self, data: typing.Any) -> typing.Any:
        """Computes the spectral embedding of the input data using eigenvalue decomposition.

        Note:
            We use the fact that (I-A)v = (1-λ)v and thus compute the largest eigenvalues of the identity minus the data and return one minus the
            eigenvalue.

        Args:
            data (typing.Any): The input data to be embedded. The data should be a NumPy array of shape `(number_of_samples, number_of_features)`.

        Raises:
            ValueError: The eigenvalue and eigenvector type is not valid.

        Returns:
            typing.Any: Returns a tuple containing the eigenvalues and eigenvectors of the input data.
        """

        # This is necessary to ensure that MyPy does not complain that the "which" argument is not valid; ideally, we would use literals ourselves,
        # but unfortunately, Sphinx AutoDoc cannot handle type aliases correctly unless we use Postponed Evaluation of Annotations (PEP 563), which in
        # turn breaks our usage of typing.Annotated for slots
        EigenvalueAndEigenvectorType: TypeAlias = Literal['LM', 'SM', 'LR', 'SR', 'LI', 'SI']
        eigenvalue_and_eigenvector_types = list(get_args(EigenvalueAndEigenvectorType))

        def check_if_eigenvalue_and_eigenvector_type_is_valid(eigenvalue_and_eigenvector_type: str) -> TypeGuard[EigenvalueAndEigenvectorType]:
            return eigenvalue_and_eigenvector_type in eigenvalue_and_eigenvector_types

        if not check_if_eigenvalue_and_eigenvector_type_is_valid(self.which):
            raise ValueError(f'Invalid eigenvalue and eigenvector type: {self.which}.')

        input_data: numpy.ndarray[typing.Any, numpy.dtype[numpy.floating] | numpy.dtype[numpy.integer]] = data
        eigenvalues: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]
        eigenvectors: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]
        eigenvalues, eigenvectors = eigsh(input_data, k=self.n_eigval, which=self.which, **self.kwargs)
        eigenvalues = 1.0 - eigenvalues

        if self.normalize:
            eigenvectors /= numpy.linalg.norm(eigenvectors, axis=1, keepdims=True)

        return eigenvalues, eigenvectors


class TSNEEmbedding(Embedding):
    """An embedding :py:class:`~corelay.processor.base.Processor` that uses the t-SNE algorithm to reduce the dimensionality of the input data.

    Args:
        is_output (bool): A value indicating whether this :py:class:`TSNEEmbedding` embedding processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        kwargs (dict[str, typing.Any]): Additional keyword arguments for the t-SNE embedding algorithm. Defaults to an empty :py:class:`dict`.
        n_components (int): The number of dimensions to reduce the data to. Defaults to 2.
        metric (str): The distance metric to use. Defaults to "euclidean".
        perplexity (float): The perplexity parameter for the t-SNE algorithm. Defaults to 30.
        early_exaggeration (float): The early exaggeration parameter for the t-SNE algorithm. Defaults to 12.
    """

    n_components: Annotated[int, Param(int, default=2, identifier=True)]
    """The number of dimensions to reduce the data to. Defaults to 2."""

    metric: Annotated[str, Param(str, default='euclidean', identifier=True)]
    """The distance metric to use. Can be one of

    * "braycurtis"
    * "canberra"
    * "chebychev", "chebyshev", "cheby", "cheb", "ch"
    * "cityblock", "cblock", "cb", "c"
    * "correlation", "co"
    * "cosine", "cos"
    * "dice"
    * "euclidean", "euclid", "eu", "e"
    * "hamming", "hamm", "ha", "h"
    * "minkowski", "mi", "m"
    * "pnorm"
    * "jaccard", "jacc", "ja", "j"
    * "jensenshannon", "js"
    * "kulczynski1"
    * "mahalanobis", "mahal", "mah"
    * "rogerstanimoto"
    * "russellrao"
    * "seuclidean", "se", "s"
    * "sokalmichener"
    * "sokalsneath"
    * "sqeuclidean", "sqe", "sqeuclid"
    * "yule"

    Defaults to "euclidean".
    """

    perplexity: Annotated[float, Param(float, default=30.0, identifier=True)]
    """The perplexity parameter for the t-SNE algorithm. Defaults to 30."""

    early_exaggeration: Annotated[float, Param(float, default=12.0, identifier=True)]
    """The early exaggeration parameter for the t-SNE algorithm. Defaults to 12."""

    def function(self, data: typing.Any) -> typing.Any:
        """Computes the t-SNE embedding of the input data.

        Args:
            data (typing.Any): The input data to be embedded. The data should be a NumPy array of shape `(number_of_samples, number_of_features)` or
                `(number_of_samples, number_of_samples)`.

        Returns:
            typing.Any: Returns the t-SNE embedding of the input data as a NumPy array of shape `(number_of_samples, number_of_components)`.
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
    """An embedding :py:class:`~corelay.processor.base.Processor` that uses the principal component analysis (PCA) algorithm to reduce the
    dimensionality of the input data.

    Args:
        is_output (bool): A value indicating whether this :py:class:`PCAEmbedding` embedding processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        kwargs (dict[str, typing.Any]): Additional keyword arguments for the PCA embedding algorithm. Defaults to an empty :py:class:`dict`.
        n_components (int): The number of dimensions to reduce the data to. Defaults to 2.
        whiten (bool): A value indicating whether to whiten the data. Defaults to :py:obj:`False`.
    """

    n_components: Annotated[int, Param(int, default=2, identifier=True)]
    """The number of dimensions to reduce the data to. Defaults to 2."""

    whiten: Annotated[bool, Param(bool, default=False, identifier=True)]
    """A value indicating whether to whiten the data. Defaults to :py:obj:`False`."""

    def function(self, data: typing.Any) -> typing.Any:
        """Computes the PCA embedding of the input data.

        Args:
            data (typing.Any): The input data to be embedded. The data should be a NumPy array of shape `(number_of_samples, number_of_features)`.

        Returns:
            typing.Any: Returns the PCA embedding of the input data as a NumPy array of shape `(number_of_samples, number_of_components)`.
        """

        pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            **self.kwargs
        )
        return pca.fit_transform(data)


class LLEEmbedding(Embedding):
    """An embedding :py:class:`~corelay.processor.base.Processor` that uses the locally linear embedding (LLE) algorithm to reduce the dimensionality
    of the input data.

    Args:
        is_output (bool): A value indicating whether this :py:class:`LLEEmbedding` embedding processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        kwargs (dict[str, typing.Any]): Additional keyword arguments for the LLE embedding algorithm. Defaults to an empty :py:class:`dict`.
        n_components (int): The number of dimensions to reduce the data to. Defaults to 2.
        n_neighbors (int): The number of neighbors to use for the LLE algorithm. Defaults to 5.
    """

    n_components: Annotated[int, Param(int, default=2, identifier=True)]
    """The number of dimensions to reduce the data to. Defaults to 2."""

    n_neighbors: Annotated[int, Param(int, default=5, identifier=True)]
    """The number of neighbors to use for the LLE algorithm. Defaults to 5."""

    def function(self, data: typing.Any) -> typing.Any:
        """Computes the LLE embedding of the input data.

        Args:
            data (typing.Any): The input data to be embedded. The data should be a NumPy array of shape `(number_of_samples, number_of_features)`.

        Returns:
            typing.Any: Returns the LLE embedding of the input data as a NumPy array of shape `(number_of_samples, number_of_components)`.
        """

        lle = LocallyLinearEmbedding(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            **self.kwargs
        )
        return lle.fit_transform(data)


class UMAPEmbedding(Embedding):
    """An embedding :py:class:`~corelay.processor.base.Processor` that uses the Uniform Manifold Approximation and Projection (UMAP) algorithm to
    reduce the dimensionality of the input data.

    Args:
        is_output (bool): A value indicating whether this :py:class:`UMAPEmbedding` embedding processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        kwargs (dict[str, typing.Any]): Additional keyword arguments for the UMAP embedding algorithm. Defaults to an empty :py:class:`dict`.
        n_neighbors (int): The number of neighbors to use for the UMAP algorithm. Defaults to 15.
        min_dist (float): The minimum distance between points in the UMAP algorithm. Defaults to 0.1.
        metric (str): The distance metric to use for the UMAP algorithm. Defaults to "correlation".
    """

    n_neighbors: Annotated[int, Param(int, default=15, identifier=True)]
    """The number of neighbors to use for the UMAP algorithm. Defaults to 15."""

    min_dist: Annotated[float, Param(float, default=0.1, identifier=True)]
    """The minimum distance between points in the UMAP algorithm. Defaults to 0.1."""

    metric: Annotated[str, Param(str, default='correlation', identifier=True)]
    """The distance metric to use for the UMAP algorithm. This can be one of "euclidean", "manhattan", "chebyshev", "minkowski", "canberra",
    "braycurtis", "mahalanobis", "wminkowski", "seuclidean", "cosine", "correlation", "haversine", "hamming", "jaccard", "dice", "russelrao",
    "kulsinski", "ll_dirichlet", "hellinger", "rogerstanimoto", "sokalmichener", "sokalsneath", or "yule" Defaults to "correlation".
    """

    def function(self, data: typing.Any) -> typing.Any:
        """Computes the UMAP embedding of the input data.

        Note:
            For information on the UMAP algorithm, see the `UMAP documentation <https://umap-learn.readthedocs.io/en/latest/index.html>`_.

        Args:
            data (typing.Any): The input data to be embedded. The data should be a NumPy array of shape `(number_of_samples, number_of_features)`.

        Returns:
            typing.Any: Returns the UMAP embedding of the input data as a NumPy array of shape `(number_of_samples, number_of_new_features)`.
        """

        umap = UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric
        )
        return umap.fit_transform(data)
