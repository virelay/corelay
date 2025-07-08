"""A module that contains processors for pair-wise distance metrics."""

import typing
from typing import Annotated, Literal, TypeAlias, TypeGuard, get_args

import numpy
from scipy.spatial.distance import pdist, squareform

from corelay.base import Param
from corelay.processor.base import Processor


class Distance(Processor):
    """The abstract base class for distance processors.

    Args:
        is_output (bool): A value indicating whether this :py:class:`Distance` processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
    """


class SciPyPDist(Distance):
    """A distance metric, that computes the pair-wise distance between observations in n-dimensional space using
    :py:func:`scipy.spatial.distance.pdist`.

    Args:
        is_output (bool): A value indicating whether this :py:class:`SciPyPDist` distance processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        metric (str): The distance metric to use. Default is "euclidean".
        m_kwargs (dict): Additional keyword arguments to pass to the distance function.
    """

    metric: Annotated[str, Param(str, 'euclidean', identifier=True)]
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

    m_kwargs: Annotated[dict[str, typing.Any], Param(dict, {})]
    """Additional keyword arguments to pass to the distance function."""

    def function(self, data: typing.Any) -> typing.Any:
        """Applies the pairwise distance function to the input data.

        Args:
            data (typing.Any): The input data that is to be processed. The input data should be a NumPy array of shape
                `(number_of_samples, number_of_features)`.

        Raises:
            ValueError: The distance metric is not valid.

        Returns:
            typing.Any: Returns the pairwise distance matrix of shape `(number_of_samples, number_of_samples)`.
        """

        # This is necessary to ensure that MyPy does not complain that the metric is not valid; ideally, we would use literals ourselves, but
        # unfortunately, Sphinx AutoDoc cannot handle type aliases correctly unless we use Postponed Evaluation of Annotations (PEP 563), which in
        # turn breaks our usage of typing.Annotated for slots
        Metric: TypeAlias = Literal[
            'braycurtis',
            'canberra',
            'chebychev',
            'chebyshev',
            'cheby',
            'cheb',
            'ch',
            'cityblock',
            'cblock',
            'cb',
            'c',
            'correlation',
            'co',
            'cosine',
            'cos',
            'dice',
            'euclidean',
            'euclid',
            'eu',
            'e',
            'hamming',
            'hamm',
            'ha',
            'h',
            'minkowski',
            'mi',
            'm',
            'pnorm',
            'jaccard',
            'jacc',
            'ja',
            'j',
            'jensenshannon',
            'js',
            'kulczynski1',
            'mahalanobis',
            'mahal',
            'mah',
            'rogerstanimoto',
            'russellrao',
            'seuclidean',
            'se',
            's',
            'sokalmichener',
            'sokalsneath',
            'sqeuclidean',
            'sqe',
            'sqeuclid',
            'yule'
        ]
        metrics = list(get_args(Metric))

        def check_if_metric_is_valid(metric: str) -> TypeGuard[Metric]:
            return metric in metrics

        if not check_if_metric_is_valid(self.metric):
            raise ValueError(f'Invalid metric: {self.metric}.')

        input_data: numpy.ndarray[typing.Any, numpy.dtype[numpy.floating] | numpy.dtype[numpy.integer]] = data
        distance_matrix: numpy.ndarray[typing.Any, typing.Any] = pdist(input_data, metric=self.metric, **self.m_kwargs)
        return squareform(distance_matrix)
