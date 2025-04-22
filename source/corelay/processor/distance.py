"""A module that contains pair-wise distance ``Processors``."""

from typing import Annotated, Any, Literal

from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform

from corelay.base import Param
from corelay.processor.base import Processor


class Distance(Processor):
    """The abstract base class for distance processors.

    Args:
        is_output (bool, optional): A value indicating whether this ``Distance`` processor is the output of a ``Pipeline``. Defaults to `False`.
        is_checkpoint (bool | None, optional): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to `False`.
        io (Storable | None, optional): An IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this
            run or in subsequent runs of the ``Pipeline``. Defaults to an instance of ``NoStorage``.
    """


PairWiseDistanceMeasure = Literal[
    'braycurtis',
    'canberra',
    'chebychev', 'chebyshev', 'cheby', 'cheb', 'ch',
    'cityblock', 'cblock', 'cb', 'c',
    'correlation', 'co',
    'cosine', 'cos',
    'dice',
    'euclidean', 'euclid', 'eu', 'e',
    'hamming', 'hamm', 'ha', 'h',
    'minkowski', 'mi', 'm',
    'pnorm',
    'jaccard', 'jacc', 'ja', 'j',
    'jensenshannon', 'js',
    'kulczynski1',
    'mahalanobis', 'mahal', 'mah',
    'rogerstanimoto',
    'russellrao',
    'seuclidean', 'se', 's',
    'sokalmichener',
    'sokalsneath',
    'sqeuclidean', 'sqe', 'sqeuclid',
    'yule'
]
"""An enumeration of the distance measures supported by ``scipy.spatial.distance.pdist``."""


class SciPyPDist(Distance):
    """A distance metric, that computes the pair-wise distance between observations in n-dimensional space using ``scipy.spatial.distance.pdist``.

    Args:
        is_output (bool, optional): A value indicating whether this ``SciPyPDist`` distance processor is the output of a ``Pipeline``. Defaults to
            `False`.
        is_checkpoint (bool | None, optional): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to `False`.
        io (Storable | None, optional): An IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this
            run or in subsequent runs of the ``Pipeline``. Defaults to an instance of ``NoStorage``.
        metric (str): The distance metric to use. Default is "euclidean".
        m_kwargs (dict): Additional keyword arguments to pass to the distance function.
    """

    metric: Annotated[PairWiseDistanceMeasure, Param(str, 'euclidean', identifier=True)]
    """The distance metric to use. Defaults to "euclidean"."""

    m_kwargs: Annotated[dict[str, Any], Param(dict, {})]
    """Additional keyword arguments to pass to the distance function."""

    def function(self, data: Any) -> Any:
        """Applies the pairwise distance function to the input data.

        Args:
            data (Any): The input data that is to be processed. The input data should be a NumPy array of shape
                `(number_of_samples, number_of_features)`.

        Returns:
            Any: Returns the pairwise distance matrix of shape `(number_of_samples, number_of_samples)`.
        """

        input_data: NDArray[Any] = data
        distance_matrix: NDArray[Any] = pdist(input_data, metric=self.metric, **self.m_kwargs)
        return squareform(distance_matrix)
