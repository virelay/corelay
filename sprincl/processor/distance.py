"""Pair-wise distance processors

"""
import logging

from scipy.spatial.distance import pdist, squareform

from .base import Processor, Param

LOGGER = logging.getLogger(__name__)


class Distance(Processor):
    """Distance Processor

    """


class SciPyPDist(Distance):
    """Pairwise distances using scipy.spatial.distance.pdist

    Note
    ----
    See scipy.spatial.distance.pdist for valid values of metric
    """
    metric = Param(str, 'euclidean', identifier=True)
    m_args = Param(list, [])
    m_kwargs = Param(dict, {})

    def function(self, data):
        # pylint: disable=not-an-iterable,not-a-mapping
        return squareform(pdist(data, metric=self.metric, *self.m_args, **self.m_kwargs))
