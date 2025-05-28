"""A module that contains unit tests for the :py:mod:`corelay.processor.distance` module."""

import typing

import numpy
import pytest

from corelay.processor.distance import SciPyPDist


def test_scipy_distance_processor() -> None:
    """Tests the :py:class:`corelay.processor.distance.ScipyDistanceProcessor` processor."""

    processor = SciPyPDist(metric='euclidean')
    data_points = numpy.array([[1, 1], [1, 2], [1, 3]])
    distance_matrix: numpy.ndarray[typing.Any, typing.Any] = processor(data_points)
    numpy.testing.assert_array_equal(distance_matrix, numpy.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]))


def test_scipy_distance_processor_invalid_metric() -> None:
    """Tests the :py:class:`corelay.processor.distance.ScipyDistanceProcessor` processor with an invalid metric."""

    processor = SciPyPDist(metric='invalid_metric')

    data_points = numpy.array([[1, 1], [1, 2], [1, 3]])
    with pytest.raises(ValueError):
        processor(data_points)
