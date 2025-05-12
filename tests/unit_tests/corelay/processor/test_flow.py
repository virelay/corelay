"""A module that contains unit tests for the :py:mod:`corelay.processor.flow` module."""

import pytest

from corelay.processor.base import FunctionProcessor
from corelay.processor.flow import Shaper, Parallel, Sequential


class TestShaper:
    """Contains unit tests for the :py:class:`~corelay.processor.flow.Shaper` class."""

    @staticmethod
    def test_extract() -> None:
        """Tests that extracting a single element succeeds."""

        shaper = Shaper(indices=(1,))
        assert shaper([1, 2, 3]) == (2,)

    @staticmethod
    def test_extract_multi() -> None:
        """Tests that extracting multiple elements succeeds."""

        shaper = Shaper(indices=(1, 2))
        assert shaper([1, 2, 3]) == (2, 3)

    @staticmethod
    def test_copy() -> None:
        """Tests that specifying the same index multiple times succeeds."""

        shaper = Shaper(indices=(1, 2, 1))
        assert shaper([1, 2, 3]) == (2, 3, 2)

    @staticmethod
    def test_stacked() -> None:
        """Tests that using another inner tuple for specifying indices succeeds."""

        shaper = Shaper(indices=(1, 2, (0, 2)))
        assert shaper([1, 2, 3]) == (2, 3, (1, 3))

    @staticmethod
    def test_stacked_multiple_levels() -> None:
        """Tests that specifying indices that nest tuples deeper than depth 2 succeeds."""

        shaper = Shaper(indices=(0, (1, (2,))))
        assert shaper([1, 2, 3]) == (1, (2, (3,)))


class TestParallel:
    """Contains unit tests for the :py:class:`~corelay.processor.flow.Parallel` class."""

    @staticmethod
    def test_non_iterable() -> None:
        """Tests that non-iterables are simply copied as many times as there are children."""

        parallel = Parallel(children=[FunctionProcessor(processing_function=lambda x, n=n: x + n) for n in range(5)])
        assert parallel(1) == (1, 2, 3, 4, 5)

    @staticmethod
    def test_iterable() -> None:
        """Tests that iterables are only valid if their length is the same as the number of children, when the children were given as a keyword
        argument.
        """

        parallel = Parallel(children=[FunctionProcessor(processing_function=lambda x, n=n: x + n) for n in range(5)])
        assert parallel((4, 3, 2, 1, 0)) == (4, 4, 4, 4, 4)

    @staticmethod
    def test_iterable_positional() -> None:
        """Tests that iterables are only valid if their length is the same as the number of children with children, when the children were given as
        positional arguments.
        """

        parallel = Parallel([FunctionProcessor(processing_function=lambda x, n=n: x + n) for n in range(5)])
        assert parallel((4, 3, 2, 1, 0)) == (4, 4, 4, 4, 4)

    @staticmethod
    def test_iterable_length_mismatch() -> None:
        """Tests that iterables are invalid if their length is different from the number of children."""

        parallel = Parallel(children=[FunctionProcessor(processing_function=lambda x, n=n: x + n) for n in range(5)])
        with pytest.raises(TypeError):
            parallel((4, 3, 2, 1))


class TestSequential:
    """Contains unit tests for the:py:class:`~corelay.processor.flow.Sequential` class."""

    @staticmethod
    def test_sequential() -> None:
        """Tests that the input to the :py:class:`~corelay.processor.flow.Sequential` is passed sequentially through the processors as argument in the
        same order that the child processors were specified, when the child processors were specified as a keyword argument.
        """

        sequential = Sequential(children=[FunctionProcessor(processing_function=lambda x, c=c: c + x) for c in 'bcde'])
        assert sequential('a') == 'edcba'

    @staticmethod
    def test_sequential_positional() -> None:
        """Tests that the input to the :py:class:`~corelay.processor.flow.Sequential` is passed sequentially through the processors as argument in the
        same order that the child processors were specified, when the child processors were specified as a positional argument.
        """

        sequential = Sequential([FunctionProcessor(processing_function=lambda x, c=c: c + x) for c in 'bcde'])
        assert sequential('a') == 'edcba'
