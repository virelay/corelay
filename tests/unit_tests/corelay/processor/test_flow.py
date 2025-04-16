"""Test module for corelay/processor/flow.py"""
import pytest

from corelay.processor.flow import Shaper, Parallel, Sequential
from corelay.processor.base import FunctionProcessor


class TestShaper:
    """Test class for Shaper"""
    @staticmethod
    def test_extract():
        """Extracting a single element should succeed"""
        shaper = Shaper(indices=(1,))
        assert shaper([1, 2, 3]) == (2,)

    @staticmethod
    def test_extract_multi():
        """Extracting multiple elements should succeed"""
        shaper = Shaper(indices=(1, 2))
        assert shaper([1, 2, 3]) == (2, 3)

    @staticmethod
    def test_copy():
        """Specifying the same index multiple times should succeed"""
        shaper = Shaper(indices=(1, 2, 1))
        assert shaper([1, 2, 3]) == (2, 3, 2)

    @staticmethod
    def test_stacked():
        """Using another inner tuple for specifying indices should succeed"""
        shaper = Shaper(indices=(1, 2, (0, 2)))
        assert shaper([1, 2, 3]) == (2, 3, (1, 3))

    @staticmethod
    def test_stacked_too_much():
        """Specifying tuples any deeper than depth 2 should fail"""
        with pytest.raises(TypeError):
            Shaper(indices=(0, (1, (2,))))


class TestParallel:
    """Test class for Parallel"""
    @staticmethod
    def test_non_iterable():
        """Non-iterables should simply be copied as many times as there are children"""
        parallel = Parallel(children=[FunctionProcessor(function=(lambda x, n=n: x + n)) for n in range(5)])
        assert parallel(1) == (1, 2, 3, 4, 5)

    @staticmethod
    def test_iterable():
        """Iterables are valid if they have the same length as there are children"""
        parallel = Parallel(children=[FunctionProcessor(function=(lambda x, n=n: x + n)) for n in range(5)])
        assert parallel((4, 3, 2, 1, 0)) == (4, 4, 4, 4, 4)

    @staticmethod
    def test_iterable_positional():
        """Iterables are valid if they have the same length as there are children with children given positionally"""
        parallel = Parallel([FunctionProcessor(function=(lambda x, n=n: x + n)) for n in range(5)])
        assert parallel((4, 3, 2, 1, 0)) == (4, 4, 4, 4, 4)

    @staticmethod
    def test_iterable_length_mismatch():
        """Iterables are invalid if they have a different length compared to the number of children"""
        parallel = Parallel(children=[FunctionProcessor(function=(lambda x, n=n: x + n)) for n in range(5)])
        with pytest.raises(TypeError):
            parallel((4, 3, 2, 1))


class TestSequential:
    """Test class for Sequential"""
    @staticmethod
    def test_sequential():
        """Input should be passed sequentially along all child processors"""
        sequential = Sequential(children=[FunctionProcessor(function=lambda x, c=c: c + x) for c in 'bcde'])
        assert sequential('a') == 'edcba'

    @staticmethod
    def test_sequential_positional():
        """Input should be passed sequentially along all child processors when given children positionally"""
        sequential = Sequential([FunctionProcessor(function=lambda x, c=c: c + x) for c in 'bcde'])
        assert sequential('a') == 'edcba'
