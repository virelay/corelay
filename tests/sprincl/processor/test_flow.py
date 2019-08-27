import pytest

from sprincl.processor.flow import Shaper


class TestShaper:
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
    pass


class TestSequential:
    pass
