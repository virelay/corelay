"""Test module for corelay/base.py"""
import pytest

from corelay.base import Param


class TestParam:
    """Test class for Param"""
    @staticmethod
    def test_instatiation():
        """Param should instatiate correctly when passing any dtype."""
        Param(object)

    @staticmethod
    def test_dtype_not_assigned():
        """A TypeError should be raised when not providing any dtype."""
        with pytest.raises(TypeError):
            # pylint: disable=no-value-for-parameter
            Param()

    @staticmethod
    def test_dtype_no_type():
        """A TypeError should be raised when dtype is not a type."""
        with pytest.raises(TypeError):
            Param('monkey')

    @staticmethod
    def test_dtype_multiple():
        """Param should support multiple dtypes in a tuple."""
        param = Param((object, type))
        assert param.dtype == (object, type)

    @staticmethod
    def test_dtype_single_to_tuple():
        """A single dtype should result in a class parameter of a tuple with a single type."""
        param = Param(object)
        assert param.dtype == (object,)
