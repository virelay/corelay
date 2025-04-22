"""A module that contains unit tests for the ``corelay.base`` module."""

import pytest

from corelay.base import Param


class TestParam:
    """Contains unit tests for the ``Param`` class."""

    @staticmethod
    def test_instantiation() -> None:
        """Tests that the ``Param`` class can be instantiated with any data type."""

        Param(object)

    @staticmethod
    def test_dtype_not_assigned() -> None:
        """Tests that a ``TypeError`` is raised when no data type is provided when instantiating a ``Param`` instance."""

        with pytest.raises(TypeError):
            Param()  # type: ignore[call-arg] # pylint: disable=no-value-for-parameter

    @staticmethod
    def test_dtype_no_type() -> None:
        """Tests that a ``TypeError`` is raised when specifying a data type when instantiating a ``Param`` instance that is not a type."""

        with pytest.raises(TypeError):
            Param('monkey')  # type: ignore[arg-type]

    @staticmethod
    def test_dtype_multiple() -> None:
        """Tests that the ``Param`` class can be instantiated with multiple data types in a tuple."""

        param = Param((object, type))
        assert param.dtype == (object, type)

    @staticmethod
    def test_dtype_single_to_tuple() -> None:
        """Tests that, when a single data type was specified when instantiating an instance of ``Param``, that the ``dtype`` property returns a tuple
        of types containing a single type.
        """

        param = Param(object)
        assert param.dtype == (object,)
