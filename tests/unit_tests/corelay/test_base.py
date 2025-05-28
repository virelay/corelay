"""A module that contains unit tests for the :py:mod:`corelay.base` module."""

import pytest

from corelay.base import Param


class TestParam:
    """Contains unit tests for the :py:class:`~corelay.base.Param` class."""

    @staticmethod
    def test_instantiation() -> None:
        """Tests that the :py:class:`~corelay.base.Param` class can be instantiated with any data type."""

        Param(object)

    @staticmethod
    def test_dtype_not_assigned() -> None:
        """Tests that a :py:class:`TypeError` is raised when no data type is provided when instantiating a :py:class:`~corelay.base.Param` instance.
        """

        with pytest.raises(TypeError):
            Param()  # type: ignore[call-arg] # pylint: disable=no-value-for-parameter

    @staticmethod
    def test_dtype_no_type() -> None:
        """Tests that a :py:class:`TypeError` is raised when specifying a data type when instantiating a :py:class:`~corelay.base.Param` instance that
        is not a type.
        """

        with pytest.raises(TypeError):
            Param('monkey')  # type: ignore[arg-type]

    @staticmethod
    def test_dtype_multiple() -> None:
        """Tests that the :py:class:`~corelay.base.Param` class can be instantiated with multiple data types in a tuple."""

        param = Param((object, type))
        assert param.dtype == (object, type)

    @staticmethod
    def test_dtype_single_to_tuple() -> None:
        """Tests that, when a single data type was specified when instantiating an instance of :py:class:`~corelay.base.Param`, that the
        :py:attr:`~corelay.plugboard.Slot.dtype` property returns a tuple of types containing a single type.
        """

        param = Param(object)
        assert param.dtype == (object,)

    @staticmethod
    def test_string_representation_of_param() -> None:
        """Tests that the string representation of a :py:class:`~corelay.base.Param` instance is correct. Sphinx AutoDoc uses :py:func:`repr` when it
        encounters :py:class:`typing.Annotated`, which in turn uses :py:func:`repr` to get a string representation of its metadata. This is a
        reasonable thing to do, but then Intersphinx tries to resolve the resulting string as types for cross-referencing, which is not possible with
        the default implementation of :py:meth:`object.__repr__`. To be able to get proper documentation, the fully-qualified name of the class needs
        to be returned, because this enable Sphinx AutoDoc to reference the class in the documentation. The tilde in front is interpreted by AutoDoc
        to mean that only the last part of the fully-qualified name should be displayed in the documentation.
        """

        param = Param(object)
        assert repr(param) == '~corelay.base.Param'
