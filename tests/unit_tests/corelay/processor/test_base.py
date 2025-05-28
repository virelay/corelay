"""A module that contains unit tests for the :py:mod:`corelay.processor.base` module."""

import typing
from collections.abc import Callable
from typing import Annotated

import pytest

from corelay.io import NoStorage
from corelay.base import Param
from corelay.processor.base import Processor, FunctionProcessor, ensure_processor


@pytest.fixture(name='processor_type', scope='module')
def get_processor_type_fixture() -> type[Processor]:
    """A fixture that produces a custom :py:class:`~corelay.processor.base.Processor` type.

    Returns:
        type[Processor]: Returns a custom :py:class:`~corelay.processor.base.Processor` type.
    """

    class MyProcessor(Processor):
        """A custom :py:class:`~corelay.processor.base.Processor` type."""

        param_1: Annotated[str, Param(str, mandatory=True)]
        param_2: Annotated[int, Param(int, -25, positional=True)]
        param_3: Annotated[str | int, Param((str, int), 'default')]
        value = 42
        text = 'apple'

        def function(self, data: typing.Any) -> typing.Any:  # pylint: disable=unused-argument
            """An example function that implements the logic of the custom :py:class:`~corelay.processor.base.Processor`.

            Args:
                data (typing.Any): The input data, which is not used.

            Returns:
                typing.Any: Returns a fixed value of 21.
            """

            return 21

    return MyProcessor


@pytest.fixture(name='kwargs', scope='module')
def get_kwargs_fixture() -> dict[str, typing.Any]:
    """A fixture that produces a :py:class:`dict` with valid :py:class:`~corelay.base.Param` values for a
    :py:class:`~corelay.processor.base.Processor`.

    Returns:
        dict[str, typing.Any]: Returns a :py:class:`dict` with valid :py:class:`~corelay.base.Param` values for a
        :py:class:`~corelay.processor.base.Processor`.
    """

    return {
        'is_output': False,
        'is_checkpoint': False,
        'param_1': 'stuff',
        'param_2': 5,
        'param_3': 6,
        'io': NoStorage()
    }


@pytest.fixture(name='function', scope='module')
def get_function_fixture() -> Callable[[Processor, int], int]:
    """A fixture that produces a test function.

    Returns:
        Callable[[Processor, int], int]: Returns a test function.
    """

    def some_function(self: Processor, data: int) -> int:  # pylint: disable=unused-argument
        """A test function that is supposed to be bound to a class and therefore has access to self.

        Args:
            self (Processor): The instance of the class that the function is bound to.
            data (int): The input data, which is not used.

        Returns:
            int: Returns a fixed value of 42.
        """

        return 42

    return some_function


@pytest.fixture(name='unbound_function', scope='module')
def get_unbound_function_fixture() -> Callable[[int], int]:
    """A fixture that produces an unbound function, i.e., a function that is not bound to a class and does not have access to self.

    Returns:
        Callable[[int], int]: Returns an unbound function.
    """

    def some_function(_data: int) -> int:
        """A test function that is not supposed to be bound to a class and therefore has no access to self.

        Args:
            _data (int): The input data, which is not used.

        Returns:
            int: Returns a fixed value of 42.
        """

        return 42

    return some_function


class TestProcessor:
    """Contains unit tests for the :py:class:`~corelay.processor.base.Processor` class."""

    @staticmethod
    def test_params_tracked(processor_type: type[Processor]) -> None:
        """Tests that processors track parameters correctly.

        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
        """

        assert (
            ['is_output', 'is_checkpoint', 'io', 'param_1', 'param_2', 'param_3'] == list(processor_type.collect(Param))
        )

    @staticmethod
    def test_creation(processor_type: type[Processor]) -> None:
        """Tests that processors instantiate properly in all cases.

        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
        """

        processor_type()

    @staticmethod
    def test_instance_assign(processor_type: type[Processor], kwargs: dict[str, typing.Any]) -> None:
        """Tests that the parameter values passed as keyword arguments during instantiation are properly set.

        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
            kwargs (dict[str, typing.Any]): A :py:class:`dict` with valid :py:class:`~corelay.base.Param` values for a
                :py:class:`~corelay.processor.base.Processor`.
        """

        processor = processor_type(**kwargs)
        assert all(getattr(processor, key) == value for key, value in kwargs.items())

    @staticmethod
    def test_instance_default(processor_type: type[Processor]) -> None:
        """Tests that the default values are be properly.

        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
        """

        processor = processor_type(param_1='bacon')

        assert hasattr(processor, 'param_2')
        assert processor.param_2 == -25

    @staticmethod
    def test_instance_positional(processor_type: type[Processor]) -> None:
        """Tests that the positional values are properly assigned.

        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
        """

        processor = processor_type(44)

        assert hasattr(processor, 'param_2')
        assert processor.param_2 == 44

    @staticmethod
    def test_unknown_param(processor_type: type[Processor]) -> None:
        """Tests that unknown parameters raise an exception.

        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
        """

        with pytest.raises(TypeError):
            processor_type(param_1='bacon', parma_0='monkey')

    @staticmethod
    def test_too_many_positional_params(processor_type: type[Processor]) -> None:
        """Tests that passing too many positional arguments raises an exception.

        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
        """

        with pytest.raises(TypeError):
            processor_type(44, 21, param1='test', param3=1)

    @staticmethod
    def test_positional_param_also_supplied_as_keyword_argument(processor_type: type[Processor]) -> None:
        """Tests that passing a positional parameter as a keyword argument raises an exception.

        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
        """

        with pytest.raises(TypeError):
            processor_type(44, param1='test', param_2=22, param3='bacon')

    @staticmethod
    def test_abstract_func() -> None:
        """Tests that the :py:class:`~corelay.processor.base.Processor` class is abstract and thus fails to instantiate."""

        with pytest.raises(TypeError):
            Processor()  # type: ignore[abstract] # pylint: disable=abstract-class-instantiated

    @staticmethod
    def test_checkpoint(processor_type: type[Processor]) -> None:
        """Tests that checkpoints properly store data.

        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
        """

        processor = processor_type(param_1='bacon', is_checkpoint=True)
        output = processor(0)

        assert processor.checkpoint_data == output

    @staticmethod
    def test_param_values(processor_type: type[Processor], kwargs: dict[str, typing.Any]) -> None:
        """Tests that the parameters are correctly set in the constructor.
        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
            kwargs (dict[str, typing.Any]): A :py:class:`dict` with valid :py:class:`~corelay.base.Param` values for a
                :py:class:`~corelay.processor.base.Processor`.
        """

        processor = processor_type(**kwargs)
        assert processor.param_values() == kwargs

    @staticmethod
    def test_copy_param_values(processor_type: type[Processor], kwargs: dict[str, typing.Any]) -> None:
        """Tests that copies of processors have identical Param values.

        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
            kwargs (dict[str, typing.Any]): A :py:class:`dict` with valid :py:class:`~corelay.base.Param` values for a
                :py:class:`~corelay.processor.base.Processor`.
        """

        processor = processor_type(**kwargs)
        processor_copy = processor.copy()

        assert processor.param_values() == processor_copy.param_values()

    @staticmethod
    def test_multiple_dtype(processor_type: type[Processor]) -> None:
        """Tests that a :py:class:`~corelay.base.Param` can be of multiple types.

        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
        """

        processor_type(param_1='soup', param_3=21)
        processor_type(param_1='soup', param_3='spoon')

    @staticmethod
    def test_mandatory_param(processor_type: type[Processor]) -> None:
        """Tests that mandatory parameters raise an exception when accessed without being set.

        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
        """

        processor = processor_type()

        with pytest.raises(TypeError):
            _ = processor.param_1  # type: ignore[attr-defined]

    @staticmethod
    def test_wrong_type_param(processor_type: type[Processor]) -> None:
        """Tests that passing a value with wrong the type raises a :py:class:`TypeError`.

        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
        """

        with pytest.raises(TypeError):
            processor_type(param_1=2)

    @staticmethod
    def test_bad_dtype() -> None:
        """Tests that the the :py:attr:`~corelay.plugboard.Slot.dtype` only accepts types, i.e., non-type values cannot be passed as argument for the
        ``dtype`` parameter.
        """

        with pytest.raises(TypeError):
            class TestProcessorWithWrongParam(Processor):
                """A custom :py:class:`~corelay.processor.base.Processor` with wrong a Param."""

                param: Annotated[int, Param(2)]
                """A wrong :py:class:`~corelay.base.Param` with a non-type value as dtype."""

                def function(self, data: typing.Any) -> typing.Any:
                    """An example function that implements the logic of the custom :py:class:`~corelay.processor.base.Processor`.

                    Args:
                        data (typing.Any): The input data.

                    Returns:
                        typing.Any: Returns the input data.
                    """

                    return data

            _ = TestProcessorWithWrongParam()

    @staticmethod
    def test_update_defaults(processor_type: type[Processor]) -> None:
        """Tests that the default values of a :py:class:`~corelay.base.Param` instance can be updated.

        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
        """

        processor = processor_type(param_1='soup')
        processor.update_defaults(param_2=1)

        assert hasattr(processor, 'param_2')
        assert processor.param_2 == 1
        assert processor.param_2 == 1

    @staticmethod
    def test_reset_defaults(processor_type: type[Processor]) -> None:
        """Tests that the default values of a :py:class:`~corelay.base.Param` instance can be reset.

        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
        """

        processor = processor_type(param_1='soup')
        processor.update_defaults(param_2=1)
        processor.reset_defaults()

        assert hasattr(processor, 'param_2')
        assert processor.param_2 == -25

    @staticmethod
    def test_reset_defaults_assigned(processor_type: type[Processor]) -> None:
        """Tests that resetting the default values of :py:class:`~corelay.base.Param` instances go back to returning instantiation-time default
        values.

        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
        """

        processor = processor_type(param_1='soup', param_2=2)
        processor.update_defaults(param_2=1)
        processor.reset_defaults()

        assert hasattr(processor, 'param_2')
        assert processor.param_2 == 2

    @staticmethod
    def test_update_defaults_wrong_dtype(processor_type: type[Processor]) -> None:
        """Tests that updating the default values of :py:class:`~corelay.base.Param` instances with the wrong type raises a :py:class:`TypeError`.

        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
        """

        processor = processor_type(param_1='soup')

        with pytest.raises(TypeError):
            processor.update_defaults(param_2='bogus')


class TestFunctionProcessor:
    """Contains unit tests for the :py:class:`~corelay.processor.base.FunctionProcessor` class."""

    @staticmethod
    def test_instantiation(unbound_function: Callable[[int], int]) -> None:
        """Tests that the instantiation of a :py:class:`~corelay.processor.base.FunctionProcessor` with an unbound function as a keyword argument
        should work.

        Args:
            unbound_function (Callable[[int], int]): The unbound function that is to be used in the test.
        """

        FunctionProcessor(processing_function=unbound_function)

    @staticmethod
    def test_instance_call(unbound_function: Callable[[int], int]) -> None:
        """Tests that calling a :py:class:`~corelay.processor.base.FunctionProcessor` instance, that was constructed with a function that was passed
        as a keyword argument, results in the same output as calling the function directly.

        Args:
            unbound_function (Callable[[int], int]): The unbound function that is to be used in the test.
        """

        processor = FunctionProcessor(processing_function=unbound_function)
        assert processor(0) == unbound_function(0)

    @staticmethod
    def test_instance_call_positional(unbound_function: Callable[[int], int]) -> None:
        """Tests that calling a :py:class:`~corelay.processor.base.FunctionProcessor` instance, that was constructed with a function that was passed
        as a positional argument, results in the same output as calling the function directly.

        Args:
            unbound_function (Callable[[int], int]): The unbound function that is to be used in the test.
        """

        processor = FunctionProcessor(unbound_function)
        assert processor(0) == unbound_function(0)

    @staticmethod
    def test_instance_call_bound(function: Callable[[Processor, int], int]) -> None:
        """Tests that calling a bound method should behave correctly.

        Args:
            function (Callable[[Processor, int], int]): The function that is to be used in the test.
        """

        processor = FunctionProcessor(processing_function=function, bind_method=True)
        assert processor(0) == function(processor, 0)

    @staticmethod
    def test_non_callable() -> None:
        """Tests that passing a non-callable object as an argument for the ``processing_function`` parameter of the
        :py:class:`~corelay.processor.base.FunctionProcessor` constructor raises a :py:class:`TypeError`.
        """

        with pytest.raises(TypeError):
            FunctionProcessor(processing_function='monkey')


class TestEnsureProcessor:
    """Contains unit tests for the :py:func:`~corelay.processor.base.ensure_processor` function."""

    @staticmethod
    def test_processor(processor_type: type[Processor]) -> None:
        """Tests that passing an existing :py:class:`~corelay.processor.base.Processor` to the :py:func:`~corelay.processor.base.ensure_processor`
        function returns the original :py:class:`~corelay.processor.base.Processor` instead of creating a new one.

        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
        """

        processor = processor_type(param_1='giraffe')
        ensured_processor = ensure_processor(processor)

        assert processor is ensured_processor

    @staticmethod
    def test_function(unbound_function: Callable[[int], int]) -> None:
        """Tests that passing a function to the :py:func:`~corelay.processor.base.ensure_processor` function returns a new
        :py:class:`~corelay.processor.base.FunctionProcessor` instance that wraps the function.

        Args:
            unbound_function (Callable[[int], int]): The unbound function that is to be used in the test.
        """

        ensured_processor = ensure_processor(unbound_function)
        assert isinstance(ensured_processor, FunctionProcessor)

    @staticmethod
    def test_invalid() -> None:
        """Tests that passing a non-callable object that is not of type :py:class:`~corelay.processor.base.Processor` to the
        :py:func:`~corelay.processor.base.ensure_processor` function raises a :py:class:`TypeError`.
        """

        with pytest.raises(TypeError):
            ensure_processor('mummy')  # type: ignore[arg-type]

    @staticmethod
    def test_default_param_omitted(processor_type: type[Processor]) -> None:
        """Tests that passing a :py:class:`~corelay.base.Param` value as a keyword argument to the :py:func:`~corelay.processor.base.ensure_processor`
        function sets the value of the :py:class:`~corelay.base.Param` in the returned :py:class:`~corelay.processor.base.Processor` instance.

        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
        """

        processor = processor_type(param_1='giraffe')
        ensured_processor = ensure_processor(processor, is_output=True)

        assert ensured_processor.is_output

    @staticmethod
    def test_default_param_assigned(processor_type: type[Processor]) -> None:
        """Tests that passing a :py:class:`~corelay.base.Param` value as a keyword argument the the
        :py:func:`~corelay.processor.base.ensure_processor` function has a lower priority than explicitly set :py:class:`~corelay.base.Param` values
        and should not overwrite them.

        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
        """

        processor = processor_type(param_1='giraffe', is_output=False)
        ensured_processor = ensure_processor(processor, is_output=True)

        assert not ensured_processor.is_output
