"""Test module for corelay/processor/base.py"""
import pytest

from corelay.io import NoStorage
from corelay.processor.base import Processor, Param, FunctionProcessor, ensure_processor


@pytest.fixture(scope='module')
def processor_type():
    """Fixture Processor Subclass"""
    class MyProcessor(Processor):
        """Custom Processor"""
        param_1 = Param(str, mandatory=True)
        param_2 = Param(int, -25, positional=True)
        param_3 = Param((str, int), 'default')
        value = 42
        text = 'apple'

        def function(self, data):
            return 21

    return MyProcessor


@pytest.fixture(scope='module')
# pylint: disable=unused-argument
def kwargs(processor_type):
    """Fixture dict for valid Processor Param values"""
    kwargs = {
        'is_output': False,
        'is_checkpoint': False,
        'param_1': 'stuff',
        'param_2': 5,
        'param_3': 6,
        'io': NoStorage()
    }
    return kwargs


@pytest.fixture(scope='module')
def function():
    """Fixture test function"""
    # pylint: disable=unused-argument
    def some_function(self, data):
        return 42
    return some_function


@pytest.fixture(scope='module')
def unbound_function():
    """Fixture unbound function"""
    # pylint: disable=unused-argument
    def some_function(data):
        return 42
    return some_function


class TestProcessor:
    """Test class for Processor"""
    @staticmethod
    def test_params_tracked(processor_type):
        """Processors should have Params tracked correctly"""
        assert (
            ['is_output', 'is_checkpoint', 'io', 'param_1', 'param_2', 'param_3'] == list(processor_type.collect(Param))
        )

    @staticmethod
    def test_creation(processor_type):
        """Processors should instatiatiate properly in all cases"""
        processor_type()

    @staticmethod
    def test_instance_assign(processor_type, kwargs):
        """Parameter values passed as keyword arguments during instatiation should be properly set"""
        processor = processor_type(**kwargs)
        assert all(getattr(processor, key) == val for key, val in kwargs.items())

    @staticmethod
    def test_instance_default(processor_type):
        """Default values should be properly assigned"""
        processor = processor_type(param_1="bacon")
        assert processor.param_2 == -25

    @staticmethod
    def test_instance_positional(processor_type):
        """Positional values should be properly assigned"""
        processor = processor_type(44)
        assert processor.param_2 == 44

    @staticmethod
    def test_unknown_param(processor_type):
        """Unknown parameters should raise an Exception"""
        with pytest.raises(TypeError):
            processor_type(param_1="bacon", parma_0='monkey')

    @staticmethod
    def test_abstract_func():
        """Processor should be abstract and thus fail to instantiate"""
        with pytest.raises(TypeError):
            Processor()

    @staticmethod
    def test_checkpoint(processor_type):
        """Checkpoints should store data"""
        processor = processor_type(param_1="bacon", is_checkpoint=True)
        out = processor(0)
        assert processor.checkpoint_data == out

    @staticmethod
    def test_param_values(processor_type, kwargs):
        """Params should be correctly set in __init__"""
        processor = processor_type(**kwargs)
        assert processor.param_values() == kwargs

    @staticmethod
    def test_copy_param_values(processor_type, kwargs):
        """Copies should have identical Param values."""
        processor = processor_type(**kwargs)
        copy = processor.copy()
        assert processor.param_values() == copy.param_values()

    @staticmethod
    def test_multiple_dtype(processor_type):
        """A Parameter can be of multiple types."""
        processor_type(param_1='soup', param_3=21)
        processor_type(param_1='soup', param_3='spoon')

    @staticmethod
    def test_mandatory_param(processor_type):
        """Mandatory parameters should raise an Exception when accessed without being set"""
        processor = processor_type()
        with pytest.raises(TypeError):
            # pylint: disable=pointless-statement
            processor.param_1

    @staticmethod
    def test_wrong_type_param(processor_type):
        """Passing a value with wrong type should raise a TypeError"""
        with pytest.raises(TypeError):
            processor_type(param_1=2)

    @staticmethod
    def test_bad_dtype():
        """Dtype should strictly only accept types."""
        with pytest.raises(TypeError):
            # pylint: disable=unused-variable
            class TestProcessor1(Processor):
                """Test class with wrong Param"""
                param = Param(2)

    @staticmethod
    def test_update_defaults(processor_type):
        """Parameter instance default values can be updated"""
        proc = processor_type(param_1='soup')
        proc.update_defaults(param_2=1)
        assert proc.param_2 == 1

    @staticmethod
    def test_reset_defaults(processor_type):
        """Parameter instance default values can be reset"""
        proc = processor_type(param_1='soup')
        proc.update_defaults(param_2=1)
        proc.reset_defaults()
        assert proc.param_2 == -25

    @staticmethod
    def test_reset_defaults_assigned(processor_type):
        """Resetting Param default values should go back to returning instatiation-time default values"""
        proc = processor_type(param_1='soup', param_2=2)
        proc.update_defaults(param_2=1)
        proc.reset_defaults()
        assert proc.param_2 == 2

    @staticmethod
    def test_update_defaults_wrong_dtype(processor_type):
        """Updating Param default values with the wrong type should raise a TypeError"""
        proc = processor_type(param_1='soup')
        with pytest.raises(TypeError):
            proc.update_defaults(param_2='bogus')


class TestFunctionProcessor:
    """Test class for FunctionProcessor"""
    @staticmethod
    def test_instantiation(unbound_function):
        """Instatiation should succeed with an unbound function as a keyword argument"""
        FunctionProcessor(function=unbound_function)

    @staticmethod
    def test_instance_call(unbound_function):
        """Calling an instance should be the same result as calling the function"""
        processor = FunctionProcessor(function=unbound_function)
        assert processor(0) == unbound_function(0)

    @staticmethod
    def test_instance_call_positional(unbound_function):
        """Calling an instance should be the same result as calling the function when given function positionally"""
        processor = FunctionProcessor(unbound_function)
        assert processor(0) == unbound_function(0)

    @staticmethod
    def test_instance_call_bound(function):
        """Calling a bound method should behave correctly"""
        processor = FunctionProcessor(function=function, bind_method=True)
        assert processor(0) == function(processor, 0)

    @staticmethod
    def test_non_callable():
        """Passing a non-callable as a function should raise a TypeError"""
        with pytest.raises(TypeError):
            FunctionProcessor(function='monkey')


class TestEnsureProcessor:
    """Test class for ensure_processor"""
    @staticmethod
    def test_processor(processor_type):
        """Passing an existing Processor should not create a new one."""
        processor = processor_type(param_1='giraffe')
        ensured = ensure_processor(processor)
        assert processor is ensured

    @staticmethod
    def test_function(unbound_function):
        """Passing a function should create a FunctionProcessor"""
        ensured = ensure_processor(unbound_function)
        assert isinstance(ensured, FunctionProcessor)

    @staticmethod
    def test_invalid():
        """Passing anythin but a callable or Processor should raise a TypeError"""
        with pytest.raises(TypeError):
            ensure_processor('mummy')

    @staticmethod
    def test_default_param_omitted(processor_type):
        """Passing a Param value should set its default"""
        processor = processor_type(param_1='giraffe')
        ensured = ensure_processor(processor, is_output=True)
        assert ensured.is_output

    @staticmethod
    def test_default_param_assigned(processor_type):
        """Passing a Param value should have lower priority than the explicitly set Param value"""
        processor = processor_type(param_1='giraffe', is_output=False)
        ensured = ensure_processor(processor, is_output=True)
        assert not ensured.is_output
