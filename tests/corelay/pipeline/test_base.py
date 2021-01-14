"""Test module for corelay/pipeline/base.py"""
from collections import OrderedDict
from types import FunctionType

import pytest

from corelay.processor.base import Processor, Param, FunctionProcessor
from corelay.pipeline.base import Pipeline, Task


@pytest.fixture(scope='module')
def processor_type():
    """Fixture for custom Processor type"""
    class MyProcessor(Processor):
        """Custom Processor"""
        param_1 = Param(object)
        param_2 = Param(object)

        def function(self, data):
            """Double input"""
            return data * 2

    return MyProcessor


@pytest.fixture(scope='module')
def pipeline_type(processor_type):
    """Fixture for custom Pipeline type"""
    class MyPipeline(Pipeline):
        """Custom Pipeline"""
        task_1 = Task(FunctionProcessor, lambda self, x: x + 3, is_output=False, bind_method=True)
        task_2 = Task(processor_type, processor_type(), is_output=True)

    return MyPipeline


@pytest.fixture(scope='module')
# pylint: disable=unused-argument
def pipeline_type_multi(processor_type):
    """Fixture for custom Pipeline type with multiple outputs"""
    class MyPipeline(Pipeline):
        """Custom pipeline with multiple outputs"""
        task_1 = Task(FunctionProcessor, lambda self, x: x + 2, is_output=True, bind_method=True)
        task_2 = Task(FunctionProcessor, lambda self, x: x * 2, is_output=True, bind_method=True)

    return MyPipeline


class TestTask:
    """Test class for Task"""
    @staticmethod
    def test_instantiation_default():
        """Instatiation without any arguments should succeed"""
        Task()

    @staticmethod
    def test_instantiation_arguments(processor_type):
        """Instatiating with correct arguments should succeed"""
        Task(proc_type=processor_type, default=processor_type(), is_output=True)

    @staticmethod
    def test_proc_type_no_proc():
        """Instatiating with a proc_type which is not a subclass for Processor should raise a TypeError"""
        with pytest.raises(TypeError):
            Task(proc_type=FunctionType, default=(lambda x: x))

    @staticmethod
    def test_default_no_proc():
        """Instatiating with a default value not of type Processor should fail"""
        with pytest.raises(TypeError):
            Task(proc_type=Processor, default='bla')

    @staticmethod
    def test_proc_type_default_type_mismatch(processor_type):
        """Instatiating with a default value not of type proc_type should raise a TypeError"""
        with pytest.raises(TypeError):
            Task(proc_type=processor_type, default=(lambda x: x))

    @staticmethod
    def test_default_function_identity():
        """The default function of a FunctionProcessor should be the identity"""
        task = Task()
        # pylint: disable=not-callable
        assert task.default(42) == 42

    @staticmethod
    def test_assigned_default(processor_type):
        """Assigning a default Processor value should succeed"""
        task = Task(proc_type=processor_type, default=processor_type())
        # pylint: disable=not-callable
        assert task.default(5) == 10


class TestPipeline:
    """Test class for Pipeline"""
    @staticmethod
    def test_instatiation_base():
        """Instatiation of the base class without any arguments should succeed"""
        Pipeline()

    @staticmethod
    def test_instatiation_default(pipeline_type):
        """Instatiation of a custom subclass without any arguments should succeed"""
        pipeline_type()

    @staticmethod
    def test_instatiation_arguments(pipeline_type, processor_type):
        """Instatiation with correct arguments should succeed"""
        pipeline_type(task_1=lambda x: x + 2, task_2=processor_type())

    @staticmethod
    def test_default_call(pipeline_type):
        """Calling with all defaults in place should succeed"""
        pipeline = pipeline_type()
        pipeline(0)

    @staticmethod
    def test_default_call_no_input(pipeline_type):
        """Calling without an input should raise a TypeError"""
        pipeline = pipeline_type()
        with pytest.raises(TypeError):
            pipeline()

    @staticmethod
    def test_default_call_output(pipeline_type):
        """Default Processors should output correctly"""
        pipeline = pipeline_type()
        out = pipeline(1)
        assert out == 8

    @staticmethod
    def test_default_call_output_multiple(pipeline_type_multi):
        """Default Processors should output correctly with multiple outputs"""
        pipeline = pipeline_type_multi()
        out = pipeline(0)
        assert out == (2, 4)

    @staticmethod
    def test_default_param_values(pipeline_type, processor_type):
        """Default Parameter values should be assigned correctly"""
        proc = processor_type(is_output=False)
        pipeline = pipeline_type(task_2=proc)
        assert not pipeline.task_2.is_output

    @staticmethod
    def test_checkpoint_processes(pipeline_type, processor_type):
        """Collecting all processors relevant to a checkpoint should succeed"""
        proc_1 = FunctionProcessor(function=lambda x: x + 5, is_checkpoint=False)
        proc_2 = processor_type(is_checkpoint=True)
        pipeline = pipeline_type(task_1=proc_1, task_2=proc_2)
        assert pipeline.checkpoint_processes() == OrderedDict(task_2=proc_2)

    @staticmethod
    def test_checkpoint_data(pipeline_type, processor_type):
        """Checkpoint data should be stored correctly"""
        proc_1 = FunctionProcessor(function=lambda x: x + 5, is_checkpoint=True)
        proc_2 = processor_type(is_checkpoint=False)
        pipeline = pipeline_type(task_1=proc_1, task_2=proc_2)
        pipeline(0)
        assert proc_1.checkpoint_data == 5

    @staticmethod
    def test_from_checkpoint(pipeline_type, processor_type):
        """Resuming from a checkpoint should succeed"""
        proc_1 = FunctionProcessor(function=lambda x: x + 5, is_checkpoint=True)
        proc_2 = processor_type(is_checkpoint=False)
        pipeline = pipeline_type(task_1=proc_1, task_2=proc_2)
        proc_1.checkpoint_data = 1
        out = pipeline.from_checkpoint()
        assert out == 2
