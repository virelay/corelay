"""A module that contains unit tests for the ``corelay.io.base`` module."""

from collections import OrderedDict
from types import FunctionType
from typing import Annotated, Any

import pytest

from corelay.base import Param
from corelay.pipeline.base import Pipeline, Task
from corelay.processor.base import Processor, FunctionProcessor


@pytest.fixture(name='processor_type', scope='module')
def get_processor_type_fixture() -> type[Processor]:
    """A fixture that produces a custom ``Processor`` type.

    Returns:
        type[Processor]: Returns a custom ``Processor`` type.
    """

    class MyProcessor(Processor):
        """A custom ``Processor`` type."""

        param_1: Annotated[Any, Param(Any)]
        param_2: Annotated[Any, Param(Any)]

        def function(self, data: Any) -> Any:
            """Multiplies the input data by 2.

            Args:
                data (Any): The input data that is to be processed.

            Returns:
                Any: Returns the processed data.
            """

            return data * 2

    return MyProcessor


@pytest.fixture(name='pipeline_type', scope='module')
def get_pipeline_type_fixture(processor_type: type[Processor]) -> type[Pipeline]:
    """A fixture that produces a custom ``Pipeline`` type.

    Args:
        processor_type (type[Processor]): The custom ``Processor`` type that is to be used in the custom ``Pipeline`` type.

    Returns:
        type[Pipeline]: Returns a custom ``Pipeline`` type.
    """

    class MyPipeline(Pipeline):
        """A custom ``Pipeline`` type."""

        task_1 = Task(FunctionProcessor, lambda self, x: x + 3, is_output=False, bind_method=True)
        task_2 = Task(processor_type, processor_type(), is_output=True)

    return MyPipeline


@pytest.fixture(name='pipeline_with_multiple_outputs_type', scope='module')
def get_pipeline_with_multiple_outputs_type_fixture() -> type[Pipeline]:
    """A fixture that produces a custom ``Pipeline`` type with multiple outputs.

    Returns:
        type[Pipeline]: Returns a custom ``Pipeline`` type with multiple outputs.
    """

    class MyPipeline(Pipeline):
        """A custom ``Pipeline`` type with multiple outputs."""

        task_1 = Task(FunctionProcessor, lambda self, x: x + 2, is_output=True, bind_method=True)
        task_2 = Task(FunctionProcessor, lambda self, x: x * 2, is_output=True, bind_method=True)

    return MyPipeline


class TestTask:
    """Contains unit tests for the ``Task`` class."""

    @staticmethod
    def test_instantiation_default() -> None:
        """Tests that the instantiation of a ``Task`` without any arguments succeeds."""

        Task()

    @staticmethod
    def test_instantiation_arguments(processor_type: type[Processor]) -> None:
        """Tests that the instantiation of a ``Task`` with correct arguments succeeds.

        Args:
            processor_type (type[Processor]): The custom ``Processor`` type that is to be used in the ``Task``.
        """

        Task(proc_type=processor_type, default=processor_type(), is_output=True)

    @staticmethod
    def test_proc_type_no_proc() -> None:
        """Tests that the instantiation of a ``Task`` with a ``proc_type`` that is not a sub-class of ``Processor`` raises a ''TypeError''."""

        with pytest.raises(TypeError):
            Task(proc_type=FunctionType, default=lambda x: x)  # type: ignore[arg-type]

    @staticmethod
    def test_default_no_proc() -> None:
        """Tests that the instantiation of a ``Task`` with a default value that is not of type ``Processor`` fails."""

        with pytest.raises(TypeError):
            Task(default='bla')  # type: ignore[arg-type]

    @staticmethod
    def test_proc_type_default_type_mismatch(processor_type: type[Processor]) -> None:
        """Tests that the instantiation of a ``Task`` with a default value that is not of type ``proc_type`` raises a ``TypeError``.

        Args:
            processor_type (type[Processor]): The custom ``Processor`` type that is to be used in the ``Task``.
        """

        with pytest.raises(TypeError):
            Task(proc_type=processor_type, default=lambda x: x)

    @staticmethod
    def test_default_function_identity() -> None:
        """Tests that the default function of a ``FunctionProcessor`` is the identity function."""

        # For some reason, PyLint does not recognize that the type of the default value is Processor and/or that Processor is callable
        task = Task()
        assert task.default is not None
        assert task.default(42) == 42  # pylint: disable=not-callable

    @staticmethod
    def test_assigned_default(processor_type: type[Processor]) -> None:
        """Tests that assigning a default ``Processor`` value succeeds.

        Args:
            processor_type (type[Processor]): The custom ``Processor`` type that is to be used in the ``Task``.
        """

        # For some reason, PyLint does not recognize that the type of the default value is Processor and/or that Processor is callable
        task = Task(proc_type=processor_type, default=processor_type())
        assert task.default is not None
        assert task.default(5) == 10  # pylint: disable=not-callable


class TestPipeline:
    """Contains unit tests for the ``Pipeline`` class."""

    @staticmethod
    def test_instantiation_base() -> None:
        """Tests that the instantiation of the base class ``Pipeline`` without any arguments succeeds."""

        Pipeline()

    @staticmethod
    def test_instantiation_default(pipeline_type: type[Pipeline]) -> None:
        """Tests that the instantiation of a custom sub-class of ``Pipeline`` without any arguments succeeds.

        Args:
            pipeline_type (type[Pipeline]): The custom ``Pipeline`` type that is to be used in the test.
        """

        pipeline_type()

    @staticmethod
    def test_instantiation_arguments(pipeline_type: type[Pipeline], processor_type: type[Processor]) -> None:
        """Tests that the instantiation of a custom sub-class of ``Pipeline`` with the correct arguments succeeds.

        Args:
            pipeline_type (type[Pipeline]): The custom ``Pipeline`` type that is to be used in the test.
            processor_type (type[Processor]): The custom ``Processor`` type that is to be used in the test.
        """

        pipeline_type(task_1=lambda x: x + 2, task_2=processor_type())

    @staticmethod
    def test_default_call(pipeline_type: type[Pipeline]) -> None:
        """Tests that running a pipeline with all defaults in place succeeds.

        Args:
            pipeline_type (type[Pipeline]): The custom ``Pipeline`` type that is to be used in the test.
        """

        pipeline = pipeline_type()
        pipeline(0)

    @staticmethod
    def test_default_call_no_input(pipeline_type: type[Pipeline]) -> None:
        """Tests that running a pipeline without an input raises a ``TypeError``.

        Args:
            pipeline_type (type[Pipeline]): The custom ``Pipeline`` type that is to be used in the test.
        """

        pipeline = pipeline_type()
        with pytest.raises(TypeError):
            pipeline()  # type: ignore[call-arg]

    @staticmethod
    def test_default_call_output(pipeline_type: type[Pipeline]) -> None:
        """Tests that running a pipeline with the default processors returns the correct output.

        Args:
            pipeline_type (type[Pipeline]): The custom ``Pipeline`` type that is to be used in the test.
        """

        pipeline = pipeline_type()
        output = pipeline(1)

        assert output == 8

    @staticmethod
    def test_default_call_output_multiple(pipeline_with_multiple_outputs_type: type[Pipeline]) -> None:
        """Tests that processors that have multiple outputs, correctly output a tuple containing the outputs.

        Args:
            pipeline_with_multiple_outputs_type (type[Pipeline]): The custom ``Pipeline`` type that is to be used in the test.
        """

        pipeline = pipeline_with_multiple_outputs_type()
        output = pipeline(0)

        assert output == (2, 4)

    @staticmethod
    def test_default_param_values(pipeline_type: type[Pipeline], processor_type: type[Processor]) -> None:
        """Tests that the default parameter values are assigned correctly.

        Args:
            pipeline_type (type[Pipeline]): The custom ``Pipeline`` type that is to be used in the test.
            processor_type (type[Processor]): The custom ``Processor`` type that is to be used in the test.
        """

        processor = processor_type(is_output=False)
        pipeline = pipeline_type(task_2=processor)

        assert not pipeline.task_2.is_output  # type: ignore[attr-defined]

    @staticmethod
    def test_checkpoint_processes(pipeline_type: type[Pipeline], processor_type: type[Processor]) -> None:
        """Tests that collecting all processors relevant to a checkpoint succeeds.

        Args:
            pipeline_type (type[Pipeline]): The custom ``Pipeline`` type that is to be used in the test.
            processor_type (type[Processor]): The custom ``Processor`` type that is to be used in the test.
        """

        first_processor = FunctionProcessor(processing_function=lambda x: x + 5, is_checkpoint=False)
        second_processor = processor_type(is_checkpoint=True)
        pipeline = pipeline_type(task_1=first_processor, task_2=second_processor)

        assert pipeline.checkpoint_processes() == OrderedDict(task_2=second_processor)

    @staticmethod
    def test_checkpoint_data(pipeline_type: type[Pipeline], processor_type: type[Processor]) -> None:
        """Tests that the checkpoint data is stored correctly.

        Args:
            pipeline_type (type[Pipeline]): The custom ``Pipeline`` type that is to be used in the test.
            processor_type (type[Processor]): The custom ``Processor`` type that is to be used in the test.
        """

        first_processor = FunctionProcessor(processing_function=lambda x: x + 5, is_checkpoint=True)
        second_processor = processor_type(is_checkpoint=False)
        pipeline = pipeline_type(task_1=first_processor, task_2=second_processor)
        pipeline(0)

        assert first_processor.checkpoint_data == 5

    @staticmethod
    def test_from_checkpoint(pipeline_type: type[Pipeline], processor_type: type[Processor]) -> None:
        """Tests that resuming from a checkpoint succeeds.

        Args:
            pipeline_type (type[Pipeline]): The custom ``Pipeline`` type that is to be used in the test.
            processor_type (type[Processor]): The custom ``Processor`` type that is to be used in the test.
        """

        first_processor = FunctionProcessor(processing_function=lambda x: x + 5, is_checkpoint=True)
        second_processor = processor_type(is_checkpoint=False)
        pipeline = pipeline_type(task_1=first_processor, task_2=second_processor)
        first_processor.checkpoint_data = 1
        output = pipeline.from_checkpoint()

        assert output == 2
