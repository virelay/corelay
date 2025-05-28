"""A module that contains unit tests for the :py:mod:`corelay.io.base` module."""

import collections
import typing
from types import FunctionType
from typing import Annotated

import pytest

from corelay.base import Param
from corelay.pipeline.base import Pipeline, Task
from corelay.processor.base import Processor, FunctionProcessor


@pytest.fixture(name='processor_type', scope='module')
def get_processor_type_fixture() -> type[Processor]:
    """A fixture that produces a custom :py:class:`~corelay.processor.base.Processor` type.

    Returns:
        type[Processor]: Returns a custom :py:class:`~corelay.processor.base.Processor` type.
    """

    class MyProcessor(Processor):
        """A custom :py:class:`~corelay.processor.base.Processor` type."""

        param_1: Annotated[str, Param(str, 'default_value')]
        param_2: Annotated[int, Param(int, 42)]

        def function(self, data: typing.Any) -> typing.Any:
            """Multiplies the input data by 2.

            Args:
                data (typing.Any): The input data that is to be processed.

            Returns:
                typing.Any: Returns the processed data.
            """

            return data * 2

    return MyProcessor


@pytest.fixture(name='pipeline_type', scope='module')
def get_pipeline_type_fixture(processor_type: type[Processor]) -> type[Pipeline]:
    """A fixture that produces a custom :py:class:`~corelay.pipeline.base.Pipeline` type.

    Args:
        processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the custom
            :py:class:`~corelay.pipeline.base.Pipeline` type.

    Returns:
        type[Pipeline]: Returns a custom :py:class:`~corelay.pipeline.base.Pipeline` type.
    """

    class MyPipeline(Pipeline):
        """A custom :py:class:`~corelay.pipeline.base.Pipeline` type."""

        task_1: Annotated[FunctionProcessor, Task(FunctionProcessor, lambda self, x: x + 3, is_output=False, bind_method=True)]
        task_2: Annotated[FunctionProcessor, Task(processor_type, processor_type(), is_output=True)]

    return MyPipeline


@pytest.fixture(name='pipeline_with_multiple_outputs_type', scope='module')
def get_pipeline_with_multiple_outputs_type_fixture() -> type[Pipeline]:
    """A fixture that produces a custom :py:class:`~corelay.pipeline.base.Pipeline` type with multiple outputs.

    Returns:
        type[Pipeline]: Returns a custom :py:class:`~corelay.pipeline.base.Pipeline` type with multiple outputs.
    """

    class MyPipeline(Pipeline):
        """A custom :py:class:`~corelay.pipeline.base.Pipeline` type with multiple outputs."""

        task_1: Annotated[FunctionProcessor, Task(FunctionProcessor, lambda self, x: x + 2, is_output=True, bind_method=True)]
        task_2: Annotated[FunctionProcessor, Task(FunctionProcessor, lambda self, x: x * 2, is_output=True, bind_method=True)]

    return MyPipeline


class TestTask:
    """Contains unit tests for the :py:class:`~corelay.pipeline.base.Task` class."""

    @staticmethod
    def test_init() -> None:
        """Tests that the instantiation of a :py:class:`~corelay.pipeline.base.Task` without any arguments succeeds."""

        Task()

    @staticmethod
    def test_init_consistent_args(processor_type: type[Processor]) -> None:
        """Tests that the instantiation of a :py:class:`~corelay.pipeline.base.Task` with correct arguments succeeds.

        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the
                :py:class:`~corelay.pipeline.base.Task`.
        """

        Task(proc_type=processor_type, default=processor_type(), is_output=True)

    @staticmethod
    def test_init_with_invalid_proc_type() -> None:
        """Tests that the instantiation of a :py:class:`~corelay.pipeline.base.Task` with a ``proc_type`` that is not a sub-class of
        :py:class:`~corelay.processor.base.Processor` raises a :py:class:`TypeError`.
        """

        with pytest.raises(TypeError):
            Task(proc_type=FunctionType, default=lambda x: x)  # type: ignore[arg-type]

    @staticmethod
    def test_init_without_proc_type() -> None:
        """Tests that the instantiation of a :py:class:`~corelay.pipeline.base.Task` with a default value that is not of type
        :py:class:`~corelay.processor.base.Processor` fails.
        """

        with pytest.raises(TypeError):
            Task(default='bla')  # type: ignore[arg-type]

    @staticmethod
    def test_init_inconsistent_args(processor_type: type[Processor]) -> None:
        """Tests that the instantiation of a :py:class:`~corelay.pipeline.base.Task` with a default value that is not of type ``proc_type`` raises a
        :py:class:`TypeError`.

        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the
                :py:class:`~corelay.pipeline.base.Task`.
        """

        with pytest.raises(TypeError):
            Task(proc_type=processor_type, default=lambda x: x)

    def test_init_without_default(self) -> None:
        """Tests that the instantiation of a :py:class:`~corelay.pipeline.base.Task` without a default value succeeds."""

        Task(proc_type=Processor, default=None)

    @staticmethod
    def test_default_function_identity() -> None:
        """Tests that the default function of a :py:class:`~corelay.processor.base.FunctionProcessor` is the identity function."""

        # For some reason, PyLint does not recognize that the type of the default value is Processor and/or that Processor is callable
        task = Task()
        assert task.default is not None
        assert task.default(42) == 42  # pylint: disable=not-callable

    @staticmethod
    def test_assigned_default(processor_type: type[Processor]) -> None:
        """Tests that assigning a default :py:class:`~corelay.processor.base.Processor` value succeeds.

        Args:
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the
                :py:class:`~corelay.pipeline.base.Task`.
        """

        # For some reason, PyLint does not recognize that the type of the default value is Processor and/or that Processor is callable
        task = Task(proc_type=processor_type, default=processor_type())
        assert task.default is not None
        assert task.default(5) == 10  # pylint: disable=not-callable

    @staticmethod
    def test_class_call() -> None:
        """Tests that calling a :py:class:`~corelay.pipeline.base.Task` yields the :py:class:`~corelay.pipeline.base.TaskPlug` associated with the
        :py:class:`~corelay.pipeline.base.Task`.
        """

        task = Task(proc_type=Processor, default=FunctionProcessor())
        assert task is task().slot

    @staticmethod
    def test_class_call_obj() -> None:
        """Tests that calling a :py:class:`~corelay.pipeline.base.Task` with an ``obj`` argument yields a :py:class:`~corelay.pipeline.base.TaskPlug`
        with the :py:attr:`~corelay.plugboard.Plug.obj` property set to that value.
        """

        default_function_processor = FunctionProcessor()
        task = Task(proc_type=FunctionProcessor, default=default_function_processor)

        first_function_processor = FunctionProcessor(processing_function=lambda x: x + 10)
        task_plug = task(obj=first_function_processor)
        assert task_plug.obj == first_function_processor

        second_function_processor = FunctionProcessor(processing_function=lambda x: x + 20)
        task_plug.obj = second_function_processor
        assert task_plug.obj == second_function_processor

        del task_plug.obj
        assert task_plug.obj == default_function_processor

        lambda_expression = lambda x: x + 30
        task_plug.obj = lambda_expression
        assert isinstance(task_plug.obj, FunctionProcessor)
        assert task_plug.obj.processing_function == lambda_expression

        task_plug.obj = None
        assert task_plug.obj == default_function_processor

        class CustomProcessor(Processor):
            """A custom :py:class:`~corelay.processor.base.Processor` type."""

            def function(self, data: typing.Any) -> typing.Any:
                """Processes the input data by returning it unchanged.

                Args:
                    data (typing.Any): The input data that is to be processed.

                Returns:
                    typing.Any: Returns the processed data.
                """

                return data

        with pytest.raises(TypeError):
            task_plug.obj = CustomProcessor()

    @staticmethod
    def test_class_call_default() -> None:
        """Tests that calling a :py:class:`~corelay.pipeline.base.Task` with ``default`` argument yields a :py:class:`~corelay.pipeline.base.TaskPlug`
        with the :py:attr:`~corelay.pipeline.base.TaskPlug.default` property set.
        """

        default_function_processor = FunctionProcessor()
        task = Task(proc_type=FunctionProcessor, default=default_function_processor)

        first_function_processor = FunctionProcessor(processing_function=lambda x: x + 10)
        task_plug = task(default=first_function_processor)
        assert task_plug.default == first_function_processor

        second_function_processor = FunctionProcessor(processing_function=lambda x: x + 20)
        task_plug.default = second_function_processor
        assert task_plug.default == second_function_processor

        del task_plug.default
        assert task_plug.default == default_function_processor

        lambda_expression = lambda x: x + 30
        task_plug.default = lambda_expression
        assert isinstance(task_plug.default, FunctionProcessor)
        assert task_plug.default.processing_function == lambda_expression  # pylint: disable=no-member

        task_plug.default = None
        assert task_plug.default == default_function_processor

        class CustomProcessor(Processor):
            """A custom :py:class:`~corelay.processor.base.Processor` type."""

            def function(self, data: typing.Any) -> typing.Any:
                """Processes the input data by returning it unchanged.

                Args:
                    data (typing.Any): The input data that is to be processed.

                Returns:
                    typing.Any: Returns the processed data.
                """

                return data

        with pytest.raises(TypeError):
            task_plug.default = CustomProcessor()

    def test_updating_default(self) -> None:
        """Tests that updating the default value of a :py:class:`~corelay.pipeline.base.Task` works as expected."""

        default_function_processor = FunctionProcessor()
        task = Task(proc_type=FunctionProcessor, default=default_function_processor)
        assert task.default == default_function_processor

        updated_function_processor = FunctionProcessor(processing_function=lambda x: x + 10)
        task.default = updated_function_processor
        assert task.default == updated_function_processor

        del task.default
        assert task.default is None

        lambda_expression = lambda x: x + 30
        task.default = lambda_expression
        assert isinstance(task.default, FunctionProcessor)
        assert task.default.processing_function == lambda_expression  # pylint: disable=no-member

        task.default = None
        assert task.default is None

        with pytest.raises(TypeError):
            task.default = 'not a processor'

        class CustomProcessor(Processor):
            """A custom :py:class:`~corelay.processor.base.Processor` type."""

            def function(self, data: typing.Any) -> typing.Any:
                """Processes the input data by returning it unchanged.

                Args:
                    data (typing.Any): The input data that is to be processed.

                Returns:
                    typing.Any: Returns the processed data.
                """

                return data

        with pytest.raises(TypeError):
            task.default = CustomProcessor()

    @staticmethod
    def test_string_representation_of_task() -> None:
        """Tests that the string representation of a :py:class:`~corelay.pipeline.base.Task` instance is correct. Sphinx AutoDoc uses :py:func:`repr`
        when it encounters :py:class:`typing.Annotated`, which in turn uses :py:func:`repr` to get a string representation of its metadata. This is a
        reasonable thing to do, but then Intersphinx tries to resolve the resulting string as types for cross-referencing, which is not possible with
        the default implementation of :py:meth:`object.__repr__`. To be able to get proper documentation, the fully-qualified name of the class needs
        to be returned, because this enable Sphinx AutoDoc to reference the class in the documentation. The tilde in front is interpreted by AutoDoc
        to mean that only the last part of the fully-qualified name should be displayed in the documentation.
        """

        param = Task()
        assert repr(param) == '~corelay.pipeline.base.Task'


class TestPipeline:
    """Contains unit tests for the :py:class:`~corelay.pipeline.base.Pipeline` class."""

    @staticmethod
    def test_instantiation_base() -> None:
        """Tests that the instantiation of the base class :py:class:`~corelay.pipeline.base.Pipeline` without any arguments succeeds."""

        Pipeline()

    @staticmethod
    def test_instantiation_default(pipeline_type: type[Pipeline]) -> None:
        """Tests that the instantiation of a custom sub-class of :py:class:`~corelay.pipeline.base.Pipeline` without any arguments succeeds.

        Args:
            pipeline_type (type[Pipeline]): The custom :py:class:`~corelay.pipeline.base.Pipeline` type that is to be used in the test.
        """

        pipeline_type()

    @staticmethod
    def test_instantiation_arguments(pipeline_type: type[Pipeline], processor_type: type[Processor]) -> None:
        """Tests that the instantiation of a custom sub-class of :py:class:`~corelay.pipeline.base.Pipeline` with the correct arguments succeeds.

        Args:
            pipeline_type (type[Pipeline]): The custom :py:class:`~corelay.pipeline.base.Pipeline` type that is to be used in the test.
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
        """

        pipeline_type(task_1=lambda x: x + 2, task_2=processor_type())

    @staticmethod
    def test_default_call(pipeline_type: type[Pipeline]) -> None:
        """Tests that running a pipeline with all defaults in place succeeds.

        Args:
            pipeline_type (type[Pipeline]): The custom :py:class:`~corelay.pipeline.base.Pipeline` type that is to be used in the test.
        """

        pipeline = pipeline_type()
        pipeline(0)

    @staticmethod
    def test_default_call_no_input(pipeline_type: type[Pipeline]) -> None:
        """Tests that running a pipeline without an input raises a :py:class:`TypeError`.

        Args:
            pipeline_type (type[Pipeline]): The custom :py:class:`~corelay.pipeline.base.Pipeline` type that is to be used in the test.
        """

        pipeline = pipeline_type()
        with pytest.raises(TypeError):
            pipeline()  # type: ignore[call-arg]

    @staticmethod
    def test_default_call_output(pipeline_type: type[Pipeline]) -> None:
        """Tests that running a pipeline with the default processors returns the correct output.

        Args:
            pipeline_type (type[Pipeline]): The custom :py:class:`~corelay.pipeline.base.Pipeline` type that is to be used in the test.
        """

        pipeline = pipeline_type()
        output = pipeline(1)

        assert output == 8

    @staticmethod
    def test_default_call_output_multiple(pipeline_with_multiple_outputs_type: type[Pipeline]) -> None:
        """Tests that processors that have multiple outputs, correctly output a tuple containing the outputs.

        Args:
            pipeline_with_multiple_outputs_type (type[Pipeline]): The custom :py:class:`~corelay.pipeline.base.Pipeline` type that is to be used in
                the test.
        """

        pipeline = pipeline_with_multiple_outputs_type()
        output = pipeline(0)

        assert output == (2, 4)

    @staticmethod
    def test_default_param_values(pipeline_type: type[Pipeline], processor_type: type[Processor]) -> None:
        """Tests that the default parameter values are assigned correctly.

        Args:
            pipeline_type (type[Pipeline]): The custom :py:class:`~corelay.pipeline.base.Pipeline` type that is to be used in the test.
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
        """

        processor = processor_type(is_output=False)
        pipeline = pipeline_type(task_2=processor)

        assert not pipeline.task_2.is_output  # type: ignore[attr-defined]

    @staticmethod
    def test_checkpoint_processes(pipeline_type: type[Pipeline], processor_type: type[Processor]) -> None:
        """Tests that collecting all checkpoint processors succeeds.

        Args:
            pipeline_type (type[Pipeline]): The custom :py:class:`~corelay.pipeline.base.Pipeline` type that is to be used in the test.
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
        """

        first_processor = FunctionProcessor(processing_function=lambda x: x + 5, is_checkpoint=False)
        second_processor = processor_type(is_checkpoint=True)
        pipeline = pipeline_type(task_1=first_processor, task_2=second_processor)

        assert pipeline.checkpoint_processes() == collections.OrderedDict(task_2=second_processor)

    @staticmethod
    def test_checkpoint_processes_empty() -> None:
        """Tests that collecting all checkpoint processors returns an empty dictionary if the pipeline contains no tasks."""

        pipeline = Pipeline()
        assert pipeline.checkpoint_processes() == collections.OrderedDict()

    @staticmethod
    def test_checkpoint_processes_no_checkpoints() -> None:
        """Tests that collecting all checkpoint processors fails if no checkpoint processors are present."""

        class MyPipeline(Pipeline):
            """A custom :py:class:`~corelay.pipeline.base.Pipeline` type."""

            task_1: Annotated[Processor, Task()]
            task_2: Annotated[Processor, Task()]

        with pytest.raises(RuntimeError):
            pipeline = MyPipeline()
            pipeline.checkpoint_processes()

    @staticmethod
    def test_checkpoint_data(pipeline_type: type[Pipeline], processor_type: type[Processor]) -> None:
        """Tests that the checkpoint data is stored correctly.

        Args:
            pipeline_type (type[Pipeline]): The custom :py:class:`~corelay.pipeline.base.Pipeline` type that is to be used in the test.
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
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
            pipeline_type (type[Pipeline]): The custom :py:class:`~corelay.pipeline.base.Pipeline` type that is to be used in the test.
            processor_type (type[Processor]): The custom :py:class:`~corelay.processor.base.Processor` type that is to be used in the test.
        """

        first_processor = FunctionProcessor(processing_function=lambda x: x + 5, is_checkpoint=True)
        second_processor = processor_type(is_checkpoint=False)
        pipeline = pipeline_type(task_1=first_processor, task_2=second_processor)

        with pytest.raises(RuntimeError):
            pipeline.from_checkpoint()

        first_processor.checkpoint_data = 1
        output = pipeline.from_checkpoint()

        assert output == 2

    @staticmethod
    def test_pipeline_string_representation(pipeline_type: type[Pipeline]) -> None:
        """Tests that the string representation of a :py:class:`~corelay.pipeline.base.Pipeline` instance is correct.

        Args:
            pipeline_type (type[Pipeline]): The custom :py:class:`~corelay.pipeline.base.Pipeline` type that is to be used in the test.
        """

        pipeline = pipeline_type()
        expected_string_representation = (
            "MyPipeline(\n"
            "    FunctionProcessor(processing_function=lambda self, x: x + 3, bind_method=True) -> numpy.ndarray\n"
            "    MyProcessor(is_output=True, param_1='default_value', param_2=42) -> numpy.ndarray\n"
            ")"
        )

        assert repr(pipeline) == expected_string_representation
