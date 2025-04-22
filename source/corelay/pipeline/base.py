"""A module that contains the base classes for tasks and pipeline."""

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from corelay.plugboard import Slot, Plug
from corelay.processor.base import ensure_processor, Processor


class TaskPlug(Plug):
    """A task plug, which ensures that all contained objects are processors."""

    def __init__(
        self,
        slot: Slot,
        obj: Processor | Callable[..., Any] | None = None,
        default: Processor | Callable[..., Any] | None = None,
        **kwargs: Any
    ) -> None:
        """Initializes a new ``TaskPlug`` instance.

        Args:
            slot (Slot): Slot instance to associate with this ``TaskPlug``.
            obj (Processor | Callable[..., Any] | None, optional): A processor held in the ``TaskPlug`` container. If not set, ``default`` is returned
                as its value. Defaults to `None`.
            default (Processor | Callable[..., Any] | None, optional): A plug-dependent lower-priority processor held in the ``TaskPlug`` container.
                If not set, ``fallback`` is returned. Defaults to `None`.
            **kwargs (Any): Keyword arguments passed down to the base class constructor, for cooperativity's sake, which is the next class in the
                inheritance hierarchy.
        """

        if default is not None:
            default = ensure_processor(default, **kwargs)
        if obj is not None:
            obj = ensure_processor(obj)

        super().__init__(slot, obj=obj, default=default, **kwargs)

    @property
    def obj(self) -> Processor | None:
        """Gets or sets the processor contained in the ``TaskPlug``. If the ``TaskPlug`` does not contain a processor, ``default`` is retrieved
        instead.

        Note:
            Actually, only the setter of the ``obj`` property needs to be overridden. The proper way of overriding the setter of the ``obj`` property
            is to use the `@Plug.obj.setter` idiom, but this, unfortunately, is detected as a false positive by MyPy. Although, they have promised to
            support this idiom (https://github.com/python/mypy/issues/5936), they have not done so yet, probably, because it is not used very often in
            the wild. Their workaround solution is to also override the getter and then use the new getter to override the setter and the deleter. I
            have tested it, only in the getter, `super().obj` can be used, as it is not, yet, overridden, but in the setter and deleter, the complete
            function must be re-implemented. Also, when the getter is overridden, not only the setter, but also the deleter must be overridden, as
            otherwise, it will not be available anymore.

        Returns:
            Processor | None: Returns the processor contained in the ``TaskPlug``. If not set, ``default`` is returned.
        """

        processor: Processor | None = super().obj
        return processor

    @obj.setter
    def obj(self, value: Processor | Callable[..., Any] | None) -> None:
        """Gets or sets the processor contained in the ``TaskPlug`` and checks for consistency. It is ensured first, that the value is a
        ``Processor``. If not, it is converted to a ``Processor`` using the ``ensure_processor`` function.

        Args:
            value (Processor | Callable[..., Any] | None): The processor to set.

        Raises:
            TypeError: The processor is not consistent with the ``dtype``, i.e., it is neither `None`, nor of the type ``dtype`` or one of the
                types in the tuple ``dtype``.
        """

        if value is not None:
            value = ensure_processor(value)
        self._obj = value

        try:
            self._consistent()
        except TypeError as exception:
            raise TypeError('The processor is not consistent with the dtype.') from exception

    @obj.deleter
    def obj(self) -> None:
        """Deletes the processor contained in the ``TaskPlug`` by setting it to `None`."""

        self.obj = None

    @property
    def default(self) -> Any:
        """Gets or sets the default processor of the ``TaskPlug``. If the ``default`` processor is not set, then the ``fallback`` processor is
        retrieved instead.

        Note:
            Actually, only the setter of the ``default`` property needs to be overridden. The proper way of overriding the setter of the ``default``
            property is to use the `@Plug.default.setter` idiom, but this, unfortunately, is detected as a false positive by MyPy. Although, they have
            promised to support this idiom (https://github.com/python/mypy/issues/5936), they have not done so yet, probably, because it is not used
            very often in the wild. Their workaround solution is to also override the getter and then use the new getter to override the setter and
            the deleter. I have tested it, only in the getter, `super().default` can be used, as it is not, yet, overridden, but in the setter and
            deleter, the complete function must be re-implemented. Also, when the getter is overridden, not only the setter, but also the deleter must
            be overridden, as otherwise, it will not be available anymore.

        Returns:
            Any: Returns the default processor of the ``TaskPlug``. If not set, ``fallback`` is returned.
        """

        return super().default

    @default.setter
    def default(self, value: Any) -> None:
        """Gets or sets the default processor of the ``TaskPlug`` and checks for consistency. It is ensured first, that the new default value is a
        ``Processor``. If not, it is converted to a ``Processor`` using the ``ensure_processor`` function.

        Args:
            value (Any): The new default processor to set.

        Raises:
            TypeError: The default processor is not consistent with the ``dtype``, i.e., it is neither `None`, nor of the type ``dtype`` or one of the
                types in the tuple ``dtype``.
        """

        if value is not None:
            value = ensure_processor(value)
        self._default = value

        try:
            self._consistent()
        except TypeError as exception:
            raise TypeError('The default processor is not consistent with the dtype.') from exception

    @default.deleter
    def default(self) -> None:
        """Deletes the default processor of the ``TaskPlug`` by setting it to `None`."""

        self.default = None


class Task(Slot):
    """Represents a single task in a ``Pipeline``. Tasks are slots that ensure all contained objects are plugs and own default values that are
    processors.
    """

    def __init__(self, proc_type: type[Processor] = Processor, default: Processor | Callable[..., Any] = lambda data: data, **kwargs: Any) -> None:
        """Initializes a new ``Task`` instance.

        Args:
            proc_type (type[Processor], optional): The type of ``Processor`` allowed for this ``Task``. Defaults to ``Processor``.
            default (Processor | Callable[..., Any], optional): The default processor for the ``Task``, which must either be a ``Processor`` or a
                function. Defaults to the identity function.
            **kwargs (Any): Keyword arguments that are passed to the constructor of the class one step up in the class hierarchy, i.e., ``Slot``.

        Raises:
            TypeError: The allowed ``Processor`` type for the ``Task``, ``proc_type``, is not of type ``Processor`` or a sub-class of ``Processor``.
        """

        if not issubclass(proc_type, Processor):
            raise TypeError('Only sub-classes of Processors are allowed.')
        if default is not None:
            default = ensure_processor(default, **kwargs)
        super().__init__(dtype=proc_type, default=default)

    @property
    def default(self) -> Processor | None:
        """Gets or sets the default processor of the ``Task``.

        Note:
            Actually, only the setter of the ``default`` property needs to be overridden. The proper way of overriding the setter of the ``default``
            property is to use the `@Slot.default.setter` idiom, but this, unfortunately, is detected as a false positive by MyPy. Although, they have
            promised to support this idiom (https://github.com/python/mypy/issues/5936), they have not done so yet, probably, because it is not used
            very often in the wild. Their workaround solution is to also override the getter and then use the new getter to override the setter and
            the deleter. I have tested it, only in the getter, `super().default` can be used, as it is not, yet, overridden, but in the setter and
            deleter, the complete function must be re-implemented. Also, when the getter is overridden, not only the setter, but also the deleter must
            be overridden, as otherwise, it will not be available anymore.

        Returns:
            Processor | None: Returns the task's default processor. If not set, `None` is returned.
        """

        default_processor: Processor | None = super().default
        return default_processor

    @default.setter
    def default(self, value: Processor | Callable[..., Any] | None) -> None:
        """Gets or sets the default processor of the ``Task``. Checks the new default processor is a ``Processor``. If not, it is converted to a
        ``Processor`` using the ``ensure_processor`` function. The default processor is checked for consistency with the ``dtype``.

        Args:
            value (Processor | Callable[..., Any] | None): The new default processor to set. If not set, `None` is returned.

        Raises:
            TypeError: The default processor is not consistent with the ``dtype``, i.e., it is neither `None`, nor of the type ``dtype`` or one of the
                types in the tuple ``dtype``.
        """

        if value is not None:
            value = ensure_processor(value)
        self._default = value

        try:
            self._consistent()
        except TypeError as exception:
            raise TypeError('The default processor is not consistent with the dtype.') from exception

    @default.deleter
    def default(self) -> None:
        """Deletes the task's default processor."""

        self._default = None

    def __call__(self, obj: Processor | Callable[..., Any] | None = None, default: Processor | Callable[..., Any] | None = None) -> TaskPlug:
        """Creates a new corresponding ``TaskPlug`` container.

        Args:
            obj (Processor | Callable[..., Any] | None, optional): A processor to initialize the newly created ``TaskPlug`` container's object value
                to. Defaults to `None`.
            default (Processor | Callable[..., Any] | None, optional): A processor to initialize the newly created ``TaskPlug`` container's default
                value to. Defaults to `None`.

        Returns:
            TaskPlug: Returns the newly created ``TaskPlug`` container instance, obeying the type and optionality constraints.
        """

        return TaskPlug(self, obj=obj, default=default)


class Pipeline(Processor):
    """The abstract base class for all pipelines."""

    def checkpoint_processes(self) -> OrderedDict[str, Processor]:
        """Finds the ``Processor`` that is a checkpoint and is closest to the output. The final checkpoint processor and all following processors are
        retrieved and returned in an ``OrderedDict``.

        Raises:
            RuntimeError: No checkpoints were defined.

        Returns:
            OrderedDict[str, Processor]: Returns an ``OrderedDict`` that contains the ``Processor`` that is closest to the output and a checkpoint, as
                well as all following processors. The processors in the ``OrderedDict`` are ordered in the same way as they were in the pipeline,
                i.e., from the checkpoint processor to the output processor.
        """

        checkpoint_processor_list = []
        for key, processor in reversed(self.collect_attr(Task).items()):
            checkpoint_processor_list.append((key, processor))
            if processor.is_checkpoint:
                break

        if checkpoint_processor_list and not checkpoint_processor_list[-1][1].is_checkpoint:
            raise RuntimeError('No checkpoints were defined.')

        checkpoint_processors = OrderedDict(checkpoint_processor_list[::-1])
        return checkpoint_processors

    def from_checkpoint(self) -> Any:
        """Re-evaluates the pipeline from the last check-pointed ``Processor`` using the output from the checkpoint as input.

        Raises:
            RuntimeError: If the check-pointed ``Processor`` closest to output does not have any ``checkpoint_data`` stored, i.e., the ``Processor``
                has not been called since being declared a checkpoint.

        Returns:
            Any: Returns the output of pipeline, starting from check-pointed ``Processor`` closest to output.
        """

        checkpoint_processes = self.checkpoint_processes()
        data_iterator = iter(checkpoint_processes.values())
        data = next(data_iterator).checkpoint_data
        if data is None:
            raise RuntimeError('No checkpoint data found, the whole pipeline must be run first for a checkpoint to exist.')
        for processor in data_iterator:
            data = processor(data)
        return data

    def function(self, data: Any) -> Any:
        """Propagate `data` through the whole pipeline from front to back, calling all Processors in series.

        Args:
            data (Any): The input data to the pipeline. This is the input data for the first ``Processor`` in the pipeline. The type of the input data
                depends on the first ``Processor`` in the pipeline.

        Returns:
            Any: Returns the output of the pipeline, which is the output of the of all processors in the pipeline that are flagged as pipeline
                outputs. If no processors are flagged as outputs, the output of the last processor is returned.
        """

        outputs = []
        for processor in self.collect_attr(Task).values():
            data = processor(data)
            if processor.is_output:
                outputs.append(data)
        if not outputs:
            return data
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def __repr__(self) -> str:
        """Generates a string representation of the pipeline, which contains all processors in the pipeline and their output types.

        Example:
            >>> MyPipeline()
            MyPipeline(
                FunctionProcessor(processing_function=lambda x: x.mean(1),) -> output:numpy.ndarray
                SciPyPDist(metric=sqeuclidean) -> output:numpy.ndarray
                RadialBasisFunction(sigma=0.1) -> output:numpy.ndarray
                MyProcess(stuff=3, func=Param(FunctionType, lambda x: x**2)) -> output:numpy.ndarray
            )

        Returns:
            str: Returns a string representation of the pipeline, which contains all processors in the pipeline and their output types.
        """

        pipeline = '\n    '.join([processor.__repr__() for processor in self.collect_attr(Task).values()])
        return f'{self.__class__.__name__}(\n    {pipeline}\n)'
