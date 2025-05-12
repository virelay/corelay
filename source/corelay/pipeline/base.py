"""A module that contains the base classes for pipelines, :py:class:`~corelay.pipeline.base.Pipeline`, and tasks of pipelines,
:py:class:`~corelay.pipeline.base.Task`, which are used to perform a specific set of operations on data.
"""

import collections
import typing
from collections.abc import Callable

from corelay.plugboard import Slot, Plug
from corelay.processor.base import ensure_processor, Processor
from corelay.utils import get_fully_qualified_name


class TaskPlug(Plug):
    """A task plug, which ensures that all contained objects are instances of :py:class:`~corelay.processor.base.Processor`."""

    def __init__(
        self,
        slot: Slot,
        obj: Processor | Callable[..., typing.Any] | None = None,
        default: Processor | Callable[..., typing.Any] | None = None,
        **kwargs: typing.Any
    ) -> None:
        """Initializes a new :py:class:`TaskPlug` instance.

        Args:
            slot (Slot): Slot instance to associate with this :py:class:`TaskPlug`.
            obj (Processor | Callable[..., typing.Any] | None): A :py:class:`~corelay.processor.base.Processor` held in the :py:class:`TaskPlug`
                container. If not set, :py:attr:`~TaskPlug.default` is returned as its value. Defaults to :py:obj:`None`.
            default (Processor | Callable[..., typing.Any] | None): A plug-dependent lower-priority :py:class:`~corelay.processor.base.Processor` held
                in the :py:class:`TaskPlug` container. If not set, :py:attr:`~corelay.plugboard.Plug.fallback` is returned. Defaults to
                :py:obj:`None`.
            **kwargs (typing.Any): Keyword arguments passed down to the base class constructor, for cooperativity's sake, which is the next class in
                the inheritance hierarchy.
        """

        if default is not None:
            default = ensure_processor(default, **kwargs)
        if obj is not None:
            obj = ensure_processor(obj)

        super().__init__(slot, obj=obj, default=default, **kwargs)

    @property
    def obj(self) -> Processor | None:
        """Gets or sets the :py:class:`~corelay.processor.base.Processor` contained in the :py:class:`TaskPlug`. If the :py:class:`TaskPlug` does not
        contain a :py:class:`~corelay.processor.base.Processor`, :py:attr:`~TaskPlug.default` is retrieved instead.

        Returns:
            Processor | None: Returns the :py:class:`~corelay.processor.base.Processor` contained in the :py:class:`TaskPlug`. If not set,
            :py:attr:`~TaskPlug.default` is returned.
        """

        # Actually, only the setter of the obj property needs to be overridden; The proper way of overriding the setter of the obj property is to use
        # the @Plug.obj.setter idiom, but this, unfortunately, is detected as a false positive by MyPy; although, they have promised to support this
        # idiom (https://github.com/python/mypy/issues/5936), they have not done so yet, probably, because it is not used very often in the wild;
        # their workaround solution is to also override the getter and then use the new getter to override the setter and the deleter; I have tested
        # it, only in the getter, super().obj can be used, as it is not, yet, overridden, but in the setter and deleter, the complete function must be
        # re-implemented; also, when the getter is overridden, not only the setter, but also the deleter must be overridden, as otherwise, it will not
        # be available anymore
        processor: Processor | None = super().obj
        return processor

    @obj.setter
    def obj(self, value: Processor | Callable[..., typing.Any] | None) -> None:
        """Gets or sets the :py:class:`~corelay.processor.base.Processor` contained in the :py:class:`TaskPlug` and checks for consistency. It is
        ensured first, that the value is a :py:class:`~corelay.processor.base.Processor`. If not, it is converted to a
        :py:class:`~corelay.processor.base.Processor` using the py:func:`ensure_processor` function.

        Args:
            value (Processor | Callable[..., typing.Any] | None): The :py:class:`~corelay.processor.base.Processor` to set.

        Raises:
            TypeError: The :py:class:`~corelay.processor.base.Processor` is not consistent with the :py:attr:`~corelay.plugboard.Plug.dtype`, i.e., it
                is neither :py:obj:`None`, nor of the type :py:attr:`~corelay.plugboard.Plug.dtype` or one of the types in the tuple
                :py:attr:`~corelay.plugboard.Plug.dtype`.
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
        """Deletes the :py:class:`~corelay.processor.base.Processor` contained in the :py:class:`TaskPlug` by setting it to :py:obj:`None`."""

        self.obj = None

    @property
    def default(self) -> typing.Any:
        """Gets or sets the default :py:class:`~corelay.processor.base.Processor` of the :py:class:`TaskPlug`. If the :py:attr:`~TaskPlug.default`
        :py:class:`~corelay.processor.base.Processor` is not set, then the :py:attr:`~corelay.plugboard.Plug.fallback`
        :py:class:`~corelay.processor.base.Processor` is retrieved instead.

        Returns:
            typing.Any: Returns the default :py:class:`~corelay.processor.base.Processor` of the :py:class:`TaskPlug`. If not set,
            :py:attr:`~corelay.plugboard.Plug.fallback` is returned.
        """

        # Actually, only the setter of the default property needs to be overridden; The proper way of overriding the setter of the default property is
        # to use the @Plug.default.setter idiom, but this, unfortunately, is detected as a false positive by MyPy; although, they have promised to
        # support this idiom (https://github.com/python/mypy/issues/5936), they have not done so yet, probably, because it is not used very often in
        # the wild; their workaround solution is to also override the getter and then use the new getter to override the setter and the deleter; I
        # have tested it, only in the getter, super().default can be used, as it is not, yet, overridden, but in the setter and deleter, the complete
        # function must be re-implemented; also, when the getter is overridden, not only the setter, but also the deleter must be overridden, as
        # otherwise, it will not be available anymore
        return super().default

    @default.setter
    def default(self, value: typing.Any) -> None:
        """Gets or sets the default :py:class:`~corelay.processor.base.Processor` of the :py:class:`TaskPlug` and checks for consistency. It is
        ensured first, that the new default value is a :py:class:`~corelay.processor.base.Processor`. If not, it is converted to a
        :py:class:`~corelay.processor.base.Processor` using the py:func:`ensure_processor` function.

        Args:
            value (typing.Any): The new default :py:class:`~corelay.processor.base.Processor` to set.

        Raises:
            TypeError: The default :py:class:`~corelay.processor.base.Processor` is not consistent with the :py:attr:`~corelay.plugboard.Plug.dtype`,
                i.e., it is neither :py:obj:`None`, nor of the type :py:attr:`~corelay.plugboard.Plug.dtype` or one of the types in the tuple
                :py:attr:`~corelay.plugboard.Plug.dtype`.
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
        """Deletes the default :py:class:`~corelay.processor.base.Processor` of the :py:class:`TaskPlug` by setting it to :py:obj:`None`."""

        self.default = None


class Task(Slot):
    """Represents a single task in a :py:class:`Pipeline`. Tasks are slots that ensure all contained objects are plugs and own default values that are
    instances of :py:class:`~corelay.processor.base.Processor`.
    """

    def __init__(
        self,
        proc_type: type[Processor] = Processor,
        default: Processor | Callable[..., typing.Any] = lambda data: data,
        **kwargs: typing.Any
    ) -> None:
        """Initializes a new :py:class:`Task` instance.

        Args:
            proc_type (type[Processor]): The type of :py:class:`~corelay.processor.base.Processor` allowed for this :py:class:`Task`. Defaults to
                :py:class:`~corelay.processor.base.Processor`.
            default (Processor | Callable[..., typing.Any]): The default :py:class:`~corelay.processor.base.Processor` for the :py:class:`Task`, which
                must either be a :py:class:`~corelay.processor.base.Processor` or a function. Defaults to the identity function.
            **kwargs (typing.Any): Keyword arguments that are passed to the constructor of the class one step up in the class hierarchy, i.e.,
                :py:class:`~corelay.plugboard.Slot`.

        Raises:
            TypeError: The allowed :py:class:`~corelay.processor.base.Processor` type for the :py:class:`Task`, ``proc_type``, is not of type
                :py:class:`~corelay.processor.base.Processor` or a sub-class of :py:class:`~corelay.processor.base.Processor`.
        """

        if not issubclass(proc_type, Processor):
            raise TypeError('Only sub-classes of Processors are allowed.')
        if default is not None:
            default = ensure_processor(default, **kwargs)
        super().__init__(dtype=proc_type, default=default)

    def __repr__(self) -> str:
        """Returns a :py:class:`str` representation of the :py:class:`Task` instance.

        Returns:
            str: Returns a :py:class:`str` representation of the :py:class:`Task` instance.
        """

        # Sphinx AutoDoc uses __repr__ when it encounters the metadata of typing.Annotated; this is a reasonable thing to do, but then it tries to
        # resolve the resulting string as types for cross-referencing, which is not possible with the default implementation of __repr__; to be able
        # to get proper documentation, the fully-qualified name of the class is returned, because this enable Sphinx AutoDoc to reference the class in
        # the documentation
        return f'~{get_fully_qualified_name(self)}'

    @property
    def default(self) -> Processor | None:
        """Gets or sets the default :py:class:`~corelay.processor.base.Processor` of the :py:class:`Task`.

        Returns:
            Processor | None: Returns the task's default :py:class:`~corelay.processor.base.Processor`. If not set, :py:obj:`None` is returned.
        """

        # Actually, only the setter of the default property needs to be overridden; The proper way of overriding the setter of the default property is
        # to use the @Plug.default.setter idiom, but this, unfortunately, is detected as a false positive by MyPy; although, they have promised to
        # support this idiom (https://github.com/python/mypy/issues/5936), they have not done so yet, probably, because it is not used very often in
        # the wild; their workaround solution is to also override the getter and then use the new getter to override the setter and the deleter; I
        # have tested it, only in the getter, super().default can be used, as it is not, yet, overridden, but in the setter and deleter, the complete
        # function must be re-implemented; also, when the getter is overridden, not only the setter, but also the deleter must be overridden, as
        # otherwise, it will not be available anymore
        default_processor: Processor | None = super().default
        return default_processor

    @default.setter
    def default(self, value: Processor | Callable[..., typing.Any] | None) -> None:
        """Gets or sets the default :py:class:`~corelay.processor.base.Processor` of the :py:class:`Task`. Checks the new default
        :py:class:`~corelay.processor.base.Processor` is a :py:class:`~corelay.processor.base.Processor`. If not, it is converted to a
        :py:class:`~corelay.processor.base.Processor` using the py:func:`ensure_processor` function. The default
        :py:class:`~corelay.processor.base.Processor` is checked for consistency with the :py:attr:`~corelay.plugboard.Slot.dtype`.

        Args:
            value (Processor | Callable[..., typing.Any] | None): The new default :py:class:`~corelay.processor.base.Processor` to set. If not set,
                :py:obj:`None` is returned.

        Raises:
            TypeError: The default :py:class:`~corelay.processor.base.Processor` is not consistent with the :py:attr:`~corelay.plugboard.Slot.dtype`,
                i.e., it is neither :py:obj:`None`, nor of the type :py:attr:`~corelay.plugboard.Slot.dtype` or one of the types in the tuple
                :py:attr:`~corelay.plugboard.Slot.dtype`.
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
        """Deletes the task's default :py:class:`~corelay.processor.base.Processor`."""

        self._default = None

    def __call__(
        self,
        obj: Processor | Callable[..., typing.Any] | None = None,
        default: Processor | Callable[..., typing.Any] | None = None
    ) -> TaskPlug:
        """Creates a new corresponding :py:class:`TaskPlug` container.

        Args:
            obj (Processor | Callable[..., typing.Any] | None): A :py:class:`~corelay.processor.base.Processor` to initialize the newly created
                :py:class:`TaskPlug` container's object value to. Defaults to :py:obj:`None`.
            default (Processor | Callable[..., typing.Any] | None): A :py:class:`~corelay.processor.base.Processor` to initialize the newly created
                :py:class:`TaskPlug` container's default value to. Defaults to :py:obj:`None`.

        Returns:
            TaskPlug: Returns the newly created :py:class:`TaskPlug` container instance, obeying the type and optionality constraints.
        """

        return TaskPlug(self, obj=obj, default=default)


class Pipeline(Processor):
    """The abstract base class for all pipelines."""

    def checkpoint_processes(self) -> collections.OrderedDict[str, Processor]:
        """Finds the :py:class:`~corelay.processor.base.Processor` that is a checkpoint and is closest to the output. The final checkpoint
        :py:class:`~corelay.processor.base.Processor` and all following instances of :py:class:`~corelay.processor.base.Processor` are retrieved and
        returned in an :py:class:`collections.OrderedDict`.

        Raises:
            RuntimeError: No checkpoints were defined.

        Returns:
            collections.OrderedDict[str, Processor]: Returns an :py:class:`collections.OrderedDict` that contains the
            :py:class:`~corelay.processor.base.Processor` that is closest to the output and a checkpoint, as well as all following instances of
            :py:class:`~corelay.processor.base.Processor`. The instances of :py:class:`~corelay.processor.base.Processor` in th
            :py:class:`collections.OrderedDict` are ordered in the same way as they were in the instance of :py:class:`Pipeline`, i.e., from the
            checkpoint :py:class:`~corelay.processor.base.Processor` to the output :py:class:`~corelay.processor.base.Processor`.
        """

        checkpoint_processor_list = []
        for key, processor in reversed(self.collect_attr(Task).items()):
            checkpoint_processor_list.append((key, processor))
            if processor.is_checkpoint:
                break

        if checkpoint_processor_list and not checkpoint_processor_list[-1][1].is_checkpoint:
            raise RuntimeError('No checkpoints were defined.')

        checkpoint_processors = collections.OrderedDict(checkpoint_processor_list[::-1])
        return checkpoint_processors

    def from_checkpoint(self) -> typing.Any:
        """Re-evaluates the :py:class:`Pipeline` from the last check-pointed :py:class:`~corelay.processor.base.Processor` using the output from the
        checkpoint as input.

        Raises:
            RuntimeError: If the check-pointed :py:class:`~corelay.processor.base.Processor` closest to output does not have any
                :py:attr:`~corelay.processor.base.Processor.checkpoint_data` stored, i.e., the :py:class:`~corelay.processor.base.Processor` has not
                been called since being declared a checkpoint.

        Returns:
            typing.Any: Returns the output of :py:class:`Pipeline`, starting from check-pointed :py:class:`~corelay.processor.base.Processor` closest
            to output.
        """

        checkpoint_processes = self.checkpoint_processes()
        data_iterator = iter(checkpoint_processes.values())
        data = next(data_iterator).checkpoint_data
        if data is None:
            raise RuntimeError('No checkpoint data found, the whole pipeline must be run first for a checkpoint to exist.')
        for processor in data_iterator:
            data = processor(data)
        return data

    def function(self, data: typing.Any) -> typing.Any:
        """Propagate `data` through the whole :py:class:`Pipeline` from front to back, calling all Processors in series.

        Args:
            data (typing.Any): The input data to the :py:class:`Pipeline`. This is the input data for the first
                :py:class:`~corelay.processor.base.Processor` in the :py:class:`Pipeline`. The type of the input data depends on the first
                :py:class:`~corelay.processor.base.Processor` in the :py:class:`Pipeline`.

        Returns:
            typing.Any: Returns the output of the :py:class:`Pipeline`, which is the output of the of all instances of
            :py:class:`~corelay.processor.base.Processor` in the :py:class:`Pipeline` that are flagged as outputs. If no instances of
            :py:class:`~corelay.processor.base.Processor` are flagged as outputs, the output of the last :py:class:`~corelay.processor.base.Processor`
            is returned.
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
        """Generates a :py:class:`str` representation of the :py:class:`Pipeline`, which contains all instances of
        :py:class:`~corelay.processor.base.Processor` in the :py:class:`Pipeline` and their output types.

        Example:
            >>> MyPipeline()
            MyPipeline(
                FunctionProcessor(processing_function=lambda x: x.mean(1),) -> numpy.ndarray
                SciPyPDist(metric=sqeuclidean) -> numpy.ndarray
                RadialBasisFunction(sigma=0.1) -> numpy.ndarray
                MyProcess(stuff=3, func=Param(FunctionType, lambda x: x**2)) -> numpy.ndarray
            )

        Returns:
            str: Returns a :py:class:`str` representation of the :py:class:`Pipeline`, which contains all instances of
            :py:class:`~corelay.processor.base.Processor` in the :py:class:`Pipeline` and their output types.
        """

        pipeline = '\n    '.join([repr(processor) for processor in self.collect_attr(Task).values()])
        return f'{self.__class__.__name__}(\n    {pipeline}\n)'
