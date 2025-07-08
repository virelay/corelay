"""A module that contains the abstract base class for all processors, :py:class:`~corelay.processor.base.Processor`, as well as a basic processor,
:py:class:`~corelay.processor.base.FunctionProcessor`, which invokes a specified function. Furthermore, the module contains a function, which ensures
that a specified argument is of type :py:class:`~corelay.processor.base.Processor` and, if it is not, but callable, makes it a
:py:class:`~corelay.processor.base.FunctionProcessor`.
"""

import collections
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable
from types import FunctionType
from typing import Annotated

from corelay.base import Param
from corelay.io import NoStorage, NoDataSource, NoDataTarget
from corelay.io.storage import Storable
from corelay.plugboard import Plugboard
from corelay.utils import get_object_representation


class Processor(ABC, Plugboard):
    """The abstract base class of processors, which perform specific tasks in a :py:class:`corelay.pipeline.base.Pipeline` instance."""

    is_output: Annotated[bool, Param(bool, False)]
    """Contains a value indicating whether this :py:class:`Processor` is the output of a
    :py:class:`~corelay.pipeline.base.Pipeline`.
    """

    is_checkpoint: Annotated[bool, Param(bool, False)]
    """Contains a value indicating whether check-pointed pipeline computations should start at this point, if there exists a previously computed
    checkpoint value.
    """

    io: Annotated[Storable, Param(Storable, NoStorage())]
    """Contains an IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can then be re-used
    in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`.
    """

    checkpoint_data: typing.Any
    """If this :py:class:`Processor` is a checkpoint, and if the processor was called at least once, stores the output of this
    processor.
    """

    def __init__(
        self,
        *args: typing.Any,
        is_output: bool | None = None,
        is_checkpoint: bool | None = None,
        io: Storable | None = None,
        **kwargs: typing.Any
    ) -> None:
        """Initializes a new :py:class:`Processor` instance. All defined :py:class:`~corelay.base.Param` class attributes are
        initialized either to their respective default values or, if supplied as keyword argument, to the value supplied.

        Args:
            *args (typing.Any): A :py:class:`list` of the positional arguments, which will be used to initialize the parameters of the
                :py:class:`Processor` that were marked as positional.
            is_output (bool | None): A value indicating whether this :py:class:`Processor` is the output of a
                :py:class:`~corelay.pipeline.base.Pipeline`. If :py:obj:`None` is specified, the corresponding
                :py:class:`~corelay.base.Param` will default to its defined default value, which is :py:obj:`False`.
            is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
                previously computed checkpoint value. If :py:obj:`None` is specified, the corresponding :py:class:`~corelay.base.Param`
                will default to its defined default value, which is :py:obj:`False`.
            io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which
                can then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. If :py:obj:`None` is
                specified, the corresponding :py:class:`~corelay.base.Param` will default to its defined default value, which is an instance of
                :py:class:`corelay.io.NoStorage`.
            **kwargs (typing.Any): A :py:class:`dict` of keyword arguments, which will be used to initialize the parameters of the
                :py:class:`Processor` that were marked as keyword arguments. The keys of the :py:class:`dict` are the names of the parameters, and the
                values are the values to be assigned to those parameters.

        Raises:
            TypeError: The number of positional arguments supplied is greater than the number of parameters that were marked as positional or a
                parameter was defined as both positional and a keyword argument.
        """

        # Pairs the parameters that are marked as positional with their corresponding positional arguments; this is a dictionary where the keys are
        # the names of the parameters and the values are the corresponding positional arguments
        parameter_arguments = dict(zip((name for name, parameter in self.collect(Param).items() if parameter.is_positional), args))
        if len(parameter_arguments) < len(args):
            raise TypeError(f'Expected at most {len(parameter_arguments)} positional arguments, got {len(args)}.')

        # PyDocLint does not support the documentation of the constructor parameters both in the __init__ method and the class docstring, so we have
        # to add the documentation for the is_output, is_checkpoint, and io parameters here; therefore, the parameters have to be added to the keyword
        # arguments manually
        kwargs['is_output'] = is_output
        kwargs['is_checkpoint'] = is_checkpoint
        kwargs['io'] = io

        # Checks if any of the positional arguments were also specified as keyword arguments
        for name in parameter_arguments:
            if name in kwargs:
                raise TypeError(f'Argument "{name}" was specified as both positional and a keyword argument.')

        # Adds the keyword arguments to the parameter arguments
        parameter_arguments.update(kwargs)

        # Calls the constructor of the parent class, Plugboard, with the parameter arguments as keyword arguments
        super().__init__(**parameter_arguments)

        # Initializes the checkpoint data to None
        self.checkpoint_data: typing.Any = None

    @abstractmethod
    def function(self, data: typing.Any) -> typing.Any:
        """Applies a function to the input data. This function should be implemented by subclasses of :py:class:`Processor`.

        Args:
            data (typing.Any): The input data to this :py:class:`Processor`.

        Raises:
            NotImplementedError: This is an abstract method and should be implemented by subclasses of :py:class:`Processor` and therefore always
                raises the :py:class:`NotImplementedError` exception.

        Returns:
            typing.Any: Returns the output of the function applied to the input data.
        """

        raise NotImplementedError('This is an abstract method and should be implemented by subclasses of Processor.')

    def __call__(self, data: typing.Any) -> typing.Any:
        """Applies :py:meth:`~Processor.function` on the input data and saves the output if the :py:attr:`~Processor.is_checkpoint`
        :py:class:`~corelay.base.Param` was set to :py:obj:`True`.

        Args:
            data (typing.Any): The input data to this :py:class:`Processor`.

        Returns:
            typing.Any: Returns the output of the function applied to the input data.
        """

        try:
            out = self.io.read(data_in=data, meta=self.identifiers())
        except NoDataSource:
            out = self.function(data)
            try:
                self.io.write(data_out=out, data_in=data, meta=self.identifiers())
            except NoDataTarget:
                pass
        if self.is_checkpoint:
            self.checkpoint_data = out
        return out

    def param_values(self) -> dict[str, typing.Any]:
        """Get values for all parameters defined through :py:class:`~corelay.base.Param` attributes.

        Returns:
            dict[str, typing.Any]: Returns a :py:class:`dict` containing the names of the parameters as keys and their values as values.
        """

        return self.collect_attr(Param)

    def identifiers(self) -> collections.OrderedDict[str, typing.Any]:
        """Returns a dict containing the class qualifier name, as well all parameters marked as identifiers with their values.

        Returns:
            collections.OrderedDict[str, typing.Any]: Returns an :py:class:`collections.OrderedDict`, containing the class qualifier name and all
            parameters marked as identifiers with their values.
        """

        result = collections.OrderedDict(name=type(self).__qualname__)
        result.update((key, getattr(self, key)) for key, parameter in self.collect(Param).items() if parameter.is_identifier)
        return result

    def copy(self) -> 'Processor':
        """Copies this processor, by creating a new :py:class:`Processor` instance with the same values for the parameters defined as
        :py:class:`~corelay.base.Param` class attributes and the same checkpoint data.

        Returns:
            Processor: Returns a copy of this :py:class:`Processor` instance.
        """

        new_processor = type(self)(**self.param_values())
        new_processor.checkpoint_data = self.checkpoint_data
        return new_processor

    @property
    def _output_repr(self) -> str:
        """Gets a :py:class:`str` for the output of the :py:class:`Processor`.

        Returns:
            str: Returns the :py:class:`str` representation of the output of the :py:class:`Processor`.
        """

        return 'numpy.ndarray'

    def __repr__(self) -> str:
        """Generates a :py:class:`str` representation of the :py:class:`Processor` instance, including the class name, the parameters and their
        values, and the output representation, e.g., `ProcessorName(metric=sqeuclidean, function=lambda x: x.mean(1)) -> numpy.ndarray`.

        Returns:
            str: Returns a :py:class:`str` representation of the :py:class:`Processor` instance.
        """

        name = self.__class__.__name__
        parameters = ', '.join([
            f'{parameter_key}={get_object_representation(parameter_value)}'
            for parameter_key, parameter_value
            in self.param_values().items()
            if parameter_value
        ])
        return f'{name}({parameters}) -> {self._output_repr}'


class FunctionProcessor(Processor):
    """A :py:class:`Processor` that executes a user-defined function.

    Args:
        processing_function (FunctionType): The function around which to create the :py:class:`FunctionProcessor`. This function will be invoked when
            the :py:meth:`~FunctionProcessor.function` method is invoked or the :py:class:`FunctionProcessor` object is called like a function.
            Depending on whether :py:attr:`~FunctionProcessor.bind_method` is :py:obj:`True` or :py:obj:`False`, it wil be bound as a method to the
            :py:class:`FunctionProcessor` object.
        is_output (bool): A value indicating whether this :py:class:`FunctionProcessor` is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        bind_method (bool): A value indicating whether the :py:attr:`~FunctionProcessor.processing_function` will be bound to this class, enabling it
            to access `self`. Defaults to :py:obj:`False`.
    """

    processing_function: Annotated[FunctionType, Param(FunctionType, lambda _self, data: data, positional=True)]
    """The function around which to create the :py:class:`FunctionProcessor`. This function will be invoked when the
    :py:meth:`~FunctionProcessor.function` method is invoked or the :py:class:`FunctionProcessor` object is called like a function. Depending on
    whether :py:attr:`~FunctionProcessor.bind_method` is :py:obj:`True` or :py:obj:`False`, it wil be bound as a method to the
    :py:class:`FunctionProcessor` object.
    """

    bind_method: Annotated[bool, Param(bool, False)]
    """A value indicating whether the :py:attr:`~FunctionProcessor.processing_function` will be bound to this class, enabling it to access `self`."""

    def function(self, data: typing.Any) -> typing.Any:
        """Invokes the function bound to this class with the input data.

        Note:
            In a previous version of CoRelAy, the :py:attr:`~FunctionProcessor.processing_function` was actually bound to the class in the
            :py:meth:`~Processor.__call__` method, but this caused typing issues, as static type checkers like MyPy believed that the
            :py:class:`FunctionProcessor` class was still abstract, as it did not explicitly override the :py:meth:`~FunctionProcessor.function`
            method. The :py:attr:`~FunctionProcessor.processing_function` used to be called just :py:meth:`~FunctionProcessor.function`, which meant,
            that during runtime, functionally, the :py:meth:`~FunctionProcessor.function` method was overridden, as its slot would have been taken by
            the :py:meth:`~FunctionProcessor.function` parameter. Statically, however, this was not the case. Overriding the
            :py:meth:`~FunctionProcessor.function` method and still binding the :py:attr:`~FunctionProcessor.processing_function` to the class in the
            :py:meth:`~Processor.__call__` method causes more typing issues, as the static type checker does not allow us to write to a method slot.
            For this reason, the :py:meth:`~FunctionProcessor.function` method was overridden and internally calls the
            :py:attr:`~FunctionProcessor.processing_function` method with `self` as the first argument. Functionally, this should be equivalent to the
            previous version, but it is not guaranteed that it is in every use case. This might have rethought and changed in the future.

        Args:
            data (typing.Any): The input data to this :py:class:`Processor`.

        Returns:
            typing.Any: Returns the output of the function applied to the input data.
        """

        if self.bind_method:
            return self.processing_function(self, data)
        return self.processing_function(data)


def ensure_processor(processor_or_callable: Processor | Callable[..., typing.Any], **kwargs: typing.Any) -> Processor:
    """Ensures that the specified processor or callable argument ``processor_or_callable`` is of type :py:class:`Processor` and, if it is not, but
    callable, make it a :py:class:`FunctionProcessor`. Sets the attributes of resulting processor as stated in `**kwargs`.

    Args:
        processor_or_callable (Processor | Callable[..., typing.Any]): The processor or callable for which to ensure that it is a
            :py:class:`Processor`.
        **kwargs (typing.Any): The keyword arguments to be passed to the :py:class:`Processor`. These keyword arguments are used to set the values of
            the parameters of the :py:class:`Processor`.

    Raises:
        TypeError: The supplied processor or callable ``processor_or_callable`` is neither a :py:class:`Processor` nor callable.

    Returns:
        Processor: Returns the original ``processor_or_callable`` if it was a :py:class:`Processor`, or a new :py:class:`FunctionProcessor`, which
        calls it if it was a callable. The attributes of the resulting processor are set as stated in `**kwargs`.
    """

    if not isinstance(processor_or_callable, Processor):
        if callable(processor_or_callable):
            processor_or_callable = FunctionProcessor(processing_function=processor_or_callable)
        else:
            raise TypeError(f'Supplied processor {processor_or_callable} is neither a Processor, nor callable!')
    processor_or_callable.update_defaults(**kwargs)
    return processor_or_callable
