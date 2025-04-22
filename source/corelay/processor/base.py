"""A module that contains the base classes ``Param`` and ``Processor``."""

import inspect
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from types import FunctionType, LambdaType
from typing import Annotated, Any

from corelay.base import Param
from corelay.io import NoStorage, NoDataSource, NoDataTarget
from corelay.io.storage import Storable
from corelay.plugboard import Plugboard


class Processor(ABC, Plugboard):
    """The abstract base class of processors of tasks in a pipeline instance."""

    is_output: Annotated[bool, Param(bool, False)]
    """Contains a value indicating whether this ``Processor`` is the output of a ``Pipeline``."""

    is_checkpoint: Annotated[bool, Param(bool, False)]
    """Contains a value indicating whether check-pointed pipeline computations should start at this point, if there exists a previously computed
    checkpoint value.
    """

    io: Annotated[Storable, Param(Storable, NoStorage())]
    """Contains an IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this run or in subsequent runs
    of the ``Pipeline``.
    """

    checkpoint_data: Any
    """If this ``Processor`` is a checkpoint, and if the processor was called at least once, stores the output of this processor."""

    def __init__(
        self,
        *args: Any,
        is_output: bool | None = None,
        is_checkpoint: bool | None = None,
        io: Storable | None = None,
        **kwargs: Any
    ) -> None:
        """Initializes a new ``Processor`` instance. All defined ``Param`` class attributes are initialized either to their respective default values
        or, if supplied as keyword argument, to the value supplied.

        Args:
            *args (Any): A list of the positional arguments, which will be used to initialize the parameters of the ``Processor`` that were marked as
                positional.
            is_output (bool | None, optional): A value indicating whether this ``Processor`` is the output of a ``Pipeline``. If `None` is specified,
                the corresponding ``Param`` will default to its defined default value, which is `False`.
            is_checkpoint (bool | None, optional): A value indicating whether check-pointed pipeline computations should start at this point, if there
                exists a previously computed checkpoint value. If `None` is specified, the corresponding ``Param`` will default to its defined default
                value, which is `False`.
            io (Storable | None, optional): An IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this
                run or in subsequent runs of the ``Pipeline``. If `None` is specified, the corresponding ``Param`` will default to its defined default
                value, which is an instance of ``NoStorage``.
            **kwargs (Any): A dictionary of keyword arguments, which will be used to initialize the parameters of the ``Processor`` that were marked
                as keyword arguments. The keys of the dictionary are the names of the parameters, and the values are the values to be assigned to
                those parameters.

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
        self.checkpoint_data: Any = None

    @abstractmethod
    def function(self, data: Any) -> Any:
        """Applies a function to the input data. This function should be implemented by subclasses of ``Processor``.

        Args:
            data (Any): The input data to this ``Processor``.

        Raises:
            NotImplementedError: This is an abstract method and should be implemented by subclasses of ``Processor`` and therefore always raises the
                ``NotImplementedError`` exception.

        Returns:
            Any: Returns the output of the function applied to the input data.
        """

        raise NotImplementedError('This is an abstract method and should be implemented by subclasses of Processor.')

    def __call__(self, data: Any) -> Any:
        """Apply ``function`` on the input data and saves the output if the ``is_checkpoint`` ``Param`` was set to `True`.

        Args:
            data (Any): The input data to this ``Processor``.

        Returns:
            Any: Returns the output of the function applied to the input data.
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

    def param_values(self) -> dict[str, Any]:
        """Get values for all parameters defined through :obj:`Param` attributes.

        Returns:
            dict[str, Any]: Returns a dictionary containing the names of the parameters as keys and their values as values.
        """

        return self.collect_attr(Param)

    def identifiers(self) -> OrderedDict[str, Any]:
        """Returns a dict containing the class qualifier name, as well all Parameters marked as identifiers with their
        values

        Returns:
            OrderedDict[str, Any]: Returns an ordered dictionary, containing the class qualifier name and all parameters marked as identifiers with
                their values.
        """

        result = OrderedDict(name=type(self).__qualname__)
        result.update((key, getattr(self, key)) for key, parameter in self.collect(Param).items() if parameter.is_identifier)
        return result

    def copy(self) -> 'Processor':
        """Copies this processor, by creating a new ``Processor`` instance with the same values for the parameters defined as ``Param`` class
        attributes and the same checkpoint data.

        Returns:
            Processor: Returns a copy of this ``Processor`` instance.
        """

        new_processor = type(self)(**self.param_values())
        new_processor.checkpoint_data = self.checkpoint_data
        return new_processor

    @property
    def _output_repr(self) -> str:
        """Gets a string for the output of the ``Processor``.

        Returns:
            str: Returns the string representation of the output of the ``Processor``.
        """

        return 'output: numpy.ndarray'

    def __repr__(self) -> str:
        """Generates a string representation of the ``Processor`` instance, including the class name, the parameters and their values, and the output
        representation, e.g., `ProcessorName(metric=sqeuclidean, function=lambda x: x.mean(1)) -> output: numpy.ndarray`.

        Returns:
            str: Returns a string representation of the ``Processor`` instance.
        """

        def get_source_code_for_lambda_expression(lambda_expression: LambdaType | Any) -> str:
            """Retrieves the source code of the specified lambda expression. If the argument is not a lambda expression, its string representation is
            returned.

            Args:
                lambda_expression (LambdaType | Any): The lambda expression for which the source code should be retrieved.

            Returns:
                str: Returns the source code of the lambda expression as a string, or a string representation of the argument if it is not a lambda
                    expression.
            """

            if isinstance(lambda_expression, LambdaType):
                return inspect.getsource(lambda_expression).split('=', 1)[1].strip()
            return str(lambda_expression)

        name = self.__class__.__name__
        parameters = ', '.join([f'{k}={get_source_code_for_lambda_expression(v)}' for k, v in self.param_values().items() if v])
        return f'{name}({parameters}) -> {self._output_repr}'


class FunctionProcessor(Processor):
    """A ``Processor`` that executes a user-defined function.

    Args:
        processing_function (FunctionType): The function around which to create the :obj:`FunctionProcessor`. This function will be invoked when the
            ``function`` method is invoked or the ``FunctionProcessor`` object is called like a function. Depending on whether ``bind_method`` is
            `True` or `False`, it wil be bound as a method to the ``FunctionProcessor`` object.
        is_output (bool, optional): A value indicating whether this ``FunctionProcessor`` is the output of a ``Pipeline``. Defaults to `False`.
        is_checkpoint (bool | None, optional): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to `False`.
        io (Storable | None, optional): An IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this
            run or in subsequent runs of the ``Pipeline``. Defaults to an instance of ``NoStorage``.
        bind_method (bool, optional): A value indicating whether the ``processing_function`` will be bound to this class, enabling it to access
            `self`. Defaults to `False`.
    """

    processing_function: Annotated[FunctionType, Param(FunctionType, lambda _self, data: data, positional=True)]
    """The function around which to create the :obj:`FunctionProcessor`. This function will be invoked when the ``function`` method is invoked or the
    ``FunctionProcessor`` object is called like a function. Depending on whether ``bind_method`` is `True` or `False`, it wil be bound as a method to
    the ``FunctionProcessor`` object.
    """

    bind_method: Annotated[bool, Param(bool, False)]
    """A value indicating whether the ``processing_function`` will be bound to this class, enabling it to access `self`."""

    def function(self, data: Any) -> Any:
        """Invokes the function bound to this class with the input data.

        Note:
            In a previous version of CoRelAy, the ``processing_function`` was actually bound to the class in the ``__call__`` method, but this caused
            typing issues, as static type checkers like MyPy believed that the ``FunctionProcessor`` class was still abstract, as it did not
            explicitly override the ``function`` method. The ``processing_function`` used to be called just ``function``, which meant, that during
            runtime, functionally, the ``function`` method was overridden, as its slot would have been taken by the ``function`` parameter.
            Statically, however, this was not the case. Overriding the ``function`` method and still binding the ``processing_function`` to the class
            in the ``__call__`` method causes more typing issues, as the static type checker does not allow us to write to a method slot. For this
            reason, the ``function`` method was overridden and internally calls the ``processing_function`` method with `self` as the first argument.
            Functionally, this should be equivalent to the previous version, but it is not guaranteed that it is in every use case. This might have
            rethought and changed in the future.

        Args:
            data (Any): The input data to this ``Processor``.

        Returns:
            Any: Returns the output of the function applied to the input data.
        """

        if self.bind_method:
            return self.processing_function(self, data)
        return self.processing_function(data)


def ensure_processor(processor_or_callable: Processor | Callable[..., Any], **kwargs: Any) -> Processor:
    """Ensures that the specified processor or callable argument ``processor_or_callable`` is of type ``Processor`` and, if it is not, but callable,
    make it a ``FunctionProcessor``. Sets the attributes of resulting processor as stated in `**kwargs`.

    Args:
        processor_or_callable (Processor | Callable[..., Any]): The processor or callable for which to ensure that it is a ``Processor``.
        **kwargs (Any): The keyword arguments to be passed to the ``Processor``. These keyword arguments are used to set the values of the parameters
            of the ``Processor``.

    Raises:
        TypeError: The supplied processor or callable ``processor_or_callable`` is neither a ``Processor`` nor callable.

    Returns:
        Processor: Returns the original ``processor_or_callable`` if it was a ``Processor``, or a new ``FunctionProcessor``, which calls it if it was
            a callable. The attributes of the resulting processor are set as stated in `**kwargs`.
    """

    if not isinstance(processor_or_callable, Processor):
        if callable(processor_or_callable):
            processor_or_callable = FunctionProcessor(processing_function=processor_or_callable)
        else:
            raise TypeError(f'Supplied processor {processor_or_callable} is neither a Processor, nor callable!')
    processor_or_callable.update_defaults(**kwargs)
    return processor_or_callable
