"""An example script, which demonstrates the usage of CoRelAy's pipeline and processor classes."""

from types import FunctionType
from typing import Annotated, Any

import numpy

from corelay.base import Param
from corelay.pipeline.base import Pipeline, Task
from corelay.processor.affinity import Affinity, RadialBasisFunction
from corelay.processor.base import FunctionProcessor, Processor
from corelay.processor.distance import Distance, SciPyPDist


class MyProcessor(Processor):
    """A custom CoRelAy processor, which applies a configurable function to its input data and multiplies it by a configurable value."""

    multiplier: Annotated[int, Param(dtype=int, default=2)]
    """An integer parameter, which is multiplied with the result of the function.

    Note:
        Parameters are registered by defining class attributes of type ``Param``. These will be automatically realized as instance attributes at
        runtime and can be initialized in ``__init__`` using positional and keyword arguments of the same name, depending on the ``is_positional``
        parameter of the ``Param``. The first value is a type specification, the second a default value.
    """

    function_to_apply: Annotated[FunctionType, Param(FunctionType, lambda x: x**2)]
    """A function, which is applied to the input data.

    Note:
        As class methods have to be bound explicitly, ``function_to_apply`` here acts like a static function of ``MyProcessor``. If you want to have
        processor that applies a custom function and that is bound to the class and has access to self, please refer to
        :obj:`corelay.processor.base.FunctionProcessor`.
    """

    def function(self, data: Any) -> Any:
        """Applies the custom function ``function_to_apply`` to the input data and multiplies it by the parameter ``multiplier``.

        Args:
            data (Any): The input data that is to be processed.

        Returns:
            Any: Returns the processed data.
        """

        # Parameters can be accessed as self.<parameter-name>
        return self.multiplier * self.function_to_apply(data) + 3


class MyPipeline(Pipeline):
    """A custom CoRelAy pipeline, which applies a series of processors to its input data."""

    pre_pre_process = Task(proc_type=FunctionProcessor, default=lambda self, x: x * 2, bind_method=True)
    """A pre-pre-processing task, which applies a function to the input data. By default, the input data is multiplied by 2. The ``FunctionProcessor``
    class is a ``Processor`` that applies a function to the input data.

    Note:
        Tasks are registered by creating a class attribute of type ``Task`` and, like parameters, are expected to be supplied with the same name in
        ``__init__`` as a keyword argument. The first value is an optional type that determines which type of ``Process`` is expected, second is a
        default value, which has to be an instance of that type. If the default argument is not a ``Process``, it will be converted to a
        ``FunctionProcessor``. By default, functions fed to ``FunctionProcessors`` are by not bound to the class. To bind them, we can supply
        `bind_method=True` to the ``FunctionProcessor``. Supplying it to the task changes the default value of the ``Processor`` before creation.
    """

    pre_process = Task(proc_type=FunctionProcessor, default=lambda x: x**2)
    """A pre-processing task, which applies a function to the input data. By default, the input data is squared.

    Note:
        The ``bind_method`` parameter is omitted here and therefore defaults to False. This means that the function is not bound to the class and does
        not have access to ``self``.
    """

    pairwise_distance = Task(Distance, SciPyPDist(metric='sqeuclidean'))
    """A task, which applies a pairwise distance function to the input data. By default, the squared euclidean distance is used. The ``Distance``
    class is a base class for all distance processors.
    """

    affinity = Task(Affinity, RadialBasisFunction(sigma=1.0))
    """A task, which applies an affinity function to the input data. By default, the radial basis function is used. The ``Affinity`` class is a base
    class for all affinity processors.
    """

    post_process = Task()
    """A post-processing task, which does nothing by default and returns the input data as is."""


def main() -> None:
    """The entrypoint to the corelay_basics script."""

    # Creates a new pipeline without specifying any parameters, which means that the default values of the tasks will be used
    pipeline = MyPipeline()
    first_output = pipeline(numpy.random.rand(5, 3))
    print('Pipeline output:', first_output)

    # Tasks are filled with processors during initialization of the Pipeline class; keyword arguments do not have to be in order, and if not supplied,
    # the default value will be used
    custom_pipeline = MyPipeline(

        # By setting the ``bind_method`` parameter to ``False``, the function is not bound to the class and we do not need to a ``self`` argument
        pre_pre_process=FunctionProcessor(processing_function=lambda x: x + 1, bind_method=False),

        # The ``pre_process`` task is set to a custom function, which is not of type ``Distance`` and is therefore automatically converted to a
        # ``FunctionProcessor``
        pre_process=lambda x: x.mean(1),

        # The ``pairwise_distance`` task is omitted and therefore defaults to the squared euclidean distance; the ``affinity`` task is set to a
        # ``RadialBasisFunction`` with a lower sigma value
        affinity=RadialBasisFunction(sigma=0.1),

        # The empty ``post_process`` task is set to an instance of our custom processor ``MyProcessor`` and the ``multiplier`` parameter is set to 3
        post_process=MyProcessor(multiplier=3)
    )
    second_output = custom_pipeline(numpy.ones((5, 3, 5)))
    print('Custom pipeline output:', second_output)


if __name__ == '__main__':
    main()
