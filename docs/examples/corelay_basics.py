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
    """An :py:class:`int` parameter, which is multiplied with the result of the function.

    Note:
        Parameters can be registered with the processor, by defining a class attribute of type :py:class:`Annotated`, where the first argument is the
        data type of the parameter, and the second is an instance of :py:class:`~corelay.base.Param`. The first argument to
        :py:meth:`Param.__init__ <corelay.base.Param.__init__>` is the runtime data type of the parameter (which may be different from the type hint
        used as the first argument to :py:class:`Annotated`, e.g., the type hint may be a generic type like ``dict[int]``, while the runtime type must
        be a concrete type like :py:class:`dict`, i.e., the same type that would be returned by :py:class:`type`). The second argument is the default
        value of the parameter.
    """

    function_to_apply: Annotated[FunctionType, Param(FunctionType, lambda x: x**2)]
    """A function, which is applied to the input data.

    Note:
        As class methods have to be bound explicitly, :py:attr:`function_to_apply` here acts like a static function of :py:class:`MyProcessor`. If you
        want to have processor that applies a custom function and that is bound to the class and has access to self, please refer to
        :py:class:`~corelay.processor.base.FunctionProcessor`.
    """

    def function(self, data: Any) -> Any:
        """Applies the custom function :py:attr:`function_to_apply` to the input data and multiplies it by the parameter :py:attr:`multiplier`.

        Args:
            data (Any): The input data that is to be processed.

        Returns:
            Any: Returns the processed data.
        """

        # Parameters can be accessed as self.<parameter-name>
        return self.multiplier * self.function_to_apply(data)


class MyPipeline(Pipeline):
    """A custom CoRelAy pipeline, which applies a series of processors to its input data."""

    pre_pre_process: Annotated[FunctionProcessor, Task(proc_type=FunctionProcessor, default=lambda self, x: x * 2, bind_method=True)]
    """A pre-pre-processing task, which applies a function to the input data. By default, the input data is multiplied by 2.

    Note:
        Tasks are registered by creating a class attribute of type :py:class:`Annotated`, with the first argument being the type of the processor that
        is expected to be used in the task, and the second being an instance of :py:class:`~corelay.pipeline.base.Task`. The first argument to
        :py:meth:`Task.__init__ <corelay.pipeline.base.Task.__init__>` is the type of the processor that is expected to be used in the task, and
        the second argument is the default processor that is used by the task, if the user does not specify a custom processor. This can also be a
        function, which will be converted to a :py:class:`~corelay.processor.base.FunctionProcessor`. Like parameters, the processors of the tasks can
        be supplied to the :py:class:`Pipeline.__init__ <corelay.processor.base.Processor.__init__>` method as keyword arguments with the same name as
        the corresponding attribute. All additional keyword arguments that are passed to the :py:class:`~corelay.pipeline.base.Task` are passed to the
        processor during instantiation.

        The :py:class:`~corelay.processor.base.FunctionProcessor` class is a :py:class:`~corelay.processor.base.Processor` that applies a customizable
        function to the input data. By default, functions fed to :py:class:`~corelay.processor.base.FunctionProcessor` are not bound to the class. To
        bind them, we can supply `bind_method=True` to the :py:class:`~corelay.processor.base.FunctionProcessor`.
    """

    pre_process: Annotated[FunctionProcessor, Task(proc_type=FunctionProcessor, default=lambda x: x**2)]
    """A pre-processing task, which applies a function to the input data. By default, the input data is squared.

    Note:
        The ``bind_method`` parameter of :py:class:`~corelay.processor.base.FunctionProcessor` is omitted here and therefore defaults to
        :py:obj:`False`. This means that the function is not bound to the class and does not have access to `self`.
    """

    pairwise_distance: Annotated[Distance, Task(Distance, SciPyPDist(metric='sqeuclidean'))]
    """A task, which applies a pairwise distance function to the input data. By default, the squared euclidean distance is used. The
    :py:class:`~corelay.processor.distance.Distance` class is a base class for all distance processors.
    """

    affinity: Annotated[Affinity, Task(Affinity, RadialBasisFunction(sigma=1.0))]
    """A task, which applies an affinity function to the input data. By default, the radial basis function is used. The
    :py:class:`~corelay.processor.distance.Affinity` class is a base class for all affinity processors.
    """

    post_process: Annotated[Processor, Task()]
    """A post-processing task, which does nothing by default and returns the input data as is."""


def main() -> None:
    """The entrypoint to the :py:mod:`corelay_basics` script."""

    # Creates a new pipeline without specifying any parameters, which means that the default values of the tasks will be used
    pipeline = MyPipeline()
    first_output = pipeline(numpy.random.rand(5, 3))
    print('Pipeline output:', first_output)

    # Tasks are filled with processors during initialization of the Pipeline class; keyword arguments do not have to be in order, and if not supplied,
    # the default value will be used
    custom_pipeline = MyPipeline(

        # By setting the bind_method parameter to False, the function is not bound to the class and we do not need to a self argument
        pre_pre_process=FunctionProcessor(processing_function=lambda x: x + 1, bind_method=False),

        # The pre_process task is set to a custom function, which is not of type Distance and is therefore automatically converted to a
        # FunctionProcessor
        pre_process=lambda x: x.mean(1),

        # The pairwise_distance task is omitted and therefore defaults to the squared euclidean distance; the affinity task is set to a
        # RadialBasisFunction with a lower sigma value
        affinity=RadialBasisFunction(sigma=0.1),

        # The empty post_process task is set to an instance of our custom processor MyProcessor and the multiplier parameter is set to 3
        post_process=MyProcessor(multiplier=3)
    )
    second_output = custom_pipeline(numpy.ones((5, 3, 5)))
    print('Custom pipeline output:', second_output)


if __name__ == '__main__':
    main()
