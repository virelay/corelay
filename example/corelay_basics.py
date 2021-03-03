import numpy as np

from corelay.pipeline.base import Pipeline, Task
from corelay.processor.base import Processor, Param, FunctionProcessor
from corelay.processor.affinity import Affinity, RadialBasisFunction
from corelay.processor.distance import Distance, SciPyPDist
from types import FunctionType


# Processors
class MyProcess(Processor):
    # Parameters are registered by defining a class attribute of type Param, and will be set in __init__ automatically,
    # which expects keyword arguments with the same name the first value is a type specification, the second a default
    # value
    stuff = Param(dtype=int, default=2)
    # as class methods have to be bound explicitly, func here acts like a static function of MyProcess to bind it, see
    # :obj:`corelay.processor.base.FunctionProcessor`
    func = Param(FunctionType, lambda x: x**2)

    # Parameters can be accessed as self.<parameter-name>
    def function(self, data):
        return self.stuff * self.func(data) + 3


# Pipelines
class MyPipeline(Pipeline):
    # Task are registered in order by creating a class attribute of type Task() and, like params, are expected to be
    # supplied with the same name in __init__ as a keyword argument. The first value is an optional expected Process
    # type, second is a default value, which has to be an instance of that type. If the default argument is not a
    # Process, it will be converted to a FunctionProcessor by default, functions fed to FunctionProcessors are by
    # default not bound to the class. To bind them, we can suppy `bind_method=True` to the FunctionProcessor. Supplying
    # it to the task changes the default value of the Processor before creation:
    prepreprocess = Task(proc_type=FunctionProcessor, default=(lambda self, x: x * 2), bind_method=True)
    # Otherwise, we do not need to supply `self` for the default function:
    preprocess = Task(proc_type=FunctionProcessor, default=(lambda x: x**2))
    pdistance = Task(Distance, SciPyPDist(metric='sqeuclidean'))
    affinity = Task(Affinity, RadialBasisFunction(sigma=1.0))
    # empty task, does nothing (except return input) by default
    postprocess = Task()


def main():
    # Use Pipeline 'as is'
    pipeline = MyPipeline()
    output1 = pipeline(np.random.rand(5, 3))
    print('Pipeline output:', output1)

    # Tasks are filled with Processes during initialization of the Pipeline class
    # keyword arguments do not have to be in order, and if not supplied, the default value will be used
    custom_pipeline = MyPipeline(
        # The pipeline's Task sets the `bind_method` Parameter's default to True. Supplying a value here avoids falling
        # back to the default value, and thus we do not need a `self` argument for our function:
        prepreprocess=FunctionProcessor(function=(lambda x: x + 1), bind_method=False),
        preprocess=(lambda x: x.mean(1)),
        affinity=RadialBasisFunction(sigma=.1),
        postprocess=MyProcess(stuff=3)
    )
    output2 = custom_pipeline(np.ones((5, 3, 5)))
    print('Custom pipeline output:', output2)


if __name__ == '__main__':
    main()
