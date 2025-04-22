================
 Getting started
================

CoRelAy is a tool to compose small-scale (single-machine) analysis pipelines. It
was created to swiftly implement pipelines to generate analysis data which can
then be visualized using ViRelAy.

Install
-------

CoRelAy can be installed directly from PyPI:

.. code-block:: console

   $ pip install virelay

To install optional HDBSCAN and UMAP support, use

.. code-block:: console

    $ pip install corelay[umap,hdbscan]

For the current development version, or to try out examples, clone and install
with:

.. code-block:: console

   $ git clone https://github.com/virelay/virelay.git
   $ pip install ./virelay

Basic Usage
-----------

The main objective of ViRelAy is the quick and hassle-free composition of
analysis **Pipelines**. **Pipelines** are designed with a number of steps,
called **Task**, which each have their own default function, defined using a
**Processor**. Any step of the pipeline may be individually changed by assigning
a new **Processor**. **Processors** have **Params** which configure their
functional hyperparameters.


Processor and Param
^^^^^^^^^^^^^^^^^^^

A **Processor** can be defined in the following way:

.. code-block:: python

    from types import FunctionType
    from typing import Annotated

    from corelay.processor.base import Processor, Param


    class MyProcess(Processor):
        # Parameters are registered by defining a class attribute of type Param,
        # and will be set in __init__ automatically, which expects keyword
        # arguments with the same name the first value is a type specification,
        # the second a default value
        stuff: Annotated[int, Param(dtype=int, default=2)]

        # as class methods have to be bound explicitly, func here acts like a
        # static function of MyProcess. For more information see
        # corelay.processor.base.FunctionProcessor
        func: Annotated[FunctionType, Param(FunctionType, lambda x: x**2)]

        # Parameters can be accessed as self.<parameter-name>
        def function(self, data: Any) -> Any:
            return self.stuff * self.func(data) + 3

Every **Processor** is a sub-class of
:py:class:`~corelay.processor.base.Processor`, and must implement
:py:meth:`~corelay.processor.base.Processor.function`, which typically only uses
a single positional argument.
Parameters for the Processor can be specified by assigning an instance of
:py:class:`~corelay.processor.base.Param` as a class attribute.
The name of the attribute can be used as a keyword argument to specify the value
when creating an instance of the **Processor**, and accessed under the same name.
Each **Param** has a datatype ``dtype`` and a default value ``default``.
**Processor** instances can be used like functions, or assigned to a **Task** of
a **Pipeline**.

Pipeline and Task
^^^^^^^^^^^^^^^^^

**Pipelines** consist of multiple, sequential, pre-determined steps, called
**Tasks**, and can be defined in the following way:

.. code-block:: python

    from corelay.pipeline.base import Pipeline, Task
    from corelay.processor.base import FunctionProcessor
    from corelay.processor.affinity import Affinity, RadialBasisFunction
    from corelay.processor.distance import Distance, SciPyPDist


    class MyPipeline(Pipeline):
        # Task are registered in order by creating a class attribute of type
        # Task() and, like params, are expected to be supplied with the same name
        # in __init__ as a keyword argument. The first value is an optional
        # expected Process type, second is a default value, which has to be an
        # instance of that type. If the default argument is not a Process, it will
        # be converted to a FunctionProcessor by default, functions fed to
        # FunctionProcessors are by default not bound to the class. To bind them,
        # we can supply `bind_method=True` to the FunctionProcessor. Supplying it
        # to the task changes the default value of the Processor before creation:
        prepreprocess = Task(
            proc_type=FunctionProcessor,
            default=lambda self, x: x * 2,
            bind_method=True
        )
        # Otherwise, we do not need to supply `self` for the default function:
        preprocess = Task(proc_type=FunctionProcessor, default=lambda x: x**2)
        pdistance = Task(Distance, SciPyPDist(metric='sqeuclidean'))
        affinity = Task(Affinity, RadialBasisFunction(sigma=1.0))
        # empty task, does nothing (except return input) by default
        postprocess = Task()

Every **Pipeline** is a sub-class of :py:class:`~corelay.pipeline.base.Pipeline`.
**Tasks** of a pipeline are created by assigning an instance of
:py:class:`~corelay.pipeline.base.Task` as a class attribute, similar to
**Params** in **Processors**.
Each **Task** has, each optional, a **Processor**-type ``proc_type``, a default
**Processor** for the Task ``default``. Additional keyword arguments can be
specified as default parameter values that should be assigned to any
**Processor** that is used for the **Task**. The keyword argument
``bind_method`` is specific to :py:class:`~corelay.processor.base.FunctionProcessor`,
and describes, whether the function is static (default, ``bind_method=False``),
or whether it should have access to the **Processor** instance.
Functions can be passed instead of **Processors**, which will be implicitly
converted to a :py:class:`~corelay.processor.base.FunctionProcessor`.
**Tasks** can be assigned by passing **Processors** with their respective
keyword argument during instantiation of the **Pipeline**, or by directly
assigning them to the respective attribute.

**Pipelines** and **Processors** can be instantiated and used in the following
way:

.. code-block:: python

    import numpy

    from corelay.processor.base import FunctionProcessor
    from corelay.processor.affinity import RadialBasisFunction
    from types import FunctionType

    # Use Pipeline 'as is'
    pipeline = MyPipeline()
    output1 = pipeline(numpy.random.rand(5, 3))
    print('Pipeline output:', output1)

    # Tasks are filled with Processes during initialization of the Pipeline
    # class keyword arguments do not have to be in order, and if not supplied,
    # the default value will be used
    custom_pipeline = MyPipeline(
        # The pipeline's Task sets the `bind_method` Parameter's default to
        # True. Supplying a value here avoids falling back to the default
        # value, and thus we do not need a `self` argument for our function:
        prepreprocess=FunctionProcessor(
            processing_function=lambda x: x + 1, bind_method=False
        ),
        preprocess=lambda x: x.mean(1),
        postprocess = MyProcess(stuff=3)
    )
    custom_pipeline.affinity = RadialBasisFunction(sigma=0.1),
    output2 = custom_pipeline(numpy.ones((5, 3, 5)))
    print('Custom pipeline output:', output2)

Like **Processors**, executing a **Pipeline** can be done by simply calling it
like a function.

Examples
--------

More examples to highlight some features of **CoRelAy** can be found in
:repo:`example/`. The following demonstrates, how to create a functional
pipeline based on :py:class:`corelay.pipeline.spectral.SpectralClustering`. A
similar version of the following code may be found in
:repo:`example/memoize_spectral_pipeline.py`.

.. code-block:: python

    """An example script, which uses memoization to store (intermediate) results."""

    import time
    from collections.abc import Sequence
    from typing import Annotated, Any, SupportsIndex

    import h5py
    import numpy
    from numpy.typing import NDArray

    from corelay.base import Param
    from corelay.io.storage import HashedHDF5
    from corelay.pipeline.spectral import SpectralClustering
    from corelay.processor.base import Processor
    from corelay.processor.clustering import KMeans
    from corelay.processor.embedding import TSNEEmbedding, EigenDecomposition
    from corelay.processor.flow import Sequential, Parallel


    class Flatten(Processor):
        """Represents a CoRelAy processor, which flattens its input data."""

        def function(self, data: Any) -> Any:
            """Applies the flattening to the input data.

            Args:
                data (Any): The input data that is to be flattened.

            Returns:
                Any: Returns the flattened data.
            """

            input_data: NDArray[Any] = data
            input_data.sum()
            return input_data.reshape(input_data.shape[0], numpy.prod(input_data.shape[1:]))


    class SumChannel(Processor):
        """Represents a CoRelAy processor, which sums its input data across channels, i.e., its second axis."""

        def function(self, data: Any) -> Any:
            """Applies the summation over the channels to the input data.

            Args:
                data (Any): The input data that is to be summed over its channels.

            Returns:
                Any: Returns the data that was summed up over its channels.
            """

            input_data: NDArray[Any] = data
            return input_data.sum(axis=1)


    class Normalize(Processor):
        """Represents a CoRelAy processor, which normalizes its input data."""

        axes: Annotated[SupportsIndex | Sequence[SupportsIndex], Param((SupportsIndex, Sequence), (1, 2))]
        """A parameter of the processor, which determines the axis over which the data is to be normalized. Defaults to the second and third axes."""

        def function(self, data: Any) -> Any:
            """Normalizes the specified input data.

            Args:
                data (Any): The input data that is to be normalized.

            Returns:
                Any: Returns the normalized input data.
            """

            input_data: NDArray[Any] = data
            return input_data / input_data.sum(self.axes, keepdims=True)


    def main() -> None:
        """The entrypoint to the memoize_spectral_pipeline script."""

        # Fixes the random seed for reproducibility
        numpy.random.seed(0xDEADBEEF)

        # Opens an HDF5 file in append mode for the storing the results of the analysis and the memoization of intermediate pipeline results
        with h5py.File('test.analysis.h5', 'a') as analysis_file:

            # Creates a HashedHDF5 IO object, which is an IO object that stores outputs of processors based on hashes in an HDF5 file
            io_object = HashedHDF5(analysis_file.require_group('proc_data'))

            # Generates some exemplary data
            data = numpy.random.normal(size=(64, 3, 32, 32))
            number_of_clusters = range(2, 20)

            # Creates a SpectralClustering pipeline, which is one of the pre-defined built-in pipelines
            pipeline = SpectralClustering(

                # Processors, such as EigenDecomposition, can be assigned to pre-defined tasks
                embedding=EigenDecomposition(n_eigval=8, io=io_object),

                # Flow-based processors, such as Parallel, can combine multiple processors; broadcast=True copies the input as many times as there are
                # processors; broadcast=False instead attempts to match each input to a processor
                clustering=Parallel([
                    Parallel([
                        KMeans(n_clusters=k, io=io_object) for k in number_of_clusters
                    ], broadcast=True),

                    # IO objects will be used during computation when supplied to processors, if a corresponding output value (here identified by hashes)
                    # already exists, the value is not computed again but instead loaded from the IO object
                    TSNEEmbedding(io=io_object)
                ], broadcast=True, is_output=True)
            )

            # Processors (and Params) can be updated by simply assigning corresponding attributes
            pipeline.preprocessing = Sequential([
                SumChannel(),
                Normalize(),
                Flatten()
            ])

            # Processors flagged with "is_output=True" will be accumulated in the output; the output will be a tree of tuples, with the same hierarchy as
            # the pipeline (i.e., _clusterings here contains a tuple of the k-means outputs)
            start_time = time.perf_counter()
            _clusterings, _tsne = pipeline(data)

            # Since we memoize our results in an HDF5 file, subsequent calls will not compute the values (for the same inputs), but rather load them from
            # the HDF5 file; try running the script multiple times
            duration = time.perf_counter() - start_time
            print(f'Pipeline execution time: {duration:.4f} seconds')


    if __name__ == '__main__':
        main()
