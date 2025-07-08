===========
Basic Usage
===========

The primary goal of CoRelAy is to facilitate the efficient and streamlined creation of data analysis pipelines. At its core, a **pipeline** (:py:class:`~corelay.pipeline.base.Pipeline`) consists of multiple, modular components called **tasks** (:py:class:`~corelay.pipeline.base.Task`), which are executed in sequence to achieve the desired processing outcome.

Each task's operation is defined by an associated **processor** (:py:class:`~corelay.processor.base.Processor`), which encapsulates the specific processing logic required for that step. Tasks provide default functionality that can be easily customized by replacing their corresponding processors with alternative implementations.

Processors in CoRelAy are highly configurable entities, allowing users to tailor their behavior using **parameters** (:py:class:`~corelay.base.Param`), which dictate the specific processing actions taken by each processor.

The following sections provide an overview of these key concepts and demonstrate how to leverage them when working with CoRelAy.

Processors & Parameters
=======================

In CoRelAy, a **Processor** is defined by sub-classing :py:class:`~corelay.processor.base.Processor`. The functionality of a processor is defined by overriding the :py:meth:`Processor.function <corelay.processor.base.Processor.function>` method, which is called with a single positional argument, the data to be processed, and returns the processed data.

**Parameters** can be registered with the processor, by defining a class attribute of type :py:class:`~typing.Annotated`, where the first argument is the data type of the parameter, and the second is an instance of :py:class:`~corelay.base.Param`. The first argument to :py:meth:`Param.__init__ <corelay.base.Param.__init__>` is the runtime data type of the parameter (which may be different from the type hint used as the first argument to :py:class:`~typing.Annotated`, e.g., the type hint may be a generic type like ``dict[int]``, while the runtime type must be a concrete type like :py:class:`dict`, i.e., the same type that would be returned by :py:class:`type`). The second argument is the default value of the parameter.

.. note::

    If you come from a version of CoRelAy before 1.0.0, you may be used to the old syntax of registering parameters by assigning an instance of :py:class:`~corelay.base.Param` to a class attribute. For more information on this change and how to migrate, please refer to the :doc:`migration guide <../migration-guide/migrating-from-v0.2-to-v1.0>`.

The processor will automatically track the parameters and allows users to set them in the constructor using the attribute's name as a keyword argument. Parameters can, however, also be made into positional arguments by setting the ``is_positional`` argument of :py:meth:`Param.__init__ <corelay.base.Param.__init__>` to :py:obj:`True`. This allows for a more flexible and user-friendly interface when creating custom processors. The parameters can be accessed as attributes of the processor instance. Invoking a processor  to perform the associated operation is as easy as calling it like a function.

The following example demonstrates how to create a custom processor by subclassing :py:class:`~corelay.processor.base.Processor` and how to define parameters using :py:class:`~corelay.base.Param`:

.. code-block:: python

    from types import FunctionType
    from typing import Annotated, Any

    import numpy

    from corelay.base import Param
    from corelay.processor.base import Processor


    class MyProcessor(Processor):
        """A custom CoRelAy processor, which applies a configurable function to its input data and multiplies it by a configurable value."""

        multiplier: Annotated[int, Param(dtype=int, default=2)]
        """An :py:class:`int` parameter, which is multiplied with the result of the function."""

        function_to_apply: Annotated[FunctionType, Param(FunctionType, lambda x: x**2)]
        """A function, which is applied to the input data."""

        def function(self, data: Any) -> Any:
            """Applies the custom function :py:attr:`function_to_apply` to the input data and multiplies it by the parameter :py:attr:`multiplier`.

            Args:
                data (Any): The input data that is to be processed.

            Returns:
                Any: Returns the processed data.
            """

            # Parameters can be accessed as self.<parameter-name>
            return self.multiplier * self.function_to_apply(data)

.. note::

    Please note, that in the above example, the type of the ``function_to_apply`` parameter is :py:class:`~types.FunctionType`. Unfortunately, Python does not have a unified type for functions. Instead, functions, lambda functions, methods, built-in functions, built-in methods, and other functions like NumPy array and universal functions are all represented by different types. CoRelAy is smart enough to recognize this and will allow you to pass any kind of function or method to a parameter of type :py:class:`~types.FunctionType`.

Pipelines & Tasks
=================

**Pipelines** represent entire data processing workflows. They consist of multiple, sequential, pre-determined steps, called **tasks**. Every pipeline is a sub-class of :py:class:`~corelay.pipeline.base.Pipeline`. Tasks are registered by creating a class attribute of type :py:class:`~typing.Annotated`, with the first argument being the type of the processor that is expected to be used in the task, and the second being an instance of :py:class:`~corelay.pipeline.base.Task`. The first argument to :py:meth:`Task.__init__ <corelay.pipeline.base.Task.__init__>` is the type of the processor that is expected to be used in the task, and the second argument is the default processor that is used by the task, if the user does not specify a custom processor. Like parameters, the processors of the tasks can be supplied to the :py:meth:`Pipeline.__init__ <corelay.processor.base.Processor.__init__>` method as keyword arguments with the same name as the corresponding attribute. All additional keyword arguments that are passed to the :py:class:`~corelay.pipeline.base.Task` are assigned to the parameters of the processor. Like processors, pipelines can be executed by simply calling it like a function.

The following example demonstrates how to create a custom pipeline by subclassing :py:class:`~corelay.pipeline.base.Pipeline` and how to define tasks using :py:class:`~corelay.pipeline.base.Task`:

.. code-block:: python

    from typing import Annotated, Any

    from corelay.pipeline.base import Pipeline, Task
    from corelay.processor.base import FunctionProcessor, Processor
    from corelay.processor.affinity import Affinity, RadialBasisFunction
    from corelay.processor.distance import Distance, SciPyPDist


    class MyPipeline(Pipeline):
        """A custom CoRelAy pipeline, which applies a series of processors to its input data."""

        pre_pre_process: Annotated[FunctionProcessor, Task(proc_type=FunctionProcessor, default=lambda self, x: x * 2, bind_method=True)]
        """A pre-pre-processing task, which applies a function to the input data. By default, the input data is multiplied by 2."""

        pre_process: Annotated[FunctionProcessor, Task(proc_type=FunctionProcessor, default=lambda x: x**2)]
        """A pre-processing task, which applies a function to the input data. By default, the input data is squared."""

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

The :py:class:`~corelay.processor.base.FunctionProcessor` class is a :py:class:`~corelay.processor.base.Processor` that applies a customizable function to the input data. In essence it can be used to turn any Python function into a processor. If the value or default value of a task is a function, it will be automatically converted to a :py:class:`~corelay.processor.base.FunctionProcessor` (this is irrespective of the task's processor type; if the type is neither :py:class:`~corelay.processor.base.Processor` nor :py:class:`~corelay.processor.base.FunctionProcessor`, the task would still convert a function to a :py:class:`~corelay.processor.base.FunctionProcessor`, which will lead to an error as the task verifies that the processor type and the processor/default processor are consistent). This is why we can also just supply a lambda expression as the default value of the task.

By default, functions fed to :py:class:`~corelay.processor.base.FunctionProcessor` are not bound to the class. To bind them, we can supply `bind_method=True` to the :py:class:`~corelay.processor.base.FunctionProcessor`. Please note how the ``bind_method`` parameter of :py:class:`~corelay.processor.base.FunctionProcessor` is omitted in the ``pre_process`` task and therefore defaults to :py:obj:`False`. This means that the function is not bound to the class and does not have access to ``self``.

Pipelines and processors can be instantiated and used in the following way:

.. code-block:: python

    import numpy

    from corelay.processor.base import FunctionProcessor
    from corelay.processor.affinity import RadialBasisFunction

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

Memoization
===========

CoRelAy provides a built-in memoization mechanism that allows you to cache the results of expensive computations and reuse them when the same inputs are encountered again. This can significantly speed up your data processing pipelines, especially when dealing with large datasets or complex calculations. When adding a storage container to a pipeline, intermediate results are automatically cached and will be reused both during the pipeline execution and when the pipeline is called again with the same input data and parameters, as the intermediate results are stored on disk. To enable memoization, you need to add a storage container to your pipeline. The following example demonstrates how to do this:

.. code-block:: python

    import time

    import h5py
    import numpy

    from corelay.io.storage import HashedHDF5

    # Opens an HDF5 file in append mode for the storing the results of the analysis and the memoization of intermediate pipeline results
    with h5py.File('test.analysis.h5', 'a') as analysis_file:

        # Creates a HashedHDF5 IO object, which is storage container that stores outputs of processors based on hashes in an HDF5 file
        io_object = HashedHDF5(analysis_file.require_group('proc_data'))

        # Creates a new pipeline with the storage container as the IO object
        pipeline = MyPipeline(io=io_object)

        # Runs the pipeline and measures the execution time
        start_time = time.perf_counter()
        output = pipeline(numpy.ones((1000, 1000)))
        duration = time.perf_counter() - start_time

        # Since we memoize our results in an HDF5 file, subsequent calls will not compute the values (for the same inputs), but rather load them
        # from the HDF5 file; try running the script multiple times
        print(f'Pipeline output: {output}')
        print(f'Pipeline execution time: {duration:.4f} seconds')

Running the example should yield faster execution times on subsequent runs, as the intermediate results are cached in the HDF5 file. The first run will take longer, as the pipeline has to compute all intermediate results and store them in the HDF5 file. Subsequent runs will load the intermediate results from the HDF5 file, which is much faster. The difference in execution time in this example is, of course, miniscule, as the pipeline is very simple and the data is small. However, in real-world applications, the difference can be significant.

Fleshed out versions of the above examples and more examples to highlight the features of CoRelAy can be found in :repo:`docs/examples/`.
