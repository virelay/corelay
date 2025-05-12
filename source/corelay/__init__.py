"""CoRelAy is a package to compose small-scale (single-machine) analysis pipelines. Pipelines are designed with a number of steps
(:py:class:`~corelay.pipeline.base.Task`) with default operations (:py:class:`~corelay.processor.base.Processor`). Any step of the pipeline may then
be individually changed by assigning a new operator (:py:class:`~corelay.processor.base.Processor`). Processors have parameters
(:py:class:`~corelay.base.Param`) which define their operation.
"""
