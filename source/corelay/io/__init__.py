"""A sub-package containing IO-related modules for storing intermediate results of operations performed by instances of
:py:class:`~corelay.processor.base.Processor`. This can be used in a :py:class:`~corelay.pipeline.base.Pipeline` to prevent the re-computation of
intermediate results needed multiple times, or as a cache for subsequent runs of the same pipeline.
"""

from corelay.io.storage import (
    NoDataSource,
    NoDataTarget,
    DataStorageBase,
    NoStorage,
    PickleStorage,
    HDF5Storage
)

__all__ = [
    'NoDataSource',
    'NoDataTarget',
    'DataStorageBase',
    'NoStorage',
    'PickleStorage',
    'HDF5Storage',
]
