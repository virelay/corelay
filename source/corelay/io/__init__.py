"""A sub-package containing IO-related modules for ``Processor`` data."""

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
