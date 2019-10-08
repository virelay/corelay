"""Persistent, non-cryptographic hashing of python objects.

Note
----
See https://github.com/chr5tphr/funcache/blob/master/funcache/hashing.py

"""
import pickle

import numpy as np
from numpy import ndarray
# pylint: disable=no-name-in-module
from metrohash import MetroHash128
try:
    from torch import Tensor
except ImportError:
    class Tensor:
        """Dummy Tensor"""


class Hasher(MetroHash128):
    """Hasher object with a write function for file-like updates"""
    def write(self, data):
        """Update using write for file-like behaviour"""
        self.update(data)
        return len(data)


class HashPickler(pickle.Pickler):
    """Pickler for computing Hashes"""
    @staticmethod
    def numpy_id(obj):
        """Persistent id for numpy arrays"""
        mantissa, exponent = np.frexp(obj)
        np.around(mantissa, decimals=2, out=mantissa)
        return (
            obj.dtype.name,
            obj.shape,
            bytes(mantissa),
            bytes(exponent),
        )

    def persistent_id(self, obj):
        """Persistent ids for persistent pickles"""
        if isinstance(obj, ndarray):
            return self.numpy_id(obj)
        if isinstance(obj, Tensor):
            return self.numpy_id(obj.numpy())
        return None


def ext_hash(data):
    """Extended non-cryptographic Hashing using Pickle and MetroHash"""
    hasher = Hasher()
    HashPickler(hasher).dump(data)
    return hasher.hexdigest()
