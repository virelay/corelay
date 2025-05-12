"""A module that contains non-cryptographic hashing functionality for Python objects. These are used to compute hashes of the inputs of operations
performed by instances of :py:class:`~corelay.processor.base.Processor` to identify them in a way that is independent of their memory address
and can be used to identify data between subsequent runs of the same :py:class:`~corelay.pipeline.base.Pipeline`.

Note:
    Please refer to the `Funcache Project <https://github.com/chr5tphr/funcache/blob/master/funcache/hashing.py>`_ to see the original implementation
    of this module.
"""

import importlib
import pickle
import typing
from types import ModuleType
from typing import Protocol, runtime_checkable

import numpy
from metrohash import MetroHash128  # pylint: disable = no-name-in-module
from numpy import ndarray


@runtime_checkable
class SupportsConversionToNumPyArray(Protocol):
    """A protocol that defines an interface for objects that can be converted to a :py:class:`~numpy.ndarray`."""

    def numpy(self) -> numpy.ndarray[typing.Any, typing.Any]:
        """Converts the object to a :py:class:`~numpy.ndarray`.

        Returns:
            numpy.ndarray[typing.Any, typing.Any]: Returns a :py:class:`~numpy.ndarray` representation of the object.
        """


class TensorPlaceholder:
    """A placeholder class to stand in for PyTorch's :py:class:`~torch.Tensor` class in case PyTorch is not installed."""

    def numpy(self) -> numpy.ndarray[typing.Any, typing.Any]:
        """Converts the :py:class:`~torch.Tensor` to a :py:class:`~numpy.ndarray`.

        Raises:
            NotImplementedError: This method should not be called, as this is a placeholder class.

        Returns:
            numpy.ndarray[typing.Any, typing.Any]: Returns a :py:class:`~numpy.ndarray` representation of the :py:class:`~torch.Tensor`.
        """

        raise NotImplementedError('This method should not be called, as this is a placeholder class.')


Tensor: type[SupportsConversionToNumPyArray]
"""Either the PyTorch :py:class:`~torch.Tensor` class or the placeholder :py:class:`TensorPlaceholder` class if PyTorch is not installed.

Note:
    This is used to check if an object that is to be pickled is a :py:class:`~torch.Tensor` or not, because PyTorch :py:class:`~torch.Tensor` objects
    are converted to :py:class:`~numpy.ndarray` before pickling.
"""


# Tries to import PyTorch and assign the Tensor class to the variable Tensor, if PyTorch is not installed, then the Tensor class will be replaced with
# a placeholder class
try:
    torch: ModuleType = importlib.import_module('torch')
    Tensor = torch.Tensor
except ImportError:
    Tensor = TensorPlaceholder


class Hasher(MetroHash128):
    """Hasher object with a write function for file-like updates"""

    def write(self, data: bytes) -> int:
        """Updates the hash, by adding the specified data to the end of the input.

        Note:
            This method was made to give the ``MetroHash128`` object a file-like interface.

        Args:
            data (bytes): The data to add to the hash. This can be any :py:class:`bytes`-like object.

        Returns:
            int: Returns the number of bytes added to the hash.
        """

        self.update(data)
        return len(data)


class HashPickler(pickle.Pickler):
    """A pickler for computing hashes."""

    @staticmethod
    def numpy_id(array: numpy.ndarray[typing.Any, typing.Any]) -> tuple[str, tuple[int, ...], bytes, bytes]:
        """Computes a unique ID for a :py:class:`~numpy.ndarray`, which consists of the data type name, the array's shape, and the values of the array
        decomposed into their respective mantissas and exponents as a :py:class:`bytes` sequence.

        Args:
            array (numpy.ndarray[typing.Any, typing.Any]): The :py:class:`~numpy.ndarray` to compute the ID for.

        Returns:
            tuple[str, tuple[int, ...], bytes, bytes]: Returns a tuple containing the data type name, the array's shape, and the values of the array
            decomposed into their respective mantissas and exponents as a :py:class:`bytes` sequence.
        """

        mantissa, exponent = numpy.frexp(array)
        numpy.around(mantissa, decimals=2, out=mantissa)
        return (
            array.dtype.name,
            array.shape,
            bytes(mantissa),
            bytes(exponent),
        )

    def persistent_id(self, obj: typing.Any) -> tuple[str, tuple[int, ...], bytes, bytes] | None:
        """Computes a persistent ID for an object that is to be pickled, which can be used by the :py:mod:`pickle` module to identify two objects as
        "the same" during the un-pickling process. The persistent ID is used to identify the object in a way that is independent of its memory
        address. This is useful for caching and serialization purposes.

        Args:
            obj (typing.Any): The object to compute the persistent ID for.

        Returns:
            tuple[str, tuple[int, ...], bytes, bytes] | None: Returns a persistent ID for the object. If the object is a :py:class:`~numpy.ndarray`,
            it returns a tuple containing the data type name, the array's shape, and the values of the array decomposed into their respective
            mantissas and exponents as a :py:class:`bytes` sequence. If the object is a :py:class:`~torch.Tensor`, it converts the tensor to a
            :py:class:`~numpy.ndarray` and computes a unique ID for the array. If the object is neither, it returns :py:obj:`None`.
        """

        if isinstance(obj, ndarray):
            return self.numpy_id(obj)
        if isinstance(obj, Tensor):
            return self.numpy_id(obj.numpy())
        return None


def ext_hash(data: typing.Any) -> str:
    """Hashes the specified data. It uses an extended, non-cryptographic hashing algorithm, which first pickles the specified object and then hashes
    the resulting :py:class:`bytes` sequence using MetroHash.

    Args:
        data (typing.Any): The data to hash. This can be any Python :py:class:`object`, including :py:class:`~numpy.ndarray` and
            :py:class:`~torch.Tensor`.

    Returns:
        str: Returns the hash of the data as a hexadecimal :py:class:`str`.
    """

    hasher = Hasher()
    HashPickler(hasher).dump(data)
    hash_value: str = hasher.hexdigest()
    return hash_value
