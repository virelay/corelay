"""A module that contains non-cryptographic hashing functionality for Python objects.

Note:
    See https://github.com/chr5tphr/funcache/blob/master/funcache/hashing.py to see the original implementation of this module.
"""

import importlib
import pickle
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

import numpy
from metrohash import MetroHash128  # pylint: disable = no-name-in-module
from numpy import ndarray
from numpy.typing import NDArray


@runtime_checkable
class SupportsConversionToNumPyArray(Protocol):
    """A protocol that defines an interface for objects that can be converted to a NumPy array."""

    def numpy(self) -> NDArray[Any]:
        """Converts the object to a NumPy array.

        Returns:
            NDArray[Any]: Returns a NumPy array representation of the object.
        """


class TensorPlaceholder:
    """A placeholder class to stand in for PyTorch's ``Tensor`` class in case PyTorch is not installed."""

    def numpy(self) -> NDArray[Any]:
        """Converts the PyTorch tensor to a NumPy array.

        Raises:
            NotImplementedError: This method should not be called, as this is a placeholder class.

        Returns:
            NDArray[Any]: Returns a NumPy array representation of the PyTorch tensor.
        """

        raise NotImplementedError('This method should not be called, as this is a placeholder class.')


Tensor: type[SupportsConversionToNumPyArray]
"""Either the PyTorch ``Tensor`` class or a placeholder class if PyTorch is not installed.

Note:
    This is used to check if an object that is to be pickled is a PyTorch tensor or not, because PyTorch ``Tensor`` objects are converted to NumPy
    arrays before pickling.
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
            This method was made to give the ``Hasher`` object a file-like interface.

        Args:
            data (bytes): The data to add to the hash. This can be any bytes-like object.

        Returns:
            int: Returns the number of bytes added to the hash.
        """

        self.update(data)
        return len(data)


class HashPickler(pickle.Pickler):
    """A pickler for computing hashes."""

    @staticmethod
    def numpy_id(array: NDArray[Any]) -> tuple[str, tuple[int, ...], bytes, bytes]:
        """Computes a unique ID for NumPy arrays, which consists of the data type name, the array's shape, and the values of the array decomposed into
        their respective mantissas and exponents as a ``bytes`` sequence.

        Args:
            array (NDArray[Any]): The NumPy array to compute the ID for.

        Returns:
            tuple[str, tuple[int, ...], bytes, bytes]: A tuple containing the data type name, the array's shape, and the values of the array
                decomposed into their respective mantissas and exponents as a ``bytes`` sequence.
        """

        mantissa, exponent = numpy.frexp(array)
        numpy.around(mantissa, decimals=2, out=mantissa)
        return (
            array.dtype.name,
            array.shape,
            bytes(mantissa),
            bytes(exponent),
        )

    def persistent_id(self, obj: Any) -> tuple[str, tuple[int, ...], bytes, bytes] | None:
        """Computes a persistent ID for an object that is to be pickled, which can be used by the ``pickle`` module to identify two objects as "the
        same" during the un-pickling process. The persistent ID is used to identify the object in a way that is independent of its memory address.
        This is useful for caching and serialization purposes.

        Args:
            obj (Any): The object to compute the persistent ID for.

        Returns:
            tuple[str, tuple[int, ...], bytes, bytes] | None: Returns a persistent ID for the object. If the object is a NumPy array, it returns a
                tuple containing the data type name, the array's shape, and the values of the array decomposed into their respective mantissas and
                exponents as a ``bytes`` sequence. If the object is a PyTorch tensor, it converts the tensor to a NumPy array and computes a unique ID
                for the array. If the object is neither, it returns ``None``.
        """

        if isinstance(obj, ndarray):
            return self.numpy_id(obj)
        if isinstance(obj, Tensor):
            return self.numpy_id(obj.numpy())
        return None


def ext_hash(data: Any) -> str:
    """Hashes the specified data. It uses an extended, non-cryptographic hashing algorithm, which first pickles the specified object and then hashes
    the resulting ``bytes`` sequence using MetroHash.

    Args:
        data (Any): The data to hash. This can be any Python object, including NumPy arrays and PyTorch tensors.

    Returns:
        str: Returns the hash of the data as a hexadecimal string.
    """

    hasher = Hasher()
    HashPickler(hasher).dump(data)
    hash_value: str = hasher.hexdigest()
    return hash_value
