"""A module that contains classes to read and write different file formats like HDF5."""

import copy
import json
import pickle
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import KeysView
from os import PathLike
from types import TracebackType
from typing import Annotated, Any, IO, Literal, NamedTuple, Protocol, runtime_checkable

import h5py
import numpy
from numpy.typing import NDArray

from corelay.base import Param
from corelay.io.hashing import ext_hash
from corelay.plugboard import Plugboard


@runtime_checkable
class Storable(Protocol):
    """An abstract class that defines the interface for storable objects, i.e., objects that have a ``read`` and ``write`` method."""

    def read(self, data_in: Any, meta: Any) -> Any:
        """Retrieves the output data that was produced by the specified input data if it is available. The meta data can contain additional
        identifying information about the data.

        Args:
            data_in (Any): The input data to retrieve the output data for.
            meta (Any): The meta data to retrieve the output data for, which can contain additional identifying information about the data.

        Returns:
            Any: The output data that was produced by the specified input data if it is available.
        """

    def write(self, data_out: Any, data_in: Any, meta: Any) -> None:
        """Writes the specified output data to the storage container. The meta data that can be used to store additional identifying information about
        the data.

        Args:
            data_out (Any): The output data to write.
            data_in (Any): The input data that produced the output data.
            meta (Any): The meta data that can be used to store additional identifying information about the data.
        """


class NoDataSource(Exception):
    """An exception, which is raised when no data source available."""

    def __init__(self, message: str = 'No Data Source available.') -> None:
        """Initializes a new ``NoDataSource`` instance.

        Args:
            message (str): The error message to be displayed. Defaults to 'No Data Source available.'.
        """

        super().__init__(message)


class NoDataTarget(Exception):
    """An exception, which is raised when no target source available."""

    def __init__(self) -> None:
        """Initializes a new ``NoDataTarget`` instance."""

        super().__init__('No Data Target available.')


RecursiveNDArrayTuple = tuple['NDArray[Any] | RecursiveNDArrayTuple', ...]
"""A recursive tuple of NumPy arrays, i.e., a tuple that contains NumPy arrays or other tuples of NumPy arrays, which themselves can contain other
tuples of NumPy arrays, and so on. This is used to represent a nested structure of NumPy arrays.
"""


RecursiveHashTuple = tuple['str | RecursiveHashTuple', ...]
"""A recursive tuple of strings, i.e., a tuple that contains strings or other tuples of strings, which themselves can contain other tuples of strings,
and so on. This is used to represent a nested structure of hashes for the data that is stored in a ``RecursiveNDArrayTuple``.
"""


class HashedHDF5:
    """A storage container, which can be used to store ``Processor`` data in HDF5 files. A hash of the input data that produced the stored data is
    stored alongside the data, so that the data can later be retrieved based on the input data.
    """

    base: h5py.Group
    """The HDF5 group to store the data in."""

    def __init__(self, h5group: h5py.Group) -> None:
        """Initializes a new ``HashedHDF5`` instance.

        Args:
            h5group (h5py.Group): The HDF5 group to store the data in.
        """

        self.base = h5group

    def read(self, data_in: Any, meta: Any) -> Any:
        """Retrieves the output data that was produced by the specified input data if it is available. The hash is computed from the input data and
        the meta data. The meta data can contain additional identifying information about the data.

        Args:
            data_in (Any): The input data to retrieve the output data for.
            meta (Any): The meta data to retrieve the output data for, which can contain additional identifying information about the data.

        Raises:
            NoDataSource: The data source is not available.

        Returns:
            Any: The output data that was produced by the specified input data if it is available.
        """

        hash_value = ext_hash((data_in, meta))
        try:
            group = self.base[hash_value]
        except KeyError as exception:
            raise NoDataSource() from exception

        return HashedHDF5._read_hdf5_content_recursively(group['data'])

    def write(self, data_out: Any, data_in: Any, meta: Any) -> None:
        """Writes the specified output data to a hashed HDF5 group. The hash is computed from the input data and the meta data. The meta data that can
        be used to store additional identifying information about the data.

        Args:
            data_out (Any): The output data to write.
            data_in (Any): The input data that produced the output data. Is used to compute the hash.
            meta (Any): The meta data that can be used to store additional identifying information about the data. Is used to compute the hash.

        Raises:
            TypeError: The data type of the input data is not supported.
        """

        hash_value = ext_hash((data_in, meta))
        group = self.base.require_group(hash_value)

        try:
            HashedHDF5._write_hdf5_recursively(data_out, group, 'data')
        except TypeError as exception:
            raise TypeError(
                f'The data type of the output data "{type(data_out)}" is not supported. The must either be a NumPy array or a hierarchy of tuples '
                'containing NumPy arrays and tuples of NumPy arrays.'
            ) from exception

        group['meta'] = json.dumps(meta)
        group['input'] = json.dumps(HashedHDF5._hash_data_recursively(data_in))
        group['output'] = json.dumps(HashedHDF5._hash_data_recursively(data_out))

    @staticmethod
    def _read_hdf5_content_recursively(base: h5py.Group | h5py.Dataset) -> NDArray[Any] | RecursiveNDArrayTuple:
        """Recursively goes through the hierarchy of HDF5 groups and converts the HDF5 datasets (which are the leaf nodes in this hierarchy) into
        a hierarchy of nested tuples. The keys of the groups are sorted to ensure that the order of the data is consistent.

        Args:
            base (h5py.Group | h5py.Dataset): The HDF5 group or dataset to read.

        Raises:
            TypeError: The data type of the input data is not supported.

        Returns:
            NDArray[Any] | RecursiveNDArrayTuple: Returns a tuple of NumPy arrays or other tuples of NumPy arrays, which themselves can be nested.
                Each tuple represents an HDF5 group. Nested tuples appear in the alphabetical order of the keys of the corresponding HDF5 groups.
                The NumPy arrays contain the data that was read from the HDF5 datasets contained as leaf nodes in the HDF5 groups. If the
                specified ``base`` is a dataset, the data is directly returned as a NumPy array.
        """

        if isinstance(base, h5py.Group):
            return tuple(HashedHDF5._read_hdf5_content_recursively(base[key]) for key in sorted(base))
        if isinstance(base, h5py.Dataset):
            dataset_data: NDArray[Any] = base[()]
            return dataset_data
        raise TypeError('Unsupported output type!')

    @staticmethod
    def _write_hdf5_recursively(data: RecursiveNDArrayTuple | NDArray[Any], group: h5py.Group, key: str) -> None:
        """Recursively writes the specified data to an HDF5 group from a tuple hierarchy of NumPy arrays. The keys of the group are generated from
        the indices of the content in the tuples. The NumPy arrays are stored as HDF5 datasets.

        Args:
            data (RecursiveNDArrayTuple | NDArray[Any]): The data to write to the HDF5 group, which can either be a tuple containing NumPy arrays or
                other tuples containing NumPy arrays or tuples of NumPy arrays, or a NumPy array.
            group (h5py.Group): The HDF5 group to write the data to.
            key (str): The key of the HDF5 group to write the data to. If the data is a tuple, the key is used to create a new group in the HDF5
                parent ``group``. The children of the tuple will then be written recursively to the newly created group and their key will be
                generated using the index of the child inside of the tuple. If the data is a NumPy array, the key is used to create a new dataset
                in the HDF5 parent ``group``.

        Raises:
            TypeError: The data type of the input data is not supported.
        """

        if isinstance(data, tuple):
            new_group = group.require_group(key)
            for index, array in enumerate(data):
                HashedHDF5._write_hdf5_recursively(array, new_group, f'{index:03d}')
        elif isinstance(data, numpy.ndarray):
            group[key] = data
        else:
            raise TypeError(
                f'The data type of the output data "{type(data)}" is not supported. The must either be a NumPy array or a hierarchy of tuples '
                'containing NumPy arrays and tuples of NumPy arrays.'
            )

    @staticmethod
    def _hash_data_recursively(data: tuple[Any, ...] | Any) -> str | RecursiveHashTuple:
        """Hashes the specified data recursively. If the data is a tuple, then the elements of the tuple are hashed recursively and the resulting
        hashes are returned as a tuple. For all other data types, the hash is computed directly and returned.

        Args:
            data (tuple[Any, ...] | Any): The data to hash. This can either be a tuple containing multiple elements, which themselves can be
                tuples containing the data in a hierarchy, or it can be a single element.

        Returns:
            str | RecursiveHashTuple: The hash of the data. If the data is a tuple, then the hashes of the elements are returned as a tuple. For
                all other data types, the hash is computed directly and returned as a string.
        """

        if isinstance(data, tuple):
            return tuple(HashedHDF5._hash_data_recursively(element) for element in data)
        return ext_hash(data)


class DataStorageBase(ABC, Plugboard):
    """The abstract base class for key-value stores."""

    io: Any
    """The storage object to read and write data to. Defaults to `None`."""

    def __init__(self, **kwargs: Any) -> None:
        """Initializes a new ``DataStorageBase`` instance.

        Args:
            **kwargs (Any): Keyword arguments that are passed to the constructor of the class one step up in the class hierarchy, i.e., ``Plugboard``.
        """

        super().__init__(**kwargs)

        self.io: Any = None

    @abstractmethod
    def read(self, data_in: Any = None, meta: Any = None) -> Any:
        """Reads the output data that was produced by the specified input data, if it is available. The meta data can contain additional identifying
        information about the data.

        Args:
            data_in (Any, optional): Input data that produces the data that is to be read. Defaults to `None`.
            meta (Any, optional): Meta data that contains additional identifying information about the data that is to be read. Defaults to `None`.

        Raises:
            NoDataSource: The data source is not available.

        Returns:
            Any: Returns the data that was produced by the specified input data if it is available.
        """

    @abstractmethod
    def write(self, data_out: Any, data_in: Any = None, meta: Any = None) -> None:
        """Writes the specified output data to the storage. The hash is computed from the input data and the meta data. The meta data can be used to
        store additional identifying information about the data.

        Args:
            data_out (Any): The output data to write.
            data_in (Any, optional): The input data that produced the output data. Defaults to `None`.
            meta (Any, optional): The meta data that can be used to store additional identifying information about the data. Defaults to `None`.
        """

    @abstractmethod
    def exists(self) -> bool:
        """Checks if the data if data exists.

        Returns:
            bool: Returns `True` if the data exists and `False` otherwise.
        """

    @abstractmethod
    def keys(self) -> KeysView[str]:
        """Retrieves the keys of the data stored in the storage container.

        Returns:
            KeysView[str]: Returns a list of keys of the io file object.
        """

    def __enter__(self) -> 'DataStorageBase':
        """Opens the IO object and returns the instance. This is used to implement the context manager protocol, which allows the use of the `with`
        statement to automatically close the IO object when it is no longer needed. This is useful for ensuring that the IO object is properly closed
        and resources are released when the context manager exits.

        Returns:
            DataStorageBase: Returns this instance of the ``DataStorageBase`` class.
        """

        return self

    def __exit__(self, exception_type: type[Exception] | None, exception: Exception, traceback: TracebackType | None) -> None:
        """Closes the IO object. This is used to implement the context manager protocol, which allows the use of the `with` statement to automatically
        close the IO object when it is no longer needed. This is useful for ensuring that the IO object is properly closed and resources are released
        when the context manager exits.

        Args:
            exception_type (type[Exception] | None): When the context manager exits due to an exception, this is the type of the exception that was
                raised, otherwise it is ``None``.
            exception (Exception): When the context manager exits due to an exception, this is the exception that was raised, otherwise it is
                ``None``.
            traceback (TracebackType | None): When the context manager exits due to an exception, this is the traceback of the exception that was
                raised, otherwise it is ``None``.
        """

        if self.io is not None:
            self.io.close()

    def __contains__(self, key: str) -> bool:
        """Check if the key exists in the storage.

        Args:
            key (str): The key to check for existence.

        Raises:
            TypeError: The key is not a string.

        Returns:
            bool: Returns 'True' if the key exists in the storage and `False` otherwise.
        """

        if not isinstance(key, str):
            raise TypeError(f'The specified key "{key}" is not a string.')

        return self.at(data_key=key).exists()

    def __getitem__(self, key: str) -> Any:
        """Get the data for a given key.

        Args:
            key (str): The key to get the data for.

        Raises:
            TypeError: The key is not a string.

        Returns:
            Any: Returns the data for the given key.
        """

        if not isinstance(key, str):
            raise TypeError(f'The specified key "{key}" is not a string.')

        return self.at(data_key=key).read()

    def __setitem__(self, key: str, value: Any) -> None:
        """Set the data for a given key.

        Args:
            key (str): The key to set the data for.
            value (Any): The data to set for the given key.

        Raises:
            TypeError: The key is not a string.
        """

        if not isinstance(key, str):
            raise TypeError(f'The specified key "{key}" is not a string.')

        return self.at(data_key=key).write(value)

    def __bool__(self) -> bool:
        """Converts the data storage object to a boolean value. This is used to determine if the data storage object is actually backed by a store.

        Returns:
            bool: Returns `True` if the data storage object is backed by a store and `False` otherwise.
        """

        return bool(self.io)

    def close(self) -> None:
        """Close opened io file object."""

        if self.io is not None:
            self.io.close()

    def at(self, **kwargs: Any) -> 'DataStorageBase':
        """Returns a copy of the instance where the keyword arguments were added as attributes of the class become the attributes of the class.

        Args:
            **kwargs (Any): The keyword arguments, which are added as attributes of the class.

        Raises:
            TypeError: One or more of the names in the keyword arguments are not valid attribute names.

        Returns:
            DataStorageBase: Returns a copy of the instance where the keyword arguments were added as attributes of the class become the attributes
                of the class. This allows to create a new instance of the class with new or updated attributes without modifying the original
                instance.
        """

        result = copy.copy(self)
        for key, value in kwargs.items():
            try:
                setattr(result.default, key, value)
            except AttributeError as exception:
                raise TypeError(f"'{key}' is an invalid keyword argument for '{type(self).__name__}.at'.") from exception
        return result


class NoStorage(DataStorageBase):
    """A placeholder data storage class, which does not actually use persistent storage and raises exceptions when trying to read from it or write to
    it.
    """

    def __bool__(self) -> bool:
        """Converts the data storage object to a boolean value. This is used to determine if the data storage object is actually backed by a store.

        Returns:
            bool: Returns `False` since this is a placeholder data storage class and does not actually use persistent storage.
        """

        return False

    def read(self, data_in: Any = None, meta: Any = None) -> Any:  # pylint: disable=unused-argument
        """Reads the output data that was produced by the specified input data, if it is available. The meta data can contain additional identifying
        information about the data.

        Args:
            data_in (Any, optional): Input data that produces the data that is to be read. Defaults to `None`.
            meta (Any, optional): Meta data that contains additional identifying information about the data that is to be read. Defaults to `None`.

        Raises:
            NoDataSource: This is a placeholder data storage class and does not actually use persistent storage and therefore always raises this
                exception.

        Returns:
            Any: The data that was produced by the specified input data if it is available.
        """

        raise NoDataSource()

    def write(self, data_out: Any, data_in: Any = None, meta: Any = None) -> None:  # pylint: disable=unused-argument
        """Writes the specified output data to the storage. The meta data can be used to store additional identifying information about the data.

        Args:
            data_out (Any): The output data to write.
            data_in (Any, optional): The input data that produced the output data. Defaults to `None`.
            meta (Any, optional): The meta data that can be used to store additional identifying information about the data. Defaults to `None`.

        Raises:
            NoDataTarget: This is a placeholder data storage class and does not actually use persistent storage and therefore always raises this
                exception.
        """

        raise NoDataTarget()

    def exists(self) -> bool:
        """Returns True if data exists.

        Raises:
            NoDataSource: This is a placeholder data storage class and does not actually use persistent storage and therefore always raises this
                exception.

        Returns:
            bool: Returns `False` since this is a placeholder data storage class and does not actually use persistent storage.
        """

        raise NoDataSource()

    def keys(self) -> KeysView[str]:
        """Retrieves the keys of the data stored in the storage container.

        Raises:
            NoDataSource: This is a placeholder data storage class and does not actually use persistent storage and therefore always raises this
                exception.

        Returns:
            KeysView[str]: Returns never, since this is a placeholder data storage class that does not actually use persistent storage and raises an
                exception.
        """

        raise NoDataSource()


FileOpenMode = Literal['w', 'r', 'a']
"""The file open mode to use when opening a file. The options are:

- "w": Write mode. The file is created if it does not exist and existing files will be overwritten.
- "r": Read mode. The file must already exist and the data is read from the file.
- "a": Append mode. The file is created if it does not exist and the data is appended to the end of the file if the file already exists.
"""


class PickleStorage(DataStorageBase):
    """Experimental pickle storage that uses the ``pickle`` module to store data."""

    io: IO[Any]
    """The file object to read data from and write data to. This is a binary file object that is used to store the pickled data."""

    data: dict[str, Any]
    """A dictionary that stores the data that is read from or written to the file. The keys of the dictionary are the keys of the data that is stored
    in the file, and the values are the data that is stored in the file. The dictionary is used to cache the data that is read from the file, so
    that it does not need to be read from the file again if it is already cached.
    """

    data_key: Annotated[str, Param(str, 'data', mandatory=True)]
    """The key of the data that is read from the pickle file or written to the pickle file."""

    def __init__(self, path: str | PathLike[str], mode: FileOpenMode = 'r', data_key: str | None = None, **kwargs: Any) -> None:
        """
        Initializes a new ``PickleStorage`` instance.

        Args:
            path (str | PathLike[str]): The path to the pickle file where the data is to read from or written to.
            mode (FileOpenMode, optional): The mode in which the file is opened. This can be either "w" for write mode, "r" for read mode or "a" for
                append mode. In write mode, the file is created if it does not exist and the existing file is overwritten. In read mode, the file must
                already exist and the data is read from the file. In append mode, the file is created if it does not exist and the data is appended to
                the end of the file. Defaults to "r".
            data_key (str | None): The key of the data that is read from the pickle file or written to the pickle file. Defaults to `None`.
            **kwargs (Any): Keyword arguments that are passed to the constructor of the class one step up in the class hierarchy, i.e.,
                ``DataStorageBase``.

        Raises:
            ValueError: The mode is not "w", "r", or "a".
        """

        # PyDocLint does not support the documentation of the constructor parameters both in the __init__ method and the class docstring, so we have
        # to add the documentation for the data_key parameter here; therefore, the data_key parameter has to be added to the keyword arguments for the
        # base class manually
        if data_key is not None:
            kwargs['data_key'] = data_key
        super().__init__(**kwargs)

        if mode not in ['w', 'r', 'a']:
            raise ValueError('Mode should be set to "w", "r", or "a".')

        self.io: IO[Any] = open(path, mode + 'b')  # pylint: disable=unspecified-encoding, consider-using-with
        self.data: dict[str, Any] = {}

    def _load_data(self) -> None:
        """Loads the data from the pickle file into the data dictionary. This is done by reading the file until the end of the file is reached and
        unpickling each object. This method must be called before the data can be accessed, because, unlike in some other ways of storing the data,
        pickle files are not random access. After the pickle file has be loaded, the data is cached in memory.
        """

        try:
            while True:
                loaded_data = pickle.load(self.io)
                self.data.update({loaded_data['key']: loaded_data['data']})
        except EOFError:
            pass

    def read(self, data_in: Any = None, meta: Any = None) -> Any:  # pylint: disable=unused-argument
        """Retrieves the data for a given data key.

        Args:
            data_in (Any, optional): Input data that produced the data that is to be read. Defaults to `None`.
            meta (Any, optional): Meta data that contains additional identifying information about the data that is to be read. Defaults to `None`.

        Raises:
            NoDataSource: The data source for the given data key does not exist.

        Returns:
            Any: Returns the data for the given data key.
        """

        # The exists method will call keys which in turn will call _load_data, this way, the data will be lazy-loaded
        if not self.exists():
            raise NoDataSource(f"Key: '{self.data_key}' does not exist.")
        return self.data[self.data_key]

    def write(self, data_out: Any, data_in: Any = None, meta: Any = None) -> None:
        """Writes the specified output data to the pickle file using the given data key as: `{'data': data_out, 'key': self.data_key}`.

        Args:
            data_out (Any): The data to write to the pickle file.
            data_in (Any, optional): The input data that produced the output data. Defaults to `None`.
            meta (Any, optional): The meta data that can be used to store additional identifying information about the data. Defaults to `None`.
        """

        self.data[self.data_key] = data_out
        pickle.dump({"data": data_out, "key": self.data_key}, self.io)

    def exists(self) -> bool:
        """Determines if the data key exists in the data.

        Returns:
            bool: Returns `True` if the data key exists and `False` otherwise.
        """

        # The keys will call _load_data this way, the data will be lazy-loaded
        return self.data_key in self.keys()

    def keys(self) -> KeysView[str]:
        """Retrieves the keys of the data stored in the pickle file.

        Returns:
            KeysView[str]: Returns a view of keys of the data that is stored in the file.
        """

        if not self.data:
            self._load_data()
        return self.data.keys()


class StringInfo(NamedTuple):
    """A type for the type information that the ``h5py.check_string_dtype`` function returns. This class is, unfortunately, not exported by the
    ``h5py`` module, so we have to define it ourselves to gain type safety.
    """

    encoding: str
    """The encoding of the string, e.g., "utf-8" or "ascii"."""

    length: int
    """The length of the string."""


class HDF5Storage(DataStorageBase):
    """A storage that used HDF5 files to store data."""

    io: h5py.File
    """The HDF5 file object to read data from and write data to."""

    data_key: Annotated[str, Param(str, mandatory=True)]
    """The key of the data that is read from the HDF5 file or written to the HDF5 file."""

    def __init__(self, path: str | PathLike[str], mode: FileOpenMode = 'r', data_key: str | None = None, **kwargs: Any) -> None:
        """Initializes a new ``HDF5Storage`` instance.

        Args:
            path (str | PathLike[str]): The path to the HDF5 file where the data is to read from or written to.
            mode (FileOpenMode, optional): The mode to open the HDF5 file in. This can be either "w" for write mode, "r" for read mode or "a" for
                append mode. In write mode, the file is created if it does not exist and existing files will be overwritten. In read mode, the file
                must already exist and the data is read from the file. In append mode, the file is created if it does not exist and the data is
                appended to the end of the file if the file already exists. Defaults to "r".
            data_key (str | None): The key of the data that is read from the HDF5 file or written to the HDF5 file. Defaults to `None`.
            **kwargs (Any): Keyword arguments that are passed to the constructor of the class one step up in the class hierarchy, i.e.,
                ``DataStorageBase``.
        """

        # PyDocLint does not support the documentation of the constructor parameters both in the __init__ method and the class docstring, so we have
        # to add the documentation for the data_key parameter here; therefore, the data_key parameter has to be added to the keyword arguments for the
        # base class manually
        if data_key is not None:
            kwargs['data_key'] = data_key
        super().__init__(**kwargs)

        self.io: h5py.File = h5py.File(path, mode=mode)

    def read(self, data_in: Any = None, meta: Any = None) -> Any:  # pylint: disable=unused-argument
        """Retrieves the data for a given data key.

        Args:
            data_in (Any, optional): Input data that produced the data that is to be read. Defaults to `None`.
            meta (Any, optional): Meta data that contains additional identifying information about the data that is to be read. Defaults to `None`.

        Raises:
            NoDataSource: The data source for the given data key does not exist.

        Returns:
            Any: Returns the data for the given data key.
        """

        if not self.exists():
            raise NoDataSource(f"Key: '{self.data_key}' does not exist.")
        _, data = self._unpack('/', self.io[self.data_key])
        return data

    def write(self, data_out: dict[str, Any] | tuple[Any, ...] | Any, data_in: Any = None, meta: Any = None) -> None:
        """Writes the specified output data to the HDF5 file. If the output data is a dictionary, then the output data is stored in an HDF5 group with
        the name given by the data key. The key-value pairs of the dictionary will be stored in this HDF5 group with the keys of the dictionary used
        as the names of the datasets and the values of the dictionary used as the data for the datasets. If the output data is a tuple, then the
        output data is stored in an HDF5 group with the name given by the data key. The values of the tuple will be stored as datasets in this HDF5
        group, with the indices of the tuple used as the names of the datasets and the values of the tuple used as the data for the datasets. If the
        output data is neither a dictionary nor a tuple, then the output data is stored in an HDF5 dataset with the name given by the data key and the
        output data used as the data for the dataset.

        Args:
            data_out (dict[str, Any] | tuple[Any, ...] | Any): The data to write to the HDF5 file. This can either be a dataset, a tuple, or any value
                that can be written to an HDF5 file (i.e., basic data types like ``int``, ``float``, ``bool``, or ``str``, or a NumPy array). If the
                data is a dictionary, then it will be stored as an HDF5 group with the name given by the data key. The key-value pairs of the
                dictionary will be stored in this HDF5 group with the keys of the dictionary used as the names of the datasets and the values of the
                dictionary used as the data for the datasets. If the data is a tuple, then it will be stored as an HDF5 group with the name given by
                the data key. The values of the tuple will be stored as datasets in this HDF5 group, with the indices of the tuple used as the names
                of the datasets and the values of the tuple used as the data for the datasets. If the data is neither a dictionary nor a tuple, then
                it will be stored in an HDF5 dataset with the name given by the data key and the data used as the data for the dataset.
            data_in (Any, optional): The input data that produced the output data. Defaults to `None`.
            meta (Any, optional): The meta data that can be used to store additional identifying information about the data. Defaults to `None`.
        """

        if isinstance(data_out, dict):
            for key, value in data_out.items():
                shape, dtype = self._get_shape_and_dtype(value)
                self.io.require_dataset(name=f'{self.data_key}/{key}', data=value, shape=shape, dtype=dtype)
        elif isinstance(data_out, tuple):
            for key, value in enumerate(data_out):
                shape, dtype = self._get_shape_and_dtype(value)
                self.io.require_dataset(name=f'{self.data_key}/{key}', data=value, shape=shape, dtype=dtype)
        else:
            shape, dtype = self._get_shape_and_dtype(data_out)
            self.io.require_dataset(name=self.data_key, data=data_out, shape=shape, dtype=dtype)

    def exists(self) -> bool:
        """Checks if the data key exists in the HDF5 file.

        Returns:
            bool: Returns `True` if the data key exists and `False` otherwise.
        """

        return self.data_key in self.io

    def keys(self) -> KeysView[str]:
        """Retrieves the keys of the data stored in the HDF5 file.

        Returns:
            KeysView[str]: Returns a view of keys of the data in the HDF5 file.
        """

        key_list: KeysView[str] = self.io.keys()
        return key_list

    @staticmethod
    def _unpack(key: str | int, value: h5py.Dataset | h5py.Group | Any) -> tuple[str | int, Any]:
        """Unpacks the specified value. If the value is an HDF5 dataset, it is converted to a NumPy array or a string, depending on the data type of
        the dataset. If the value is a group, then it is recursively unpacked into an ordered dictionary, which contains the keys and values of the
        group. The keys of the group are sorted to ensure that the order of the data is consistent. If the key is a string and only contains numeric
        characters, it is converted to an integer. This is done to allow the use of the dictionary as a tuple or list.

        Args:
            key (str | int): The key of the value to unpack. This can either be a string or an integer. If the key is a string and only contains
                numeric characters, it is converted to an integer.
            value (h5py.Dataset | h5py.Group | Any): The value that is to be unpacked. If the value is an HDF5 dataset, it is converted to a NumPy
                array or a string, depending on the data type of the dataset. If the value is a group, then it is recursively unpacked into an ordered
                dictionary, which contains the keys and values of the group. The keys of the group are sorted to ensure that the order of the data is
                consistent.

        Returns:
            tuple[str | int, Any]: Returns the unpacked key and value as a tuple. The key is either a string or an integer. The value is either a
                NumPy array, a string, or an ordered dictionary, which contains the keys and values of the group. The keys of the group are sorted to
                ensure that the order of the data is consistent.
        """

        # Converts the key to an integer if it is a string and only contains numeric characters
        if isinstance(key, str) and key.isdigit():
            key = int(key)

        # If the value is a dataset, it is converted to a NumPy array; if the data type of the dataset is a string, then it is decoded to a string
        # using the encoding that was used to create the dataset
        if isinstance(value, h5py.Dataset):
            string_info: StringInfo | None = h5py.check_string_dtype(value.dtype)
            value = value[()]
            if string_info is not None:
                value = value.decode(string_info.encoding)

        # If the value is a group, it is converted to an ordered dictionary, which contains the keys and values of the group; since the numeric keys
        # are converted to an integer, the dictionary can be accessed like a tuple or list
        elif isinstance(value, h5py.Group):
            value = OrderedDict((HDF5Storage._unpack(k, v) for k, v in value.items()))

        # Returns the key and value as a tuple
        return key, value

    @staticmethod
    def _get_shape_and_dtype(value: NDArray[Any] | str | int | float) -> tuple[tuple[int, ...], numpy.dtype[Any]]:
        """Infers the shape and data type of the specified value.

        Args:
            value (NDArray[Any] | str | int | float): The value for which to infer the shape and data type. This can be a NumPy array, a string, an
                integer, or a float.

        Returns:
            tuple[tuple[int, ...], numpy.dtype[Any]]: Returns a tuple containing the shape and data type of the specified value. The shape is a tuple
                of integers representing the dimensions of the value. If the value is a NumPy array, the shape is the shape of the array. If the value
                is of any other data type, the shape will be an empty tuple.
        """

        if isinstance(value, numpy.ndarray):
            return value.shape, value.dtype
        if isinstance(value, str):
            return (), h5py.string_dtype(encoding='utf-8')
        return (), numpy.dtype(type(value))
