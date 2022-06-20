"""'io module contains classes to load and dump different files like hdf5, etc.

"""

import copy
import pickle
import json
from collections import OrderedDict
from abc import abstractmethod

import numpy as np
import h5py

from ..base import Param
from ..plugboard import Plugboard
from .hashing import ext_hash


class StorableMeta(type):
    """Meta class to check for write/ read attributes via isinstance"""
    def __instancecheck__(cls, instance):
        """Is instance if object has attributes write and read"""
        return all(hasattr(instance, attr) for attr in ('write', 'read'))


class Storable(metaclass=StorableMeta):
    """Abstract class to check for write/ read attributes via isinstance"""


class NoDataSource(Exception):
    """Raise when no data source available."""
    # Following is not useless, since message becomes optional
    # pylint: disable=useless-super-delegation
    def __init__(self, message='No Data Source available.'):
        super().__init__(message)


class NoDataTarget(Exception):
    """Raise when no target source available."""
    def __init__(self):
        super().__init__('No Data Target available.')


class HashedHDF5:
    """Hashed storage of Processor data in HDF5 files"""
    def __init__(self, h5group):
        self.base = h5group

    def read(self, data_in, meta):
        """Read output from a hashed h5 group, with hash of (data_in, meta)"""
        def _iterread(base):
            """Iteratively read from HDF5 Group into tuple hierachy of ndarrays"""
            if isinstance(base, h5py.Group):
                return tuple(_iterread(base[key]) for key in sorted(base))
            if isinstance(base, h5py.Dataset):
                return base[()]
            raise TypeError('Unsupported output type!')
        hashval = ext_hash((data_in, meta))
        try:
            group = self.base[hashval]
        except KeyError as error:
            raise NoDataSource() from error

        return _iterread(group['data'])

    def write(self, data_out, data_in, meta):
        """Write output to a hashed h5 group, with hash of (data_in, meta)"""
        def _iterwrite(data, group, elem):
            """Iteratively write to a HDF5 Group from a tuple hierachy of ndarrays"""
            if isinstance(data, tuple):
                g_new = group.require_group(elem)
                for n, array in enumerate(data_out):
                    _iterwrite(array, g_new, f'{n:03d}')
            elif isinstance(data, np.ndarray):
                group[elem] = data
            else:
                raise TypeError('Unsupported output type!')

        def _iterhash(base):
            """Iteratively hash from tuple hierachy into tuple hierachy of hashes"""
            if isinstance(base, tuple):
                return tuple(_iterhash(obj) for obj in base)
            return ext_hash(base)

        hashval = ext_hash((data_in, meta))
        group = self.base.require_group(hashval)
        _iterwrite(data_out, group, 'data')
        group['meta'] = json.dumps(meta)
        group['input'] = json.dumps(_iterhash(data_in))
        group['output'] = json.dumps(_iterhash(data_out))


class DataStorageBase(Plugboard):
    """Implements a key, value storage object.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.io = None

    @abstractmethod
    def read(self, data_in=None, meta=None):
        """Should implement read functionality.

        """

    @abstractmethod
    def write(self, data_out, data_in=None, meta=None):
        """Should implement write functionality.

        """

    def close(self):
        """Close opened io file object.

        """
        self.io.close()

    @abstractmethod
    def exists(self):
        """Return True if data exists.

        """

    @abstractmethod
    def keys(self):
        """Return keys of the io file object.

        """

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        self.io.close()

    def __contains__(self, key):
        return self.at(data_key=key).exists()

    def __getitem__(self, key):
        return self.at(data_key=key).read()

    def __setitem__(self, key, value):
        return self.at(data_key=key).write(value)

    def __bool__(self):
        return bool(self.io)

    def at(self, **kwargs):
        """Return a copy of the instance where kwargs become the attributes of the class.
        I.e. a specific self.data_key is set so that self.write(data) automatically writes the data to correct key.

        """
        result = copy.copy(self)
        for key, value in kwargs.items():
            try:
                setattr(result.default, key, value)
            except AttributeError as err:
                raise TypeError(
                    f"'{key}' is an invalid keyword argument for '{type(self).__name__}.at'."
                ) from err
        return result


class NoStorage(DataStorageBase):
    """Stub class when no Storage is used."""
    def __bool__(self):
        return False

    def read(self, data_in=None, meta=None):
        raise NoDataSource()

    def write(self, data_out, data_in=None, meta=None):
        raise NoDataTarget()

    def exists(self):
        raise NoDataSource()

    def keys(self):
        raise NoDataSource()


class PickleStorage(DataStorageBase):
    """Experimental pickle storage that uses pickle to store data.

    """

    data_key = Param(str, 'data', mandatory=True)

    def __init__(self, path, mode='r', **kwargs):
        """
        Parameters
        ----------
        path: str
            Path to the pickled file.
        mode: str
            Write, Read or Append mode ['w', 'r', 'a'].

        """
        super().__init__(**kwargs)
        if mode not in ['w', 'r', 'a']:
            raise ValueError("Mode should be set to 'w', 'r' or 'a'.")
        self.io = open(path, mode + 'b')  # pylint: disable=consider-using-with
        self.data = {}

    def _load_data(self):
        try:
            while True:
                dc = pickle.load(self.io)
                self.data.update({dc['key']: dc['data']})
        except EOFError:
            pass

    def read(self, data_in=None, meta=None):
        """Return data for a given key. Need to load the complete pickle at first read. After the data is cached.

        Returns
        -------
        data for a given key

        """
        if not self.exists():
            raise NoDataSource(f"Key: '{self.data_key}' does not exist.")
        return self.data[self.data_key]

    def write(self, data_out, data_in=None, meta=None):
        """Write and pickle the data as: {"data": data, "key": key}

        Parameters
        ----------
        data_out: np.ndarray, dict
            Data being stored.

        """
        self.data[self.data_key] = data_out
        pickle.dump({"data": data_out, "key": self.data_key}, self.io)

    def keys(self):
        """Return keys from self.data. Need to load the complete pickle at first read.

        """
        if not self.data:
            self._load_data()
        return self.data.keys()

    def exists(self):
        """Return True if key exists in self.keys().

        """
        return self.data_key in self.keys()


class HDF5Storage(DataStorageBase):
    """HDF5 storage that stores data under different keys.

    """
    data_key = Param(str, 'data', mandatory=True)

    def __init__(self, path, mode='r', **kwargs):
        """
        Parameters
        ----------
        path: str
            Path to the hdf5 file.
        mode: str
            Write, Read or Append mode ['w', 'r', 'a'].

        """
        super().__init__(**kwargs)
        self.io = h5py.File(path, mode=mode)

    def read(self, data_in=None, meta=None):
        """
        Returns
        -------
        data for a given key

        """
        if not self.exists():
            raise NoDataSource(f"Key: '{self.data_key}' does not exist.")
        _, data = self._unpack('/', self.io[self.data_key])
        return data

    def write(self, data_out, data_in=None, meta=None):
        """
        Parameters
        ----------
        data_out: np.ndarray, dict
            Data being stored. Dictionaries are pickled and stored as strings.

        """
        if isinstance(data_out, dict):
            for key, value in data_out.items():
                shape, dtype = self._get_shape_dtype(value)
                self.io.require_dataset(data=value, shape=shape, dtype=dtype, name=f'{self.data_key}/{key}')
        elif isinstance(data_out, tuple):
            for key, value in enumerate(data_out):
                shape, dtype = self._get_shape_dtype(value)
                self.io.require_dataset(data=value, shape=shape, dtype=dtype, name=f'{self.data_key}/{key}')
        else:
            shape, dtype = self._get_shape_dtype(data_out)
            self.io.require_dataset(data=data_out, shape=shape, dtype=dtype, name=self.data_key)

    def exists(self):
        """Returns True if key exists in self.io.

        """
        return self.data_key in self.io

    def keys(self):
        """Return keys of the storage.

        """
        return self.io.keys()

    @staticmethod
    def _unpack(key, value):
        if key.isdigit():
            key = int(key)
        if isinstance(value, h5py.Dataset):
            check = h5py.check_string_dtype(value.dtype)
            value = value[()]
            if check is not None:
                value = value.decode(check.encoding)
        elif isinstance(value, h5py.Group):
            # Change key to integer if k is digit, so that we can use the dict like a tuple or list
            value = OrderedDict((HDF5Storage._unpack(k, v) for k, v in value.items()))
        return key, value

    @staticmethod
    def _get_shape_dtype(value):
        """Infer shape and dtype of given element v.

        Parameters
        ----------
        value: np.ndarray, str, int, float
            Element for which we want to infer the shape and dtype.

        Returns
        -------
        shape, dtype: tuple, type
            Return the shape and dtype of v that works with h5py.require_dataset

        """
        if isinstance(value, np.ndarray):
            return value.shape, value.dtype
        if isinstance(value, str):
            return (), h5py.string_dtype(encoding='utf-8')
        return (), np.dtype(type(value))
