"""A module that contains unit tests for the :py:mod:`corelay.io.storage` module."""

import typing
from io import BytesIO
from pathlib import Path

import h5py
import numpy
import pytest

from corelay import io
from corelay.io.storage import DataStorageBase, HashedHDF5


@pytest.fixture(name='data', scope='module')
def get_data_fixture() -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
    """A fixture that produces random test data with shape (10, 2).

    Returns:
        numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: Returns random test data with shape (10, 2).
    """

    return numpy.random.rand(10, 2)


@pytest.fixture(name='parameter_values', scope='module')
def get_parameter_values_fixture() -> dict[str, int | str]:
    """A fixture that produces a parameter values :py:class:`dict` with `param1=1` and `param2='string'`.

    Returns:
        dict[str, int | str]: Returns a :py:class:`dict` with `param1=1` and `param2='string'`.
    """

    return {'param1': 1, 'param2': 'string'}


class TestHashedHDF5:
    """Contains unit tests for the :py:class:`~corelay.io.storage.HashedHDF5` class."""

    @staticmethod
    def test_write_array() -> None:
        """Tests that writing a :py:class:`~numpy.ndarray` raises no exceptions"""

        with BytesIO() as buffer, h5py.File(buffer, 'w') as hdf5_file:
            group = hdf5_file.require_group('hashed')
            data_out = numpy.random.normal(size=5)
            io_object = HashedHDF5(group)
            io_object.write(data_out=data_out, data_in=1, meta=1)

    @staticmethod
    def test_write_tuple() -> None:
        """Tests that writing a tuple of :py:class:`~numpy.ndarray` raises no exceptions"""

        with BytesIO() as buffer, h5py.File(buffer, 'w') as hdf5_file:
            group = hdf5_file.require_group('hashed')
            data_out = (numpy.random.normal(size=5),) * 2
            io_object = HashedHDF5(group)
            io_object.write(data_out=data_out, data_in=1, meta=1)

    @staticmethod
    def test_write_unsupported() -> None:
        """Tests that writing an unsupported type raises a :py:class:`TypeError`"""

        with BytesIO() as buffer, h5py.File(buffer, 'w') as hdf5_file:
            group = hdf5_file.require_group('hashed')
            data_out = 'adfg'
            io_object = HashedHDF5(group)
            with pytest.raises(TypeError):
                io_object.write(data_out=data_out, data_in=1, meta=1)

    @staticmethod
    def test_read_array() -> None:
        """Tests that reading data that was previously written returns the same data"""

        with BytesIO() as buffer, h5py.File(buffer, 'w') as hdf5_file:
            group = hdf5_file.require_group('hashed')
            data_out = numpy.random.normal(size=5)
            io_object = HashedHDF5(group)
            io_object.write(data_out=data_out, data_in=1, meta=1)
            data_read = io_object.read(data_in=1, meta=1)
            assert (data_out == data_read).all()

    @staticmethod
    def test_read_tuple() -> None:
        """Tests that reading data that was previously written returns the same data"""

        with BytesIO() as buffer, h5py.File(buffer, 'w') as hdf5_file:
            group = hdf5_file.require_group('hashed')
            data_out = (numpy.random.normal(size=5),) * 2
            io_object = HashedHDF5(group)
            io_object.write(data_out=data_out, data_in=1, meta=1)
            data_read = io_object.read(data_in=1, meta=1)
            assert all((out == load).all() for out, load in zip(data_out, data_read))

    @staticmethod
    def test_read_unavailable_data() -> None:
        """Tests that reading data that was not written raises a :py:class:`~corelay.io.storage.NoDataSource` exception"""

        with BytesIO() as buffer, h5py.File(buffer, 'w') as hdf5_file:
            group = hdf5_file.require_group('hashed')
            io_object = HashedHDF5(group)
            with pytest.raises(io.NoDataSource):
                io_object.read(data_in=1, meta=1)


@pytest.mark.parametrize('data_storage_type', [io.HDF5Storage, io.PickleStorage])
def test_data_storage_at_functionality(data_storage_type: type[io.HDF5Storage | io.PickleStorage], tmp_path: Path) -> None:
    """Tests the reading and writing of data from :py:class:`~corelay.io.storage.HDF5Storage` and :py:class:`~corelay.io.storage.PickleStorage` data
    storage containers using the :py:meth:`HDF5Storage.at <corelay.io.storage.HDF5Storage.at>` or
    :py:meth:`PickleStorage.at <corelay.io.storage.PickleStorage.at>` methods.

    Args:
        data_storage_type (type[io.HDF5Storage | io.PickleStorage]): The storage class to be tested.
        tmp_path (Path): A temporary path for testing.
    """

    with data_storage_type(tmp_path / 'test.file', mode='a') as data_storage:

        data_storage_copy = data_storage.at(data_key='param_values')
        assert data_storage_copy.data_key == 'param_values', 'Data key was not changed.'  # type: ignore[attr-defined]

        # Should raise a TypeError because a non-existing key was set
        with pytest.raises(TypeError):
            data_storage.at(data_key='param_values', key='key')

        with pytest.raises(TypeError):
            data_storage.at(data_key=1)

        # Should raise a TypeError, since we try to access a data_key internally without it being ever set
        with pytest.raises(TypeError):
            data_storage.at().exists()

        data_storage.at(data_key='param_values').at(data_key='data')


@pytest.mark.parametrize('data_storage_type', [io.HDF5Storage, io.PickleStorage])
def test_data_storage_io_mode(data_storage_type: type[io.HDF5Storage | io.PickleStorage], tmp_path: Path) -> None:
    """Tests that the IO mode of :py:class:`~corelay.io.storage.HDF5Storage` and :py:class:`~corelay.io.storage.PickleStorage` must be 'r', 'w', or
    'a'.

    Args:
        data_storage_type (type[io.HDF5Storage | io.PickleStorage]): The storage class to be tested.
        tmp_path (Path): A temporary path for testing.
    """

    test_path = tmp_path / 'test.file'
    data_storage: DataStorageBase = data_storage_type(test_path, mode='w')
    assert data_storage
    data_storage.close()

    with pytest.raises(ValueError):
        data_storage = data_storage_type(test_path, mode='x')


@pytest.mark.parametrize('data_storage_type', [io.HDF5Storage, io.PickleStorage])
def test_data_storage_indexer_access(data_storage_type: type[io.HDF5Storage | io.PickleStorage], tmp_path: Path) -> None:
    """Tests that :py:class:`~corelay.io.storage.HDF5Storage` and :py:class:`~corelay.io.storage.PickleStorage` can be accessed using the indexer
    instead of using the :py:meth:`HDF5Storage.at <corelay.io.storage.HDF5Storage.at>` or
    :meth:`PickleStorage.at <corelay.io.storage.PickleStorage.at>` methods.

    Args:
        data_storage_type (type[io.HDF5Storage | io.PickleStorage]): The storage class to be tested.
        tmp_path (Path): A temporary path for testing.
    """

    # Tests writing data using the indexer
    test_path = tmp_path / 'test.file'
    data_storage_writer: DataStorageBase = data_storage_type(test_path, mode='w')
    data_storage_writer['data'] = {'key1': 'value1', 'key2': 'value2'}

    # Tests that writing at an index that is not a string raises a TypeError
    with pytest.raises(TypeError):
        data_storage_writer[123] = 'value'  # type: ignore[index]
    data_storage_writer.close()

    # Tests reading data using the indexer
    data_storage_reader: DataStorageBase = data_storage_type(test_path, mode='r')
    assert 'data' in data_storage_reader
    data_read = data_storage_reader['data']
    assert isinstance(data_read, dict)
    assert data_read['key1'] == 'value1'
    assert data_read['key2'] == 'value2'
    assert len(data_read) == 2

    # Tests that reading from an index that is not a string raises a TypeError
    with pytest.raises(TypeError):
        _ = 123 in data_storage_reader  # type: ignore[operator]
    with pytest.raises(TypeError):
        _ = data_storage_reader[123]  # type: ignore[index]

    # Tests that accessing a non-existing key raises an exception
    with pytest.raises(io.NoDataSource):
        _ = data_storage_reader['non_existing']
    data_storage_reader.close()


def test_hdf5_storage_treats_tuples_like_dictionaries_with_indices_as_keys(tmp_path: Path) -> None:
    """Tests that :py:class:`~corelay.io.storage.HDF5Storage` treats tuples like dictionaries with indices as keys.

    Args:
        tmp_path (Path): A temporary path for testing.
    """

    test_path = tmp_path / 'test.file'
    data_storage_writer: io.HDF5Storage = io.HDF5Storage(test_path, mode='w')
    data_storage_writer['data'] = (numpy.array([1, 2, 3]), numpy.array([4, 5, 6]), numpy.array([7, 8, 9]))
    data_storage_writer.close()

    data_storage_reader: io.HDF5Storage = io.HDF5Storage(test_path, mode='r')
    assert 'data' in data_storage_reader
    numpy.testing.assert_equal(data_storage_reader['data'][0], numpy.array([1, 2, 3]))
    numpy.testing.assert_equal(data_storage_reader['data'][1], numpy.array([4, 5, 6]))
    numpy.testing.assert_equal(data_storage_reader['data'][2], numpy.array([7, 8, 9]))
    data_storage_reader.close()


def test_pickle_storage_can_returns_keys(tmp_path: Path) -> None:
    """Tests that :py:class:`~corelay.io.storage.PickleStorage` can return the keys of the data storage container.

    Args:
        tmp_path (Path): A temporary path for testing.
    """

    test_path = tmp_path / 'test.file'
    with io.PickleStorage(test_path, mode='w') as data_storage_writer:
        data_storage_writer['data'] = {'key1': 'value1', 'key2': 'value2'}
        data_storage_writer['parameter_values'] = {'key3': 3, 'key4': 4}
        data_storage_writer['new_entry'] = {'key5': 5.0, 'key6': 6.0}
        data_storage_writer['test/foo'] = {'key7': True, 'key8': False}
        data_storage_writer['test/bar'] = {'key9': numpy.array([1, 2, 3]), 'key10': numpy.array([4, 5, 6])}

        keys = data_storage_writer.keys()
        assert len(keys) == 5
        assert 'data' in keys
        assert 'parameter_values' in keys
        assert 'new_entry' in keys
        assert 'test/foo' in keys
        assert 'test/bar' in keys


def test_hdf5_storage_can_returns_keys(tmp_path: Path) -> None:
    """Tests that :py:class:`~corelay.io.storage.HDF5Storage` can return the keys of the data storage container.

    Args:
        tmp_path (Path): A temporary path for testing.
    """

    test_path = tmp_path / 'test.file'
    with io.HDF5Storage(test_path, mode='w') as data_storage_writer:
        data_storage_writer['data'] = {'key1': 'value1', 'key2': 'value2'}
        data_storage_writer['parameter_values'] = {'key3': 3, 'key4': 4}
        data_storage_writer['new_entry'] = {'key5': 5.0, 'key6': 6.0}
        data_storage_writer['test/foo'] = {'key7': True, 'key8': False}
        data_storage_writer['test/bar'] = {'key9': numpy.array([1, 2, 3]), 'key10': numpy.array([4, 5, 6])}

        keys = data_storage_writer.keys()
        assert len(keys) == 4
        assert 'data' in keys
        assert 'parameter_values' in keys
        assert 'new_entry' in keys
        assert 'test' in keys


def test_hdf5_storage_unpacks_nested_groups_to_ordered_dictionary(tmp_path: Path) -> None:
    """Tests that :py:class:`~corelay.io.storage.HDF5Storage` unpacks nested groups to an :py:class:`~collections.OrderedDict` when writing data.

    Args:
        tmp_path (Path): A temporary path for testing.
    """

    test_path = tmp_path / 'test.file'
    with io.HDF5Storage(test_path, mode='w') as data_storage_writer:
        data_storage_writer['test/foo'] = {'key1': True, 'key2': False}
        data_storage_writer['test/bar'] = {'key3': numpy.array([1, 2, 3]), 'key4': numpy.array([4, 5, 6])}
        data_storage_writer['test/baz'] = 'This is a string value'

        data_dictionary = data_storage_writer.at(data_key='test').read()
        assert isinstance(data_dictionary, typing.OrderedDict)

        assert 'foo' in data_dictionary
        assert 'key1' in data_dictionary['foo']
        numpy.testing.assert_equal(data_dictionary['foo']['key1'], numpy.True_)
        assert 'key2' in data_dictionary['foo']
        numpy.testing.assert_equal(data_dictionary['foo']['key2'], numpy.False_)

        assert 'bar' in data_dictionary
        assert 'key3' in data_dictionary['bar']
        numpy.testing.assert_equal(data_dictionary['bar']['key3'], numpy.array([1, 2, 3]))
        assert 'key4' in data_dictionary['bar']
        numpy.testing.assert_equal(data_dictionary['bar']['key4'], numpy.array([4, 5, 6]))

        assert 'baz' in data_dictionary
        assert data_dictionary['baz'] == 'This is a string value'


@pytest.mark.parametrize('data_storage_type', [io.HDF5Storage, io.PickleStorage])
def test_data_storage(
    data_storage_type: type[io.HDF5Storage | io.PickleStorage],
    tmp_path: Path,
    data: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]],
    parameter_values: dict[str, int | str]
) -> None:
    """Tests the reading and writing of data to and from :py:class:`~corelay.io.storage.HDF5Storage` and :py:class:`~corelay.io.storage.PickleStorage`
    data storage containers.

    Args:
        data_storage_type (type[io.HDF5Storage | io.PickleStorage]): The storage class to be tested.
        tmp_path (Path): A temporary path for testing.
        data (numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]): Random test data with shape (10, 2).
        parameter_values (dict[str, int | str]): A :py:class:`dict` with `param1=1` and `param2='string'`.
    """

    # Tests writing data
    test_path = tmp_path / 'test.file'
    data_storage_writer: DataStorageBase = data_storage_type(test_path, mode='w')
    assert data_storage_writer
    data_storage_writer.at(data_key='data').write(data)
    data_storage_writer.close()

    # Tests writing data without using the at method to set the data key
    data_storage_writer = data_storage_type(test_path, mode='a', data_key='new_data')
    data_storage_writer.write(data)
    data_storage_writer.close()

    # Tests reading data
    data_storage_reader: DataStorageBase = data_storage_type(test_path, mode='r')
    keys = data_storage_reader.keys()
    assert len(keys) == 2
    assert 'data' in keys
    assert 'new_data' in keys
    returned_data = data_storage_reader.at(data_key='data').read()
    numpy.testing.assert_equal(returned_data, data)
    data_storage_reader.close()

    # Tests reading data without using the at method to get the data
    data_storage_reader = data_storage_type(test_path, mode='r', data_key='new_data')
    returned_data = data_storage_reader.read(data)
    numpy.testing.assert_equal(returned_data, data)
    data_storage_reader.close()

    # Tests writing data with a context manager
    with data_storage_type(test_path, mode='a') as data_storage_writer:
        data_storage_writer.at(data_key='parameter_values').write(parameter_values)

    # Tests reading data with a context manager
    with data_storage_type(test_path, mode='r') as data_storage_reader:
        assert 'parameter_values' in data_storage_reader
        assert 'data' in data_storage_reader
        assert data_storage_reader.at(data_key='parameter_values').exists()
        assert data_storage_reader.at(data_key='data').exists()
        assert not data_storage_reader.at(data_key='non_existing').exists()
        first_returned_parameter_values = data_storage_reader['parameter_values']
        second_returned_parameter_values = data_storage_reader.at(data_key='parameter_values').read()
        with pytest.raises(io.NoDataSource):
            _ = data_storage_reader['non_existing']
        numpy.testing.assert_equal(first_returned_parameter_values, parameter_values)
        numpy.testing.assert_equal(second_returned_parameter_values, parameter_values)

    # Tests that a data storage container opened for writing can also be read from
    with data_storage_type(test_path, mode='a') as data_storage_reader_writer:
        data_storage_reader_writer.at(data_key='new_entry/data').write(data)
        assert data_storage_reader_writer.at(data_key='new_entry/data').exists()
        returned_data = data_storage_reader_writer.at(data_key='new_entry/data').read()
        numpy.testing.assert_equal(returned_data, data)


def test_no_storage(data: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> None:
    """Tests the reading and writing of data to and from a :py:class:`~corelay.io.NoStorage` data storage container that raises exceptions when
    reading and writing.

    Args:
        data (numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]): Random test data with shape (10, 2).
    """

    with io.NoStorage() as data_storage:
        with pytest.raises(io.NoDataSource):
            data_storage.read()
        with pytest.raises(io.NoDataTarget):
            data_storage.write(data)
        with pytest.raises(io.NoDataSource):
            data_storage.exists()
        with pytest.raises(io.NoDataSource):
            data_storage.keys()
        assert not data_storage
