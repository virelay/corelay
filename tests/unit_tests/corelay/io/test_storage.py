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
    data_storage_writer.at(data_key='data').write(data)
    data_storage_writer.close()

    # Tests reading data
    data_storage_reader: DataStorageBase = data_storage_type(test_path, mode='r')
    returned_data = data_storage_reader.at(data_key='data').read()
    numpy.testing.assert_equal(returned_data, data)
    data_storage_reader.close()

    # Tests writing data with a context manager
    with data_storage_type(test_path, mode='a') as data_storage_writer:
        data_storage_writer.at(data_key='param_values').write(parameter_values)

    # Tests reading data with a context manager
    with data_storage_type(test_path, mode='r') as data_storage_reader:
        assert 'param_values' in data_storage_reader
        assert 'data' in data_storage_reader
        assert data_storage_reader.at(data_key='param_values').exists()
        assert data_storage_reader.at(data_key='data').exists()
        assert not data_storage_reader.at(data_key='non_existing').exists()
        first_returned_parameter_values = data_storage_reader['param_values']
        second_returned_parameter_values = data_storage_reader.at(data_key='param_values').read()
        with pytest.raises(io.NoDataSource):
            _ = data_storage_reader['non_existing']

    numpy.testing.assert_equal(first_returned_parameter_values, parameter_values)
    numpy.testing.assert_equal(second_returned_parameter_values, parameter_values)

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

    data_storage = io.NoStorage()
    with pytest.raises(io.NoDataSource):
        data_storage.read()
    with pytest.raises(io.NoDataTarget):
        data_storage.write(data)
