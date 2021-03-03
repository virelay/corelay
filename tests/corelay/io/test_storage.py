"""Test io functionalities

"""
from io import BytesIO

import pytest
import numpy as np
import h5py

from corelay import io
from corelay.io.storage import HashedHDF5


@pytest.fixture
def data():
    """Return random data with shape (10, 2).

    """
    return np.random.rand(10, 2)


@pytest.fixture
def param_values():
    """Return dict with param1=1 and param2='string'.

    """
    return dict(param1=1, param2='string')


class TestHashedHDF5:
    """Test class for HashedHDF5"""
    @staticmethod
    def test_write_array():
        """Writing a numpy array should raise no Exceptions"""
        with BytesIO() as buf, h5py.File(buf, 'w') as fd:
            group = fd.require_group('hashed')
            data_out = np.random.normal(size=5)
            iobj = HashedHDF5(group)
            iobj.write(data_out=data_out, data_in=1, meta=1)

    @staticmethod
    def test_write_tuple():
        """Writing a tuple of numpy arrays should raise no Exceptions"""
        with BytesIO() as buf, h5py.File(buf, 'w') as fd:
            group = fd.require_group('hashed')
            data_out = (np.random.normal(size=5),) * 2
            iobj = HashedHDF5(group)
            iobj.write(data_out=data_out, data_in=1, meta=1)

    @staticmethod
    def test_write_unsupported():
        """Writing an unsupported type should raise a TypeError"""
        with BytesIO() as buf, h5py.File(buf, 'w') as fd:
            group = fd.require_group('hashed')
            data_out = 'adfg'
            iobj = HashedHDF5(group)
            with pytest.raises(TypeError):
                iobj.write(data_out=data_out, data_in=1, meta=1)

    @staticmethod
    def test_read_array():
        """Reading after writing should return the same data"""
        with BytesIO() as buf, h5py.File(buf, 'w') as fd:
            group = fd.require_group('hashed')
            data_out = np.random.normal(size=5)
            iobj = HashedHDF5(group)
            iobj.write(data_out=data_out, data_in=1, meta=1)
            loaded = iobj.read(data_in=1, meta=1)
            assert (data_out == loaded).all()

    @staticmethod
    def test_read_tuple():
        """Reading after writing should return the same data"""
        with BytesIO() as buf, h5py.File(buf, 'w') as fd:
            group = fd.require_group('hashed')
            data_out = (np.random.normal(size=5),) * 2
            iobj = HashedHDF5(group)
            iobj.write(data_out=data_out, data_in=1, meta=1)
            loaded = iobj.read(data_in=1, meta=1)
            assert all((out == load).all() for out, load in zip(data_out, loaded))


@pytest.mark.parametrize("storage", [io.HDF5Storage, io.PickleStorage])
def test_data_storage_at_functionality(storage, tmp_path):
    """Test HDF5storage and PickleStorage using .at(data_key) functionality.

    """
    # pylint: disable=protected-access
    test_path = tmp_path / "test.file"
    with storage(test_path, mode='a') as data_storage:
        c = data_storage.at(data_key='param_values')
        assert c.data_key == 'param_values', "Data key was not changed."
        with pytest.raises(TypeError):
            # raises TypeError because non existing key was set.
            data_storage.at(data_key='param_values', key='key')
        with pytest.raises(TypeError):
            data_storage.at(data_key=1)

        with pytest.raises(TypeError):
            # raise TypeError, since we try to access data_key internally without it being ever set
            data_storage.at().exists()

        data_storage.at(data_key='param_values').at(data_key='data')


@pytest.mark.parametrize("storage", [io.HDF5Storage, io.PickleStorage])
def test_data_storage(storage, tmp_path, data, param_values):
    """Test HDF5storage and PickleStorage writing and loading functionalities.

    """
    # Test writing
    test_path = tmp_path / "test.file"
    data_storage = storage(test_path, mode='w')
    data_storage.at(data_key='data').write(data)
    data_storage.close()

    # Test reading
    data_storage = storage(test_path, mode='r')
    ret_data = data_storage.at(data_key='data').read()
    np.testing.assert_equal(ret_data, data)
    data_storage.close()

    # Test same with context manager
    with storage(test_path, mode='a') as data_storage:
        data_storage.at(data_key='param_values').write(param_values)

    with storage(test_path, mode='r') as data_storage:
        assert 'param_values' in data_storage
        assert 'data' in data_storage
        assert data_storage.at(data_key='param_values').exists()
        assert data_storage.at(data_key='data').exists()
        assert not data_storage.at(data_key='non_existing').exists()
        ret_param_values = data_storage['param_values']
        ret_param_values_2 = data_storage.at(data_key='param_values').read()
        with pytest.raises(io.NoDataSource):
            _ = data_storage['non_existing']

    np.testing.assert_equal(ret_param_values, param_values)
    np.testing.assert_equal(ret_param_values_2, param_values)

    with storage(test_path, mode='a') as data_storage:
        data_storage.at(data_key='new_entry/data').write(data)
        assert data_storage.at(data_key='new_entry/data').exists()
        ret_data = data_storage.at(data_key='new_entry/data').read()
        np.testing.assert_equal(ret_data, data)


def test_no_storage(data):
    """Test NoStorage instance that raises error when reading and writing.

    """
    data_storage = io.NoStorage()
    with pytest.raises(io.NoDataSource):
        data_storage.read()
    with pytest.raises(io.NoDataTarget):
        data_storage.write(data)
