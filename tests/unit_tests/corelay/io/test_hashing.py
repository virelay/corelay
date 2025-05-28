"""A module that contains unit tests for the :py:mod:`corelay.io.hashing` module."""

import importlib
import sys
import typing
from io import BytesIO
from types import ModuleType

import numpy
import pytest

import corelay.io.hashing
from corelay.io.hashing import HashPickler, Hasher, TensorPlaceholder, Tensor


def test_tensor_placeholder_is_not_implemented() -> None:
    """Tests that the :py:class:`~corelay.io.hashing.TensorPlaceholder` class raises a :py:class:`NotImplementedError` exception when calling
    :py:meth:`~corelay.io.hashing.TensorPlaceholder.numpy`.
    """

    tensor_placeholder = TensorPlaceholder()
    with pytest.raises(NotImplementedError):
        tensor_placeholder.numpy()


def test_tensor_class_is_placeholder_when_pytorch_not_installed() -> None:
    """Tests that the :py:class:`~corelay.io.hashing.TensorPlaceholder` class is used as a placeholder for :py:class:`~torch.Tensor` when PyTorch is
    not installed.
    """

    assert Tensor is TensorPlaceholder


def test_tensor_class_is_pytorch_tensor_when_pytorch_installed() -> None:
    """Tests that the :py:class:`~corelay.io.hashing.Tensor` class is a :py:class:`~torch.Tensor` when PyTorch is installed."""

    class PyTorchTensorMock:
        """A mock class to simulate the PyTorch Tensor class."""

    # Since PyTorch is not actually installed in the test environment, we need to mock the torch module
    pytorch_module_mock = ModuleType('torch')
    pytorch_module_mock.__dict__.update({'Tensor': PyTorchTensorMock})
    sys.modules['torch'] = pytorch_module_mock

    # Since the corelay.io.hashing module has already been imported, we need to reload it to ensure that the Tensor class is updated
    importlib.reload(corelay.io.hashing)

    # Now we can test that the Tensor class is a PyTorch Tensor
    assert corelay.io.hashing.Tensor is PyTorchTensorMock  # type: ignore[comparison-overlap]

    # After the test, we need to ensure that the PyTorch module is no longer available and reload the corelay.io.hashing module, so that it reverts
    # back to using the TensorPlaceholder class
    del sys.modules['torch']
    importlib.reload(corelay.io.hashing)


class TestHasher:
    """Contains unit tests for the :py:class:`~corelay.io.hashing.Hasher` class."""

    @staticmethod
    def test_hasher_produces_correct_hash() -> None:
        """Tests that the :py:func:`~corelay.io.hashing.hasher` function produces the correct hash for a given input."""

        # Both the test string and the expected hash were taken from the test cases in the original source code of the MetroHash Python library on
        # which the Hasher class is based (the test string was selected such that it will hit each internal branch of the MetroHash algorithm and
        # therefore must be at least 63 bytes long)
        hasher = Hasher()
        test_string = b'012345678901234567890123456789012345678901234567890123456789012'
        number_of_bytes_written = hasher.write(test_string)
        assert number_of_bytes_written == 63
        hash_digest = hasher.hexdigest()
        assert hash_digest == 'c77ce2bfa4ed9f9b0548b2ac5074a297'


class TestHashPickler:
    """Contains unit tests for the :py:class:`~corelay.io.hashing.HashPickler` class."""

    @staticmethod
    def test_correct_numpy_id_is_generated() -> None:
        """Tests that the :py:func:`~corelay.io.hashing.HashPickler.numpy_id` function generates the correct NumPy ID for a given NumPy array."""

        test_array = numpy.array([1.0, 2.0, 3.0])
        test_array_numpy_id = HashPickler.numpy_id(test_array)

        data_type_name, shape, mantissa, exponent = test_array_numpy_id
        assert data_type_name == 'float64'
        assert shape == (3,)
        assert mantissa == b'\x00\x00\x00\x00\x00\x00\xe0?\x00\x00\x00\x00\x00\x00\xe0?\x00\x00\x00\x00\x00\x00\xe8?'
        assert exponent == b'\x01\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00'

    @staticmethod
    def test_correct_persistent_id_is_generated_for_numpy_array() -> None:
        """Tests that the :py:func:`~corelay.io.hashing.HashPickler.persistent_id` function generates the correct persistent ID for a given NumPy
        array.
        """

        hash_pickler = HashPickler(BytesIO())
        test_array = numpy.array([1.0, 2.0, 3.0])
        test_array_persistent_id = hash_pickler.persistent_id(test_array)

        assert test_array_persistent_id is not None
        data_type_name, shape, mantissa, exponent = test_array_persistent_id
        assert data_type_name == 'float64'
        assert shape == (3,)
        assert mantissa == b'\x00\x00\x00\x00\x00\x00\xe0?\x00\x00\x00\x00\x00\x00\xe0?\x00\x00\x00\x00\x00\x00\xe8?'
        assert exponent == b'\x01\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00'

    @staticmethod
    def test_correct_persistent_id_is_generated_for_pytorch_tensor() -> None:
        """Tests that the :py:func:`~corelay.io.hashing.HashPickler.persistent_id` function generates the correct persistent ID for a given PyTorch
        tensor.
        """

        class PyTorchTensorMock:
            """A mock class to simulate the PyTorch Tensor class."""

            def numpy(self) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
                """Converts the tensor to a NumPy array.

                Returns:
                    numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: Returns a NumPy array representation of the PyTorch tensor.
                """

                return numpy.array([1.0, 2.0, 3.0])

        # Since PyTorch is not actually installed in the test environment, we need to mock the torch module
        pytorch_module_mock = ModuleType('torch')
        pytorch_module_mock.__dict__.update({'Tensor': PyTorchTensorMock})
        sys.modules['torch'] = pytorch_module_mock

        # Since the corelay.io.hashing module has already been imported, we need to reload it to ensure that the Tensor class is updated
        importlib.reload(corelay.io.hashing)

        # Now we can test that the persistent_id function generates the correct ID for a PyTorch tensor
        hash_pickler = corelay.io.hashing.HashPickler(BytesIO())
        test_tensor = PyTorchTensorMock()
        test_tensor_persistent_id = hash_pickler.persistent_id(test_tensor)

        # Checks if the returned persistent ID contains the expected values
        assert test_tensor_persistent_id is not None
        data_type_name, shape, mantissa, exponent = test_tensor_persistent_id
        assert data_type_name == 'float64'
        assert shape == (3,)
        assert mantissa == b'\x00\x00\x00\x00\x00\x00\xe0?\x00\x00\x00\x00\x00\x00\xe0?\x00\x00\x00\x00\x00\x00\xe8?'
        assert exponent == b'\x01\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00'

        # After the test, we need to ensure that the PyTorch module is no longer available and reload the corelay.io.hashing module, so that it
        # reverts back to using the TensorPlaceholder class
        del sys.modules['torch']
        importlib.reload(corelay.io.hashing)

    @staticmethod
    def test_correct_persistent_id_is_generated_for_string() -> None:
        """Tests that the :py:func:`~corelay.io.hashing.HashPickler.persistent_id` function generates the correct persistent ID for a given string.
        For anything other than a NumPy array or a PyTorch tensor, the persistent ID should be :py:obj:`None`.
        """

        hash_pickler = HashPickler(BytesIO())
        test_string = 'This is a test string.'
        test_string_persistent_id = hash_pickler.persistent_id(test_string)

        assert test_string_persistent_id is None


def test_ext_hash_produces_correct_hash() -> None:
    """Tests that the :py:func:`~corelay.io.hashing.ext_hash` function correctly uses the :py:class:`~corelay.io.hashing.HashPickler` to produce a
    hash of test data.
    """

    class PyTorchTensorMock:
        """A mock class to simulate the PyTorch Tensor class."""

        def numpy(self) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
            """Converts the tensor to a NumPy array.

            Returns:
                numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: Returns a NumPy array representation of the PyTorch tensor.
            """

            return numpy.array([1.0, 2.0, 3.0])

    # Since PyTorch is not actually installed in the test environment, we need to mock the torch module
    pytorch_module_mock = ModuleType('torch')
    pytorch_module_mock.__dict__.update({'Tensor': PyTorchTensorMock})
    sys.modules['torch'] = pytorch_module_mock

    # Since the corelay.io.hashing module has already been imported, we need to reload it to ensure that the Tensor class is updated
    importlib.reload(corelay.io.hashing)

    # Now we can test that the HashPickler can pickle and unpickle test data correctly (the hash was pre-computed, this test is merely to detect if a
    # breaking change was introduced in the MetroHash or MetroHash Python libraries or in the pickle module)
    test_data = {
        'string': 'This is a test string.',
        'numpy_array': numpy.array([1.0, 2.0, 3.0]),
        'pytorch_tensor': PyTorchTensorMock(),
    }
    test_data_hash = corelay.io.hashing.ext_hash(test_data)
    assert test_data_hash == '6015b8a73a4548ea9a3d87616fc421ec'

    # After the test, we need to ensure that the PyTorch module is no longer available and reload the corelay.io.hashing module, so that it
    # reverts back to using the TensorPlaceholder class
    del sys.modules['torch']
    importlib.reload(corelay.io.hashing)
