"""A module that contains unit tests for the :py:mod:`corelay.processor.preprocessing` module."""

import typing

import numpy
import pytest

from corelay.processor.preprocessing import Histogram, Rescale, Resize, Pooling


@pytest.fixture(name='no_channels', scope='module')
def get_no_channels_fixture() -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
    """Generates a grayscale image-like array that contains all ones of shape `(number_of_samples, height, width)`.

    Returns:
        numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: Returns an image-like array of shape `(number_of_samples, height, width)`.
    """

    return numpy.ones((10, 8, 8))


@pytest.fixture(name='channels_first', scope='module')
def get_channels_first_fixture() -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
    """Generates an image-like array that contains all ones of shape `(number_of_samples, number_of_channels, height, width)`, where the channels come
    before the image size.

    Returns:
        numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: Returns an image-like array of shape
        `(number_of_samples, number_of_channels, height, width)`.
    """

    return numpy.ones((10, 3, 8, 8))


@pytest.fixture(name='channels_last', scope='module')
def get_channels_last_fixture() -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
    """Generates an image-like array that contains all ones of shape `(number_of_samples, height, width, number_of_channels)`, where the channels come
    last.

    Returns:
        numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: Returns an image-like array of shape
        `(number_of_samples, height, width, number_of_channels)`.
    """

    return numpy.ones((10, 8, 8, 3))


@pytest.fixture(name='random_noise', scope='module')
def get_random_noise_fixture() -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
    """Generates a grayscale image-like array that contains all normally distributed noise of shape `(number_of_samples, height, width)`.

    Returns:
        numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: Returns an image-like array of shape `(number_of_samples, height, width)`.
    """

    return numpy.random.normal(0, 1, (10, 8, 8))


@pytest.mark.parametrize(
    'fixture_name,shape',
    [
        ('random_noise', (10, 4, 4)),
        ('channels_first', (10, 3, 4, 4)),
        ('channels_last', (10, 4, 4, 3))
    ]
)
def test_rescaling(fixture_name: str, shape: tuple[int, ...], request: pytest.FixtureRequest) -> None:
    """Tests the rescaling pre-processing processor.

    Args:
        fixture_name (str): The name of the fixture that is to be used for the test.
        shape (tuple[int, ...]): The expected shape of the output data.
        request (pytest.FixtureRequest): The request object that is used to access the fixture.
    """

    processor = Rescale(scale=0.5, channels_first=fixture_name == 'channels_first')
    input_data: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] = request.getfixturevalue(fixture_name)

    output_data: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] = processor(input_data)

    assert output_data.shape == shape
    assert input_data.max() >= output_data.max()
    assert input_data.min() <= output_data.min()


@pytest.mark.parametrize(
    'fixture_name,shape',
    [
        ('random_noise', (10, 4, 16)),
        ('channels_first', (10, 3, 4, 16)),
        ('channels_last', (10, 4, 16, 3))
    ]
)
def test_resizing(fixture_name: str, shape: tuple[int, ...], request: pytest.FixtureRequest) -> None:
    """Tests the resizing pre-processing processor.

    Args:
        fixture_name (str): The name of the fixture that is to be used for the test.
        shape (tuple[int, ...]): The expected shape of the output data.
        request (pytest.FixtureRequest): The request object that is used to access the fixture.
    """

    processor = Resize(width=16, height=4, channels_first=fixture_name == 'channels_first')
    input_data = request.getfixturevalue(fixture_name)

    output_data = processor(input_data)

    assert output_data.shape == shape
    assert input_data.max() >= output_data.max()
    assert input_data.min() <= output_data.min()


@pytest.mark.parametrize(
    'fixture_name,shape,stride',
    [
        ('no_channels', (10, 4, 4), (1, 2, 2)),
        ('channels_first', (10, 3, 4, 4), (1, 1, 2, 2)),
        ('channels_last', (10, 4, 4, 3), (1, 2, 2, 1))
    ]
)
def test_pooling(fixture_name: str, shape: tuple[int, ...], stride: tuple[int, ...], request: pytest.FixtureRequest) -> None:
    """Tests the pooling pre-processing processor.

    Args:
        fixture_name (str): The name of the fixture that is to be used for the test.
        shape (tuple[int, ...]): The expected shape of the output data.
        stride (tuple[int, ...]): The stride of the pooling operation.
        request (pytest.FixtureRequest): The request object that is used to access the fixture.
    """

    processor = Pooling(stride=stride, pooling_function=numpy.sum)
    input_data = request.getfixturevalue(fixture_name)

    output_data = processor(input_data)

    assert output_data.shape == shape
    numpy.testing.assert_equal(output_data, 4 * numpy.ones(output_data.shape))


def test_histogram() -> None:
    """Tests the histogram pre-processing processor."""

    processor = Histogram(bins=2)
    input_data = numpy.array([[
        [[1, 2],
         [3, 4]],
        [[1, 2],
         [3, 4]],
        [[1, 2],
         [3, 4]]
    ]])

    output_data = processor(input_data)
    assert output_data.shape == (1, 3, 2)

    expected_histogram = numpy.array([[
        [1.0 / 3.0, 1.0 / 3.0],
        [1.0 / 3.0, 1.0 / 3.0],
        [1.0 / 3.0, 1.0 / 3.0]
    ]])
    numpy.testing.assert_equal(output_data, expected_histogram)


def test_histogram_with_channel_last_input() -> None:
    """Tests the histogram pre-processing processor with images were the channel dimension comes last."""

    processor = Histogram(bins=2, channels_first=False)
    input_data = numpy.array([[[
        [1, 1, 1],
        [2, 2, 2]
    ], [
        [3, 3, 3],
        [4, 4, 4]
    ]]])

    output_data = processor(input_data)
    assert output_data.shape == (1, 3, 2)

    expected_histogram = numpy.array([[
        [1.0 / 3.0, 1.0 / 3.0],
        [1.0 / 3.0, 1.0 / 3.0],
        [1.0 / 3.0, 1.0 / 3.0]
    ]])
    numpy.testing.assert_equal(output_data, expected_histogram)


def test_histogram_with_grayscale_input() -> None:
    """Tests the histogram pre-processing processor with grayscale images."""

    processor = Histogram(bins=2)
    input_data = numpy.array([[
        [1, 2],
        [3, 4]
    ]])

    output_data = processor(input_data)
    assert output_data.shape == (1, 1, 2)

    expected_histogram = numpy.array([[
        [1.0 / 3.0, 1.0 / 3.0]
    ]])
    numpy.testing.assert_equal(output_data, expected_histogram)
