"""A module that contains processors for pre-processing data and images."""

import typing
from types import FunctionType
from typing import Annotated

import numpy
import skimage.measure
import skimage.transform

from corelay.base import Param
from corelay.processor.base import Processor


class PreProcessor(Processor):
    """The abstract base class for pre-processing processors.

    Args:
        is_output (bool): A value indicating whether this :py:class:`PreProcessor` is the output of a :py:class:`~corelay.pipeline.base.Pipeline`.
            Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        kwargs (dict): Additional keyword arguments to pass to the pre-processing function. Defaults to an empty :py:class:`dict`.
    """

    kwargs: Annotated[dict[str, typing.Any], Param(dict, {})]
    """Additional keyword arguments to pass to the pre-processing function."""


class Histogram(PreProcessor):
    """A :py:class:`~corelay.processor.base.Processor` that computes channel-wise histograms of the input data.

    Args:
        is_output (bool): A value indicating whether this :py:class:`Histogram` pre-processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        kwargs (dict): Additional keyword arguments to pass to the pre-processing function. Defaults to an empty :py:class:`dict`.
        bins (int): The number of bins for the histogram. Defaults to 32.
    """

    bins: Annotated[int, Param(int, 32)]
    """Number of bins for the histogram. Defaults to 32."""

    def function(self, data: typing.Any) -> typing.Any:
        """Computes channel-wise histograms from the input data.

        Args:
            data (typing.Any): The input data to compute histograms for, which is should be an image as a NumPy array of shape
                `(number_of_samples, number_of_channels, height, width)`.

        Returns:
            typing.Any: Returns a NumPy array, which contains the channel-wise histograms of the input data of shape
            `(number_of_samples, number_of_channels, bins)`.
        """

        input_data: numpy.ndarray[typing.Any, typing.Any] = data
        number_of_samples, number_of_channels, height, width = input_data.shape

        channel_minima: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] = input_data.min((0, 2, 3))
        channel_maxima: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]] = input_data.max((0, 2, 3))
        channel_range = list(zip(channel_minima, channel_maxima))

        histogram, _ = numpy.histogramdd(
            input_data.reshape(number_of_samples * number_of_channels, height * width),
            bins=self.bins,
            range=channel_range,
            density=True
        )
        return histogram.reshape(number_of_samples, number_of_channels, self.bins)


class ImagePreProcessor(PreProcessor):
    """The abstract base class for all processors that perform pre-processing on images.

    Args:
        is_output (bool): A value indicating whether this :py:class:`ImagePreProcessor` pre-processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        kwargs (dict): Additional keyword arguments to pass to the image pre-processing function. Defaults to an empty :py:class:`dict`.
        filter (int): The order of interpolation. The order has to be in the range 0-5. Defaults to 1 (bi-linear).
        channels_first (bool): A value indicating whether the input data is in channels-first format or not. Defaults to :py:obj:`True`.
    """

    filter: Annotated[int, Param(int, 1)]
    """The order of interpolation. The order has to be in the range 0-5:

    - 0: Nearest-neighbor
    - 1: Bi-linear (default)
    - 2: Bi-quadratic
    - 3: Bi-cubic
    - 4: Bi-quartic
    - 5: Bi-quintic

    Defaults to 1 (bi-linear).
    """

    channels_first: Annotated[bool, Param(bool, True)]
    """A value indicating whether the input data is in channels-first format or not. Defaults to :py:obj:`True`."""


class Resize(ImagePreProcessor):
    """A :py:class:`~corelay.processor.base.Processor` that resizes images to a specified width and height.

    Args:
        is_output (bool): A value indicating whether this :py:class:`Resize` image pre-processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        kwargs (dict): Additional keyword arguments to pass to the image pre-processing function. Defaults to an empty :py:class:`dict`.
        filter (int): The order of interpolation. The order has to be in the range 0-5. Defaults to 1 (bi-linear).
        channels_first (bool): A value indicating whether the input data is in channels-first format or not. Defaults to :py:obj:`True`.
        width (int): The width to which the images are resized. Defaults to 100.
        height (int): The height to which the images are resized. Defaults to 100.
    """

    width: Annotated[int, Param(int, 100)]
    """The width to which the images are resized. Defaults to 100."""

    height: Annotated[int, Param(int, 100)]
    """The height to which the images are resized. Defaults to 100."""

    def function(self, data: typing.Any) -> typing.Any:
        """Resizes the input images to the specified width and height.

        Args:
            data (typing.Any): The input data, which contains the images that are to be resized. The input data should be a NumPy array in one of the
                following formats:

                1. `(batch_size, number_of_channels, height, width)`, if :py:attr:`~ImagePreProcessor.channels_first` is set to :py:obj:`True`.
                2. `(batch_size, height, width, number_of_channels)`, if :py:attr:`~ImagePreProcessor.channels_first` is set to :py:obj:`False`.
                3. `(batch_size, height, width)`, if :py:attr:`~ImagePreProcessor.channels_first` is set to :py:obj:`False`.

        Returns:
            typing.Any: Returns a NumPy array containing the resized images, with a shape that matches the input data format.
        """

        input_images: numpy.ndarray[typing.Any, typing.Any] = data

        if self.channels_first:
            input_images = numpy.moveaxis(data, 1, -1)

        resized_images = numpy.stack([
            skimage.transform.resize(  # type: ignore[no-untyped-call]
                image,
                output_shape=(self.height, self.width),
                order=self.filter,
                **self.kwargs
            ) for image in input_images
        ])

        if self.channels_first:
            resized_images = numpy.moveaxis(resized_images, -1, 1)
        return resized_images


class Rescale(ImagePreProcessor):
    """A :py:class:`~corelay.processor.base.Processor` that rescales images by a specified scale factor.

    Args:
        is_output (bool): A value indicating whether this :py:class:`Rescale` image pre-processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        kwargs (dict): Additional keyword arguments to pass to the image pre-processing function. Defaults to an empty :py:class:`dict`.
        filter (int): The order of interpolation. The order has to be in the range 0-5. Defaults to 1 (bi-linear).
        channels_first (bool): A value indicating whether the input data is in channels-first format or not. Defaults to :py:obj:`True`.
        scale (float): The scale factor by which the images are rescaled. Defaults to 0.5.
    """

    scale: Annotated[float, Param(float, 0.5)]
    """The scale factor by which the images are rescaled. Defaults to 0.5."""

    def function(self, data: typing.Any) -> typing.Any:
        """Rescales the input images by the specified scale factor.

        Args:
            data (typing.Any): The input data, which contains the images that are to be rescaled. The input data should be a NumPy array in one of the
                following formats:

                1. `(batch_size, number_of_channels, height, width)`, if :py:attr:`~ImagePreProcessor.channels_first` is set to :py:obj:`True`.
                2. `(batch_size, height, width, number_of_channels)`, if :py:attr:`~ImagePreProcessor.channels_first` is set to :py:obj:`False`.
                3. `(batch_size, height, width) `, if :py:attr:`~ImagePreProcessor.channels_first` is set to :py:obj:`False`.

        Returns:
            typing.Any: Returns a NumPy array containing the rescaled images, with a shape that matches the input data format.
        """

        input_data: numpy.ndarray[typing.Any, typing.Any] = data
        images_are_multi_channel = len(input_data.shape) > 3

        if self.channels_first:
            input_data = numpy.moveaxis(input_data, 1, -1)

        rescaled_images = numpy.stack([
            skimage.transform.rescale(
                image,
                self.scale,
                order=self.filter,
                channel_axis=-1 if images_are_multi_channel else None,
                **self.kwargs
            ) for image in input_data
        ])

        if self.channels_first:
            rescaled_images = numpy.moveaxis(rescaled_images, -1, 1)
        return rescaled_images


class Pooling(PreProcessor):
    """A :py:class:`~corelay.processor.base.Processor` that performs image pooling on the input data.

    Args:
        is_output (bool): A value indicating whether this :py:class:`Pooling` image pre-processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        kwargs (dict): Additional keyword arguments to pass to the image pre-processing function. Defaults to an empty :py:class:`dict`.
        filter (int): The order of interpolation. The order has to be in the range 0-5. Defaults to 1 (bi-linear).
        channels_first (bool): A value indicating whether the input data is in channels-first format or not. Defaults to :py:obj:`True`.
        stride (tuple[int]): The pooling stride, which should be of shape `(batch_size, number_of_channels, height, width)`. Defaults to
            `(1, 1, 2, 2)`.
        pooling_function (FunctionType): The pooling function to use to reduce the selected blocks. Defaults to :py:func:`numpy.sum`.
    """

    stride: Annotated[tuple[int], Param(tuple, (1, 1, 2, 2))]
    """The pooling stride, which should be of shape `(batch_size, number_of_channels, height, width)`. Defaults to `(1, 1, 2, 2)`."""

    pooling_function: Annotated[FunctionType, Param(FunctionType, numpy.sum)]
    """The pooling function to use to reduce the selected blocks. Defaults to :py:func:`numpy.sum`."""

    def function(self, data: typing.Any) -> typing.Any:
        """Performs pooling on the input data.

        Args:
            data (typing.Any): The input data, which should be a NumPy array of shape `(number_of_samples, number_of_channels, height, width)`.

        Returns:
            typing.Any: Returns a NumPy array containing the pooled data.
        """

        input_data: numpy.ndarray[typing.Any, typing.Any] = data
        return skimage.measure.block_reduce(  # type: ignore[no-untyped-call]
            input_data,
            self.stride,
            self.pooling_function,
            **self.kwargs
        )
