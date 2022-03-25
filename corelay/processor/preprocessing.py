"""Preprocessing Processors

"""
from types import FunctionType

import numpy as np
import skimage.transform
import skimage.measure

from .base import Processor, Param


class PreProcessor(Processor):
    """Base-class for preprocessing.

    """
    kwargs = Param(dict, {})


class Histogram(PreProcessor):
    """Channel-wise histograms of data

    Parameters
    ----------
    bins : int
        number of bins for histograms

    """
    bins = Param(int, 32)

    def function(self, data):
        """Compute channel-wise histograms from data

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            image data with shape samples x channels x height x width

        Returns
        -------
        :obj:`numpy.ndarray`
            Channel-wise histograms with shape samples x channels x bins

        """
        n, c, h, w = data.shape
        # channel-wise range
        trange = zip(data.min((0, 2, 3), data.max(0, 2, 3)))
        # pylint: disable=no-member
        hist = np.histogramdd(data.reshape(n * c, h * w),
                              bins=self.bins, range=trange, density=True).reshape(n, c, self._bins)
        return hist


class ImagePreProcessor(PreProcessor):
    """Abstract PreProcessor for Images.

    Parameters
    ----------
    filter : int
        The order of interpolation. The order has to be in the range 0-5:
         - 0: Nearest-neighbor
         - 1: Bi-linear (default)
         - 2: Bi-quadratic
         - 3: Bi-cubic
         - 4: Bi-quartic
         - 5: Bi-quintic
    channels_first: bool
        True if channels first else False.
    """
    filter = Param(int, 1)
    channels_first = Param(bool, True)


class Resize(ImagePreProcessor):
    """Resize images.

    Parameters
    ----------
    width : int
        Width of resized images. Default: 100
    height: int
        Height of resized images. Default: 100
    filter : int
        The order of interpolation. The order has to be in the range 0-5:
         - 0: Nearest-neighbor
         - 1: Bi-linear (default)
         - 2: Bi-quadratic
         - 3: Bi-cubic
         - 4: Bi-quartic
         - 5: Bi-quintic
    channels_first: bool
        True if channels first else False.
    kwargs: dict
        Check skimage.transform.resize.

    """
    width = Param(int, 100)
    height = Param(int, 100)

    def function(self, data):
        """
        Parameters
        ----------
        data: np.ndarray
            Shape of data should be in one of the following formats:
                1. (batch_size, channels, height, width)    with channels_first=True
                2. (batch_size, height, width, channels)    with channels_first=False
                3. (batch_size, height, width)              with channels_first=False
        Returns
        -------
        data: np.ndarray
            Data is resized to given width and height.
        """
        if self.channels_first:
            data = np.moveaxis(data, 1, -1)
        # pylint: disable=not-a-mapping
        out = np.stack([skimage.transform.resize(x, output_shape=(self.height, self.width), order=self.filter,
                                                 **self.kwargs)
                        for x in data])
        if self.channels_first:
            out = np.moveaxis(out, -1, 1)
        return out


class Rescale(ImagePreProcessor):
    """Rescale images.

    Parameters
    ----------
    scale : float
        Scale to which the images are rescaled. Default: 0.5
    filter : int
        The order of interpolation. The order has to be in the range 0-5:
         - 0: Nearest-neighbor
         - 1: Bi-linear (default)
         - 2: Bi-quadratic
         - 3: Bi-cubic
         - 4: Bi-quartic
         - 5: Bi-quintic
    channels_first: bool
        True if channels first else False.
    kwargs: dict
        Check skimage.transform.rescale.

    """
    scale = Param(float, 0.5)

    def function(self, data):
        """
        Parameters
        ----------
        data: np.ndarray
            Shape of data should be in one of the following formats:
                1. (batch_size, channels, height, width)    with channels_first=True
                2. (batch_size, height, width, channels)    with channels_first=False
                3. (batch_size, height, width)              with channels_first=False
        Returns
        -------
        data: np.ndarray
            height and width axes are rescaled by scale parameter.
        """
        multichannel = len(data.shape) > 3
        if self.channels_first:
            data = np.moveaxis(data, 1, -1)
        # pylint: disable=not-a-mapping
        out = np.stack([skimage.transform.rescale(x, self.scale, order=self.filter, multichannel=multichannel,
                                                  **self.kwargs) for x in data])
        if self.channels_first:
            out = np.moveaxis(out, -1, 1)
        return out


class Pooling(PreProcessor):
    """Preform pooling operation on given stride.

    Parameters
    ----------
    stride : tuple[float]
        The pooling stride. Default: (batch, channel, height, width) -> (1, 1, 2, 2)
    pooling_function: FunctionType
        Function that reduces the selected blocks. Default: np.sum
    kwargs: dict
        Check skimage.measure.block_reduce.

    """
    stride = Param(tuple, (1, 1, 2, 2))
    pooling_function = Param(FunctionType, np.sum)

    def function(self, data):
        # pylint: disable=not-a-mapping
        return skimage.measure.block_reduce(data, self.stride, self.pooling_function, **self.kwargs)
