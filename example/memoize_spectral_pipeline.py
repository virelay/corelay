"""An example script, which uses memoization to store (intermediate) results."""

# pylint: disable=duplicate-code

import time
from collections.abc import Sequence
from typing import Annotated, Any, SupportsIndex

import h5py
import numpy
from numpy.typing import NDArray

from corelay.base import Param
from corelay.io.storage import HashedHDF5
from corelay.pipeline.spectral import SpectralClustering
from corelay.processor.base import Processor
from corelay.processor.clustering import KMeans
from corelay.processor.embedding import TSNEEmbedding, EigenDecomposition
from corelay.processor.flow import Sequential, Parallel


class Flatten(Processor):
    """Represents a CoRelAy processor, which flattens its input data."""

    def function(self, data: Any) -> Any:
        """Applies the flattening to the input data.

        Args:
            data (Any): The input data that is to be flattened.

        Returns:
            Any: Returns the flattened data.
        """

        input_data: NDArray[Any] = data
        input_data.sum()
        return input_data.reshape(input_data.shape[0], numpy.prod(input_data.shape[1:]))


class SumChannel(Processor):
    """Represents a CoRelAy processor, which sums its input data across channels, i.e., its second axis."""

    def function(self, data: Any) -> Any:
        """Applies the summation over the channels to the input data.

        Args:
            data (Any): The input data that is to be summed over its channels.

        Returns:
            Any: Returns the data that was summed up over its channels.
        """

        input_data: NDArray[Any] = data
        return input_data.sum(axis=1)


class Normalize(Processor):
    """Represents a CoRelAy processor, which normalizes its input data."""

    axes: Annotated[SupportsIndex | Sequence[SupportsIndex], Param((SupportsIndex, Sequence), (1, 2))]
    """A parameter of the processor, which determines the axis over which the data is to be normalized. Defaults to the second and third axes."""

    def function(self, data: Any) -> Any:
        """Normalizes the specified input data.

        Args:
            data (Any): The input data that is to be normalized.

        Returns:
            Any: Returns the normalized input data.
        """

        input_data: NDArray[Any] = data
        return input_data / input_data.sum(self.axes, keepdims=True)


def main() -> None:
    """The entrypoint to the memoize_spectral_pipeline script."""

    # Fixes the random seed for reproducibility
    numpy.random.seed(0xDEADBEEF)

    # Opens an HDF5 file in append mode for the storing the results of the analysis and the memoization of intermediate pipeline results
    with h5py.File('test.analysis.h5', 'a') as analysis_file:

        # Creates a HashedHDF5 IO object, which is an IO object that stores outputs of processors based on hashes in an HDF5 file
        io_object = HashedHDF5(analysis_file.require_group('proc_data'))

        # Generates some exemplary data
        data = numpy.random.normal(size=(64, 3, 32, 32))
        number_of_clusters = range(2, 20)

        # Creates a SpectralClustering pipeline, which is one of the pre-defined built-in pipelines
        pipeline = SpectralClustering(

            # Processors, such as EigenDecomposition, can be assigned to pre-defined tasks
            embedding=EigenDecomposition(n_eigval=8, io=io_object),

            # Flow-based processors, such as Parallel, can combine multiple processors; broadcast=True copies the input as many times as there are
            # processors; broadcast=False instead attempts to match each input to a processor
            clustering=Parallel([
                Parallel([
                    KMeans(n_clusters=k, io=io_object) for k in number_of_clusters
                ], broadcast=True),

                # IO objects will be used during computation when supplied to processors, if a corresponding output value (here identified by hashes)
                # already exists, the value is not computed again but instead loaded from the IO object
                TSNEEmbedding(io=io_object)
            ], broadcast=True, is_output=True)
        )

        # Processors (and Params) can be updated by simply assigning corresponding attributes
        pipeline.preprocessing = Sequential([
            SumChannel(),
            Normalize(),
            Flatten()
        ])

        # Processors flagged with "is_output=True" will be accumulated in the output; the output will be a tree of tuples, with the same hierarchy as
        # the pipeline (i.e., _clusterings here contains a tuple of the k-means outputs)
        start_time = time.perf_counter()
        _clusterings, _tsne = pipeline(data)

        # Since we memoize our results in an HDF5 file, subsequent calls will not compute the values (for the same inputs), but rather load them from
        # the HDF5 file; try running the script multiple times
        duration = time.perf_counter() - start_time
        print(f'Pipeline execution time: {duration:.4f} seconds')


if __name__ == '__main__':
    main()
