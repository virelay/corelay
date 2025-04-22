
<table align="center">
<tbody>
<tr>
<td align="center" width="1182px">

<img src="https://raw.githubusercontent.com/virelay/corelay/refs/heads/main/design/corelay-logo-with-title.png" alt="CoRelAy Logo"/>

# Composing Relevance Analysis

[![License](https://img.shields.io/pypi/l/corelay)](https://github.com/virelay/corelay/blob/main/COPYING.LESSER)
[![GitHub Actions Workflow Status](https://github.com/virelay/corelay/actions/workflows/tests.yml/badge.svg)](https://github.com/virelay/corelay/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/corelay/badge?version=latest)](https://corelay.readthedocs.io/en/latest)
[![GitHub Release](https://img.shields.io/github/v/release/virelay/corelay)](https://github.com/virelay/corelay/releases/latest)
[![PyPI Package](https://img.shields.io/pypi/v/corelay)](https://pypi.org/project/corelay/)

**CoRelAy** is a tool to compose small-scale (single-machine) analysis pipelines. Pipelines are designed with a number of steps (`Task`) with default operations (`Processor`). Any step of the pipeline may then be individually changed by assigning a new operator (`Processor`). Processors have parameters (`Params`) which define their operation.

**CoRelAy** was created to quickly implement pipelines to generate analysis data which can then be visualized using **ViRelAy**.
</td>
</tr>
</tbody>
</table>

If you find CoRelAy useful for your research, why not cite our related [paper](https://arxiv.org/abs/2106.13200):

```bibtex
@article{anders2021software,
  author  = {Anders, Christopher J. and
             Neumann, David and
             Samek, Wojciech and
             Müller, Klaus-Robert and
             Lapuschkin, Sebastian},
  title   = {Software for Dataset-wide XAI: From Local Explanations to Global Insights with {Zennit}, {CoRelAy}, and {ViRelAy}},
  journal = {CoRR},
  volume  = {abs/2106.13200},
  year    = {2021},
}
```

## Documentation

The latest documentation is hosted at [corelay.readthedocs.io](https://corelay.readthedocs.io/en/latest/).

## Install

CoRelAy may be installed using pip with

```shell
$ pip install corelay
```

> [!NOTE]
> If you experience issues installing the `metrohash-python` dependency, this may be due to the `c++` command being missing. For example, under Fedora, the `gcc-c++` package has to be installed to make the `c++` command available. You can install it using

```shell
$ sudo dnf install gcc-c++
```

To install optional HDBSCAN and UMAP support, use

```shell
$ pip install corelay[umap,hdbscan]
```

## Usage

Examples to highlight some features of **CoRelAy** can be found in `example/`.

We mainly use HDF5 files to store results. The structure used by **ViRelAy** is documented in the **ViRelAy** repository at `docs/database_specification.md`. An example to create HDF5 files which can be used with **ViRelAy** is shown in `example/hdf5_structure.py`

To do a full SpRAy analysis which can be visualized with **ViRelAy**, an advanced script can be found in `example/virelay_analysis.py`.

The following shows the contents of `example/memoize_spectral_pipeline.py`:

```python
"""An example script, which uses memoization to store (intermediate) results."""

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
```

## Contributing

TODO: Add a section on how to contribute to the project.

Installing the development dependencies can be done using the following command:

```shell
$ uv --directory source sync --all-extras
```

## License

CoRelAy is dual-licensed under the [GNU General Public License Version 3 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.html) or later, and the [GNU Lesser General Public License Version 3 (LGPL-3.0)](https://www.gnu.org/licenses/lgpl-3.0.html) or later. For more information see the [GPL-3.0](https://github.com/virelay/corelay/blob/main/COPYING) and [LGPL-3.0](https://github.com/virelay/corelay/blob/main/COPYING.LESSER) license files.
