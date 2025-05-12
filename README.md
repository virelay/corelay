
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

**CoRelAy** is a library designed for composing efficient, single-machine data analysis pipelines. It enables the rapid implementation of pipelines that can be used to analyze and process data. CoRelAy is primarily meant for the use in explainable artificial intelligence (XAI), often with the goal of producing output suitable for visualization in tools like [**ViRelAy**](https://github.com/virelay/virelay).
</td>
</tr>
</tbody>
</table>

At the core of CoRelAy are **pipelines** (`Pipeline`), which consist of a series of **tasks** (`Task`). Each task is a modular unit that can be populated with **operations** (`Processor`) to perform specific data processing tasks. These operations, known as processors, can be customized by assigning new instances or modifying their default configurations.

Tasks in CoRelAy are highly flexible and can be tailored to meet the needs of your analysis pipeline. By leveraging a wide range of configurable **processors** with their respective **parameters** (`Param`), you can easily adapt and optimize your data processing workflow.

For more information about CoRelAy, getting started guides, in-depth tutorials, and API documentation, please refer to the [documentation](https://corelay.readthedocs.io/en/latest/).

If you find CoRelAy useful for your research, why not cite our related [paper](https://arxiv.org/abs/2106.13200):

```bibtex
@article{anders2021software,
  author  = {Anders, Christopher J. and
             Neumann, David and
             Samek, Wojciech and
             Müller, Klaus-Robert and
             Lapuschkin, Sebastian},
  title   = {Software for Dataset-wide XAI: From Local Explanations to Global Insights with {Zennit}, {CoRelAy}, and {ViRelAy}},
  year    = {2021},
  volume  = {abs/2106.13200},
  journal = {CoRR}
}
```

## Features

- **Pipeline Composition**: CoRelAy allows you to compose pipelines of processors, which can be executed in parallel or sequentially.
- **Task-based Design**: Each step in the pipeline is represented as a task, which can be easily modified or replaced.
- **Processor Library**: CoRelAy comes with a library of built-in processors for common tasks, such as clustering, embedding, and dimensionality reduction.
- **Memoization**: CoRelAy supports memoization of intermediate results, allowing you to reuse previously computed results and speed up your analysis.

## Getting Started

### Installation

To get started, you first have to install CoRelAy on your system. The recommended and easiest way to install CoRelAy is to use `pip`, the Python package manager. You can install CoRelAy using the following command:

```shell
$ pip install corelay
```

> [!NOTE]
> CoRelAy depends on the [`metrohash-python`](https://pypi.org/project/metrohash-python/) library, which requires a C++ compiler to be installed. This may mean that you will have to install extra packages (GCC or Clang) for the installation to succeed. For example, on Fedora, you may have to install the `gcc-c++` package in order to make the `c++` command available, which can be done using the following command:
>
> ```shell
> $ sudo dnf install gcc-c++
> ```

To install CoRelAy with optional HDBSCAN and UMAP support, use

```shell
$ pip install corelay[umap,hdbscan]
```

### Usage

Examples to highlight some features of CoRelAy can be found in [`docs/examples`](https://github.com/virelay/corelay/tree/main/docs/examples).

We mainly use HDF5 files to store results. If you wish to visualize your analysis results using **ViRelAy**, please have a look at the [**ViRelAy documentation**](https://virelay.readthedocs.io/en/latest/contributors-guide/database-specification.html) to find out more about its database specification. An example to create HDF5 files which can be used with **ViRelAy** is shown in [`docs/examples/hdf5_structure.py`](https://github.com/virelay/corelay/tree/main/docs/examples/hdf5_structure.py).

To do a full SpRAy analysis which can be visualized with **ViRelAy**, an advanced script can be found in [`docs/examples/virelay_analysis.py`](https://github.com/virelay/corelay/tree/main/docs/examples/virelay_analysis.py).

The following shows the contents of [`docs/examples/memoize_spectral_pipeline.py`](https://github.com/virelay/corelay/tree/main/docs/examples/memoize_spectral_pipeline.py):

```python
"""An example script, which uses memoization to store (intermediate) results."""

import time
import typing
from collections.abc import Sequence
from typing import Annotated, SupportsIndex

import h5py
import numpy

from corelay.base import Param
from corelay.io.storage import HashedHDF5
from corelay.pipeline.spectral import SpectralClustering
from corelay.processor.base import Processor
from corelay.processor.clustering import KMeans
from corelay.processor.embedding import TSNEEmbedding, EigenDecomposition
from corelay.processor.flow import Sequential, Parallel


class Flatten(Processor):
    """Represents a :py:class:`~corelay.processor.base.Processor`, which flattens its input data."""

    def function(self, data: typing.Any) -> typing.Any:
        """Applies the flattening to the input data.

        Args:
            data (typing.Any): The input data that is to be flattened.

        Returns:
            typing.Any: Returns the flattened data.
        """

        input_data: numpy.ndarray[typing.Any, typing.Any] = data
        input_data.sum()
        return input_data.reshape(input_data.shape[0], numpy.prod(input_data.shape[1:]))


class SumChannel(Processor):
    """Represents a :py:class:`~corelay.processor.base.Processor`, which sums its input data across channels, i.e., its second axis."""

    def function(self, data: typing.Any) -> typing.Any:
        """Applies the summation over the channels to the input data.

        Args:
            data (typing.Any): The input data that is to be summed over its channels.

        Returns:
            typing.Any: Returns the data that was summed up over its channels.
        """

        input_data: numpy.ndarray[typing.Any, typing.Any] = data
        return input_data.sum(axis=1)


class Normalize(Processor):
    """Represents a :py:class:`~corelay.processor.base.Processor`, which normalizes its input data."""

    axes: Annotated[SupportsIndex | Sequence[SupportsIndex], Param((SupportsIndex, Sequence), (1, 2))]
    """A parameter of the :py:class:`~corelay.processor.base.Processor`, which determines the axis over which the data is to be normalized. Defaults
    to the second and third axes.
    """

    def function(self, data: typing.Any) -> typing.Any:
        """Normalizes the specified input data.

        Args:
            data (typing.Any): The input data that is to be normalized.

        Returns:
            typing.Any: Returns the normalized input data.
        """

        input_data: numpy.ndarray[typing.Any, typing.Any] = data
        return input_data / input_data.sum(self.axes, keepdims=True)


def main() -> None:
    """The entrypoint to the :py:mod:`memoize_spectral_pipeline` script."""

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

If you would like to contribute, there are multiple ways you can help out. If you find a bug or have a feature request, please feel free to [open an issue on GitHub](https://github.com/virelay/corelay/issues). If you want to contribute code, please [fork the repository](https://github.com/virelay/corelay/fork) and use a feature branch. Pull requests are always welcome. Before forking, please open an issue where you describe what you want to do. This helps to align your ideas with ours and may prevent you from doing work, that we are already planning on doing. If you have contributed to the project, please add yourself to the [contributors list](https://github.com/virelay/corelay/blob/main/CONTRIBUTORS.md).

To help speed up the merging of your pull request, please comment and document your code extensively, try to emulate the coding style of the project, and update the documentation if necessary.

For more information on how to contribute, please refer to our [contributor's guide](https://corelay.readthedocs.io/en/latest/contributors-guide/index.html).

## License

CoRelAy is dual-licensed under the [GNU General Public License Version 3 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.html) or later, and the [GNU Lesser General Public License Version 3 (LGPL-3.0)](https://www.gnu.org/licenses/lgpl-3.0.html) or later. For more information see the [GPL-3.0](https://github.com/virelay/corelay/blob/main/COPYING) and [LGPL-3.0](https://github.com/virelay/corelay/blob/main/COPYING.LESSER) license files.
