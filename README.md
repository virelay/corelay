# CoRelAy &ndash; Composing Relevance Analysis

![CoRelAy Logo](docs/images/corelay-logo.png)

[![Documentation Status](https://readthedocs.org/projects/corelay/badge/?version=latest)](https://corelay.readthedocs.io/en/latest/?badge=latest)
[![tests](https://github.com/virelay/corelay/actions/workflows/tests.yml/badge.svg)](https://github.com/virelay/corelay/actions/workflows/tests.yml)
[![PyPI Version](https://img.shields.io/pypi/v/corelay)](https://pypi.org/project/corelay/)
[![License](https://img.shields.io/pypi/l/corelay)](https://github.com/virelay/corelay/blob/master/COPYING.LESSER)

CoRelAy is a tool to compose small-scale (single-machine) analysis pipelines.
Pipelines are designed with a number of steps (Task) with default operations (Processor).
Any step of the pipeline may then be indiviually changed by assigning a new operator (Processor).
Processors have Params which define their operation.

CoRelAy was created to quickly implement pipelines to generate analysis data
which can then be visualized using ViRelAy.

If you find CoRelAy useful for your research, why not cite our related [paper](https://arxiv.org/abs/2106.13200):
```
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
The latest documentation is hosted at
[corelay.readthedocs.io](https://corelay.readthedocs.io/en/latest/).


## Install
CoRelAy may be installed using pip with
```shell
$ pip install corelay
```

To install optional HDBSCAN and UMAP support, use
```shell
$ pip install corelay[umap,hdbscan]
```

## Usage
Examples to highlight some features of **CoRelAy** can be found in `example/`.

We mainly use HDF5 files to store results. The structure used by **ViRelAy** is documented in the **ViRelAy**
repository at `docs/database_specification.md`. An example to create HDF5 files which can be used with **ViRelAy** is
shown in `example/hdf5_structure.py`

To do a full SpRAy analysis which can be visualized with **ViRelAy**, an advanced script can be found in
`example/virelay_analysis.py`.

The following shows the contents of `example/memoize_spectral_pipeline.py`:

```python
'''Example using memoization to store (intermediate) results.'''
import time

import h5py
import numpy as np

from corelay.base import Param
from corelay.processor.base import Processor
from corelay.processor.flow import Sequential, Parallel
from corelay.pipeline.spectral import SpectralClustering
from corelay.processor.clustering import KMeans
from corelay.processor.embedding import TSNEEmbedding, EigenDecomposition
from corelay.io.storage import HashedHDF5


# custom processors can be implemented by defining a function attribute
class Flatten(Processor):
    def function(self, data):
        return data.reshape(data.shape[0], np.prod(data.shape[1:]))


class SumChannel(Processor):
    # parameters can be assigned by defining a class-owned Param instance
    axis = Param(int, 1)
    def function(self, data):
        return data.sum(1)


class Normalize(Processor):
    def function(self, data):
        data = data / data.sum((1, 2), keepdims=True)
        return data


def main():
    np.random.seed(0xDEADBEEF)
    fpath = 'test.analysis.h5'
    with h5py.File(fpath, 'a') as fd:
        # HashedHDF5 is an io-object that stores outputs of Processors based on hashes in hdf5
        iobj = HashedHDF5(fd.require_group('proc_data'))

        # generate some exemplary data
        data = np.random.normal(size=(64, 3, 32, 32))
        n_clusters = range(2, 20)

        # SpectralClustering is an Example for a pre-defined Pipeline
        pipeline = SpectralClustering(
            # processors, such as EigenDecomposition, can be assigned to pre-defined tasks
            embedding=EigenDecomposition(n_eigval=8, io=iobj),
            # flow-based Processors, such as Parallel, can combine multiple Processors
            # broadcast=True copies the input as many times as there are Processors
            # broadcast=False instead attempts to match each input to a Processor
            clustering=Parallel([
                Parallel([
                    KMeans(n_clusters=k, io=iobj) for k in n_clusters
                ], broadcast=True),
                # io-objects will be used during computation when supplied to Processors
                # if a corresponding output value (here identified by hashes) already exists,
                # the value is not computed again but instead loaded from the io object
                TSNEEmbedding(io=iobj)
            ], broadcast=True, is_output=True)
        )
        # Processors (and Params) can be updated by simply assigning corresponding attributes
        pipeline.preprocessing = Sequential([
            SumChannel(),
            Normalize(),
            Flatten()
        ])

        start_time = time.perf_counter()

        # Processors flagged with "is_output=True" will be accumulated in the output
        # the output will be a tree of tuples, with the same hierachy as the pipeline
        # (i.e. clusterings here contains a tuple of the k-means outputs)
        clusterings, tsne = pipeline(data)

        # since we memoize our results in a hdf5 file, subsequent calls will not compute
        # the values (for the same inputs), but rather load them from the hdf5 file
        # try running the script multiple times
        duration = time.perf_counter() - start_time
        print(f'Pipeline execution time: {duration:.4f} seconds')


if __name__ == '__main__':
    main()
```
