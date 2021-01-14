# Compose Relevance Analysis (CoRelAy)

CoRelAy is a tool to compose small-scale (single-machine) analysis pipelines.
Pipelines are designed with a number of steps (Slots) with default operations (Processors).
Any step of the pipeline may then be indiviually changed by assigning a new operator (Processor).

CoRelAy was created to to quickly implement pipelines to generate analysis data
which can then be visualized using ViRelAy.

## Install

CoRelAy may be installed using pip with
```shell
$ pip install 'git+git://github.com/virelay/corelay'
```

To install optional HDBSCAN and UMAP support, instead use
```shell
$ pip install 'git+git://github.com/virelay/corelay[umap,hdbscan]'
```

## Usage
Examples to highlight some features of CoRelAy can be found in `example/`, or in the following:

```python
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
        data = data / data.sum((1,2), keepdims=True)
        return data


def main():
    np.random.seed(0xDEADBEEF)
    fpath = 'test.analysis.h5'
    with h5py.File(fpath, 'a') as fd:
        # HashedHDF5 is an io-object that stores outputs of Processors based on hashes in hdf5
        iobj = HashedHDF5(fd.require_group('proc_data'))
        data = np.random.normal(size=(64, 3, 32, 32))
        n_clusters = range(2, 20)

        # SpectralClustering is an Example for a pre-defined Pipeline
        pipeline = SpectralClustering(
            # processors, such as EigenDecomposition, can be assigned to pre-defined tasks
            embedding=EigenDecomposition(n_eigval=8, io=iobj),
            # flow-based Processors, such as Parallel, can combine multiple Processors
            clustering=Parallel([
                Parallel([
                    KMeans(n_clusters=k, io=iobj) for k in n_clusters
                ], broadcast=True),
                # io-objects will be used during computation when supplied to Processors
                TSNEEmbedding(io=iobj)
            ], broadcast=True, is_output=True)
        )
        # Processors (and Params) can be updated by simply assigning corresponding attributes
        pipeline.preprocessing = Sequential([
            SumChannel(),
            Normalize(),
            Flatten()
        ]),

        # Processors flagged with "is_output=True" will be accumulated in the output
        output = pipeline(data)

if __name__ == '__main__':
    main()
```
