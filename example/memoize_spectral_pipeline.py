from io import BytesIO

import h5py
import numpy as np

from corelay.base import Param
from corelay.processor.base import Processor
from corelay.processor.flow import Sequential, Parallel
from corelay.pipeline.spectral import SpectralClustering
from corelay.processor.clustering import KMeans
from corelay.processor.embedding import TSNEEmbedding, EigenDecomposition
from corelay.processor.preprocessing import Histogram
from corelay.io.storage import HashedHDF5


class Flatten(Processor):
    def function(self, data):
        return data.reshape(data.shape[0], np.prod(data.shape[1:]))


class SumChannel(Processor):
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
        iobj = HashedHDF5(fd.require_group('proc_data'))

        data = np.random.normal(size=(64, 3, 32, 32))

        n_clusters = range(2, 20)

        pipeline = SpectralClustering(
            preprocessing=Sequential([
                SumChannel(),
                Normalize(),
                Flatten()
            ]),
            embedding=EigenDecomposition(n_eigval=8, io=iobj),
            clustering=Parallel([
                Parallel([
                    KMeans(n_clusters=k, io=iobj) for k in n_clusters
                ], broadcast=True),
                TSNEEmbedding(io=iobj)
            ], broadcast=True, is_output=True)
        )

        output = pipeline(data)

if __name__ == '__main__':
    main()
