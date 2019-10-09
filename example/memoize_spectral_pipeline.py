from io import BytesIO

import h5py
import numpy as np

from sprincl.base import Param
from sprincl.processor.base import Processor
from sprincl.processor.flow import Sequential, Parallel
from sprincl.pipeline.spectral import SpectralClustering
from sprincl.processor.clustering import KMeans
from sprincl.processor.embedding import TSNEEmbedding, EigenDecomposition
from sprincl.processor.preprocessing import Histogram
from sprincl.io.storage import HashedHDF5


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
    buf = BytesIO()
    with h5py.File(buf, 'w') as fd:
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
