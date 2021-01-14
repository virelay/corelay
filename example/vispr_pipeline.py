import os

import click

from corelay import io
from corelay.pipeline.spectral import SpectralEmbedding
from corelay.processor.base import FunctionProcessor
from corelay.processor.affinity import SparseKNN
from corelay.processor.distance import SciPyPDist
from corelay.processor.embedding import EigenDecomposition, TSNEEmbedding
from corelay.processor.laplacian import SymmetricNormalLaplacian
from corelay.processor.clustering import KMeans


@click.command()
@click.argument('attribution_path')
@click.argument('analysis_path')
def main(attribution_path, analysis_path):
    os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
    with io.HDF5Storage(analysis_path, mode='w') as analysis_file, \
            io.HDF5Storage(attribution_path, mode='r') as attribution_file:

        analysis_file['index'] = attribution_file['index']
        pipeline = SpectralEmbedding(
            preprocessing=FunctionProcessor(function=lambda x: x.mean(1).reshape(x.shape[0], -1),
                                            bind_method=False),
            pairwise_distance=SciPyPDist(metric='euclidean'),
            affinity=SparseKNN(n_neighbors=10, symmetric=True),
            laplacian=SymmetricNormalLaplacian(),
            embedding=EigenDecomposition(n_eigval=32, io=analysis_file.at(data_key='embedding/spectral')),
        )

        eig_val, eig_vec = pipeline(attribution_file['attribution'])
        TSNEEmbedding(io=analysis_file.at(data_key='embedding/tsne'))(eig_vec)
        for n_clusters in range(2, 33):
            KMeans(n_clusters=n_clusters, io=analysis_file.at(data_key='cluster/kmeans-{}'.format(n_clusters)))(eig_vec)


if __name__ == '__main__':
    main()
