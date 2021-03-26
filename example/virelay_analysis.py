import json

import h5py
import click
import numpy as np
from skimage.metrics import structural_similarity
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform

from corelay.base import Param
from corelay.processor.base import Processor
from corelay.processor.flow import Sequential, Parallel
from corelay.processor.distance import SciPyPDist
from corelay.pipeline.spectral import SpectralClustering
from corelay.processor.clustering import KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering
from corelay.processor.embedding import TSNEEmbedding, UMAPEmbedding, EigenDecomposition
from corelay.processor.affinity import SparseKNN


class Flatten(Processor):
    def function(self, data):
        return data.reshape(data.shape[0], np.prod(data.shape[1:]))


class SumChannel(Processor):
    def function(self, data):
        return data.sum(1)


class Absolute(Processor):
    def function(self, data):
        return np.absolute(data)


class Normalize(Processor):
    axes = Param(tuple, (1, 2))

    def function(self, data):
        data = data / data.sum(self.axes, keepdims=True)
        return data


def csints(string):
    return tuple(int(elem) for elem in string.split(','))


class Histogram(Processor):
    bins = Param(int, 256)

    def function(self, data):
        hists = np.stack([
            np.stack([
                np.histogram(
                    arr.reshape(arr.shape[0], np.prod(arr.shape[1:3])),
                    bins=self.bins,
                    density=True
                ) for arr in channel
            ]) for channel in data.transpose(3, 0, 1, 2)])
        return hists


class PCC(Processor):
    def function(self, data):
        return squareform(pdist(data, metric=lambda x, y: pearsonr(x, y)[0]))


class SSIM(Processor):
    def function(self, data):
        N, H, W = data.shape
        return squareform(pdist(
            data.reshape(N, H * W),
            metric=lambda x, y: structural_similarity(x.reshape(H, W), y.reshape(H, W), multichannel=False)
        ))


VARIANTS = {
    'absspectral': {
        'preprocessing': Sequential([
            Absolute(),
            SumChannel(),
            Normalize(),
            Flatten()
        ]),
        'distance': SciPyPDist(metric='euclidean'),
    },
    'spectral': {
        'preprocessing': Sequential([
            SumChannel(),
            Normalize(),
            Flatten()
        ]),
        'distance': SciPyPDist(metric='euclidean'),
    },
    'fullspectral': {
        'preprocessing': Sequential([
            Normalize(axes=(1, 2, 3)),
            Flatten()
        ]),
        'distance': SciPyPDist(metric='euclidean'),
    },
    'histogram': {
        'preprocessing': Sequential([
            Normalize(axes=(1, 2, 3)),
            Histogram(),
            Flatten()
        ]),
        'distance': SciPyPDist(metric='euclidean'),
    },
    'ssim': {
        'preprocessing': Sequential([
            SumChannel(),
            Normalize(),
        ]),
        'distance': SSIM(),
    },
    'pcc': {
        'preprocessing': Sequential([
            SumChannel(),
            Normalize(),
            Flatten()
        ]),
        'distance': PCC(),
    }
}


@click.command()
@click.argument('attribution-file', type=click.Path())
@click.argument('analysis-file', type=click.Path())
@click.option('--n-clusters', type=csints, default=','.join(str(elem) for elem in range(2, 31)))
@click.option('--class-indices', type=csints)
@click.option('--label-map', 'label_map_file', type=click.Path())
@click.option('--variant', type=click.Choice(list(VARIANTS)), default='spectral')
@click.option('--n-eigval', type=int, default=32)
@click.option('--n-neighbors', type=int, default=32)
def main(variant, attribution_file, analysis_file, class_indices, label_map_file, n_eigval, n_clusters, n_neighbors):
    preprocessing = VARIANTS[variant]['preprocessing']
    distance = VARIANTS[variant]['distance']

    pipeline = SpectralClustering(
        preprocessing=preprocessing,
        pairwise_distance=distance,
        affinity=SparseKNN(n_neighbors=n_neighbors, symmetric=True),
        embedding=EigenDecomposition(n_eigval=n_eigval, is_output=True),
        clustering=Parallel([
            Parallel([
                KMeans(n_clusters=k) for k in n_clusters
            ], broadcast=True),
            Parallel([
                DBSCAN(eps=k / 10.) for k in n_clusters
            ], broadcast=True),
            HDBSCAN(),
            Parallel([
                AgglomerativeClustering(n_clusters=k) for k in n_clusters
            ], broadcast=True),
            Parallel([
                UMAPEmbedding(),
                TSNEEmbedding(),
            ], broadcast=True)
        ], broadcast=True, is_output=True)
    )

    if label_map_file is not None:
        with open(label_map_file, 'r') as fp:
            label_map = json.load(fp)
        label_map = {elem['index']: elem['word_net_id'] for elem in label_map}
    else:
        label_map = {}

    with h5py.File(attribution_file, 'r') as fp:
        label = fp['label'][:]

    for class_index in class_indices:
        print('Loading class {:03d}'.format(class_index))
        with h5py.File(attribution_file, 'r') as fp:
            index, = np.nonzero(label == class_index)
            data = fp['attribution'][index, :]
            if 'train' in fp:
                train_flag = fp['train'][index.tolist()]
            else:
                train_flag = None

        print('Computing class {:03d}'.format(class_index))
        (eigenvalues, embedding), (kmeans, dbscan, hdbscan, agglo, (umap, tsne)) = pipeline(data)

        print('Saving class {:03d}'.format(class_index))
        with h5py.File(analysis_file, 'a') as fp:
            analysis_name = label_map.get(class_index, '{:03d}'.format(class_index))
            g_analysis = fp.require_group(analysis_name)
            g_analysis['index'] = index.astype('uint32')

            g_embedding = g_analysis.require_group('embedding')
            g_embedding['spectral'] = embedding.astype('float32')
            g_embedding['spectral'].attrs['eigenvalue'] = eigenvalues.astype('float32')

            g_embedding['tsne'] = tsne.astype('float32')
            g_embedding['tsne'].attrs['embedding'] = 'spectral'
            g_embedding['tsne'].attrs['index'] = np.array([0, 1])

            g_embedding['umap'] = umap.astype('float32')
            g_embedding['umap'].attrs['embedding'] = 'spectral'
            g_embedding['umap'].attrs['index'] = np.array([0, 1])

            g_cluster = g_analysis.require_group('cluster')
            for n_cluster, clustering in zip(n_clusters, kmeans):
                s_cluster = 'kmeans-{:02d}'.format(n_cluster)
                g_cluster[s_cluster] = clustering
                g_cluster[s_cluster].attrs['embedding'] = 'spectral'
                g_cluster[s_cluster].attrs['k'] = n_cluster
                g_cluster[s_cluster].attrs['index'] = np.arange(embedding.shape[1], dtype='uint32')

            for n_cluster, clustering in zip(n_clusters, dbscan):
                s_cluster = 'dbscan-eps={:.1f}'.format(n_cluster / 10.)
                g_cluster[s_cluster] = clustering
                g_cluster[s_cluster].attrs['embedding'] = 'spectral'
                g_cluster[s_cluster].attrs['index'] = np.arange(embedding.shape[1], dtype='uint32')

            s_cluster = 'hdbscan'
            g_cluster[s_cluster] = hdbscan
            g_cluster[s_cluster].attrs['embedding'] = 'spectral'
            g_cluster[s_cluster].attrs['index'] = np.arange(embedding.shape[1], dtype='uint32')

            for n_cluster, clustering in zip(n_clusters, agglo):
                s_cluster = 'agglomerative-{:02d}'.format(n_cluster)
                g_cluster[s_cluster] = clustering
                g_cluster[s_cluster].attrs['embedding'] = 'spectral'
                g_cluster[s_cluster].attrs['k'] = n_cluster
                g_cluster[s_cluster].attrs['index'] = np.arange(embedding.shape[1], dtype='uint32')

            if train_flag is not None:
                g_cluster['train_split'] = train_flag


if __name__ == '__main__':
    main()
