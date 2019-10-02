import h5py
import numpy as np


def make_group_example():
    # input file with groups with different sizes
    with h5py.File('grouped.input.h5', 'w') as fd:
        # input data group
        keys = ('a', 'b', 'c', 'd', 'e')
        shapes = (4, 5, 6, 5, 4)
        channels = 3
        g_input = fd.require_group('data')
        for key, shape in zip(keys, shapes):
            # each sample is its own dataset
            g_input[key] = np.random.normal(size=(channels, shape, shape)).astype(np.float32)

        # true label group
        labels = (0, 1, 0, 0, 1)
        g_label = fd.require_group('label')
        for key, label in zip(keys, labels):
            # here each sample is just a single number
            # alternatively, we could use a 1d-array of type bool for multi-label data
            g_label[key] = np.uint8(label)

        # we supply a custom ordering of our samples
        indices = (0, 2, 1, 3, 4)
        g_index = fd.require_group('index')
        for key, index in zip(keys, indices):
            # each sample has only one index
            g_index[key] = np.uint32(index)

    # attribution file with groups with different sizes
    with h5py.File('grouped-attr_method-2.attribution.h5', 'w') as fd:
        # we use attribute only subset of our data
        a_indices = (2, 3, 4)
        fd['index'] = np.array(a_indices, dtype=np.uint32)

        # attribution keys of our used subset
        a_keys = keys[2:]
        a_shapes = shapes[2:]
        g_attribution = fd.require_group('data')
        for key, shape in zip(a_keys, a_shapes):
            g_attribution[key] = np.random.normal(size=(channels, shape, shape)).astype(np.float32)

        # attribution labels are the assigned attribution in the output layer
        g_label = fd.require_group('label')
        a_labels = np.array([[0, 1], [0, 1], [0, 1]])
        for key, label in zip(a_keys, a_labels):
            # the output attributions can be any real number, and have the same shape as the output
            g_label[key] = a_labels.astype(np.float32)

        # predictions are the model output logits
        a_predictions = np.array([[0, 1], [.5, .5], [1, 0]])
        g_prediction = fd.require_group('prediction')
        for key, pred in zip(a_keys, a_predictions):
            g_prediction[key] = a_predictions.astype(np.float32)

    with h5py.File('grouped-attr_method-ana_topic.analysis.h5', 'w') as fd:
        # we call this analysis 'My First Analysis'
        fd['/my_first_analysis/name'] = 'My First Analysis'
        # the used indices of the analysis, here we use all 3 in the attribution file
        fd['/my_first_analysis/index'] = np.array(a_indices, dtype=np.uint32)

        # for shorter references
        g_emb = fd.require_group('/my_first_analysis/embedding')
        g_clu = fd.require_group('/my_first_analysis/clustering')

        n_eigvals = 2
        # verbose name of the spectral embedding
        g_emb['/spectral/name'] = 'Spectral Embedding'
        # spectral embedding (eigenvalue decomposition) with key 'spectral', 2 eigenvalues, here just random data
        g_emb['/spectral/root'] = np.random.normal(size=(len(a_indices), n_eigvals)).astype(np.float32)
        # the corresponding eigenvalues, specific to spectral embedding
        g_emb['/spectral/eigenvalue'] = np.random.normal(size=n_eigvals).astype(np.float32)

        # verbose name of T-SNE
        g_emb['/tsne/name'] = 'T-SNE'
        # T-SNE embedding payload
        g_emb['/tsne/root'] = np.random.normal(size=(len(a_indices), 2)).astype(np.float32)
        # this T-SNE embedding is based on the spectral embedding
        g_emb['/tsne/base'] = g_emb['/spectral']
        # both feature dimensions of the eigenvectors are used, but for demonstration purpose, we give the regionref
        g_emb['/tsne/region'] = g_emb['/spectral/root'].regionref[:, [0, 1]]

        # we call our random clustering 'my_clustering'
        g_clu['/my_clustering/name'] = 'My Random Clustering'
        # clustering labels
        g_clu['/my_clustering/root'] = np.random.randint(0, 2, size=len(a_indices))
        # we specify this clustering to be based on 'spectral'
        g_clu['/my_clustering/base'] = g_emb['/spectral']
        # we use both feature dimensions for the spectral clustering
        g_clu['/my_clustering/region'] = g_emb['/spectral/root'].regionref[:, [0, 1]]
        # we chose 2 clusters
        g_clu['/my_clustering/#clusters'] = 2

        # we define a protoype for our clustering
        g_clu['/my_clustering/prototype/average/name'] = 'My Random Prototype'
        # for demonstration purposes, we use random data here. the first dimonsion is the number of clusters
        g_clu['/my_clustering/prototype/average/root'] = np.random.normal(size=(2, 32, 32)).astype(np.float32)


def make_dataset_example():
    # input file with datasets with identical sizes
    nsamples = 5
    shape = 7
    channels = 3
    labels = (0, 1, 0)
    with h5py.File('dataset.input.h5', 'w') as fd:
        # data samples have no identifier here and have implicit indices
        fd['data'] = np.random.normal(size=(nsamples, channels, shape, shape))
        fd['label'] = np.array(labels).astype(np.uint8)

    # attribution file with datasets
    with h5py.File('dataset-attr_method-2.attribution.h5', 'w') as fd:
        # we use attribute only subset of our data
        a_indices = (2, 3, 4)
        fd['index'] = np.array(a_indices, dtype=np.uint32)

        # attribution we only use a subset of our data
        fd['attribution'] = np.random.normal(size=(len(a_indices), channels, shape, shape)).astype(np.float32)

        # attribution labels are the assigned attribution in the output layer
        a_labels = np.array([[0, 1], [0, 1], [0, 1]])
        # the output attributions can be any real number, and have the same shape as the output
        fd['label'] = a_labels.astype(np.float32)

        # predictions are the model output logits
        a_predictions = np.array([[0, 1], [.5, .5], [1, 0]])
        fd['prediction'] = a_predictions.astype(np.float32)

    # using datasets in the input/attribution does not change the analysis file structure
    with h5py.File('dataset-attr_method-ana_topic.analysis.h5', 'w') as fd:
        # we call this analysis 'My First Analysis'
        fd['/my_first_analysis/name'] = 'My First Analysis'
        # the used indices of the analysis, here we use all 3 in the attribution file
        fd['/my_first_analysis/index'] = np.array(a_indices, dtype=np.uint32)

        # for shorter references
        g_emb = fd.require_group('/my_first_analysis/embedding')
        g_clu = fd.require_group('/my_first_analysis/clustering')

        n_eigvals = 2
        # verbose name of the spectral embedding
        g_emb['/spectral/name'] = 'Spectral Embedding'
        # spectral embedding (eigenvalue decomposition) with key 'spectral', 2 eigenvalues, here just random data
        g_emb['/spectral/root'] = np.random.normal(size=(len(a_indices), n_eigvals)).astype(np.float32)
        # the corresponding eigenvalues, specific to spectral embedding
        g_emb['/spectral/eigenvalue'] = np.random.normal(size=n_eigvals).astype(np.float32)

        # verbose name of T-SNE
        g_emb['/tsne/name'] = 'T-SNE'
        # T-SNE embedding payload
        g_emb['/tsne/root'] = np.random.normal(size=(len(a_indices), 2)).astype(np.float32)
        # this T-SNE embedding is based on the spectral embedding
        g_emb['/tsne/base'] = g_emb['/spectral']
        # both feature dimensions of the eigenvectors are used, but for demonstration purpose, we give the regionref
        g_emb['/tsne/region'] = g_emb['/spectral/root'].regionref[:, [0, 1]]

        # we call our random clustering 'my_clustering'
        g_clu['/my_clustering/name'] = 'My Random Clustering'
        # clustering labels
        g_clu['/my_clustering/root'] = np.random.randint(0, 2, size=len(a_indices))
        # we specify this clustering to be based on 'spectral'
        g_clu['/my_clustering/base'] = g_emb['/spectral']
        # we use both feature dimensions for the spectral clustering
        g_clu['/my_clustering/region'] = g_emb['/spectral/root'].regionref[:, [0, 1]]
        # we chose 2 clusters
        g_clu['/my_clustering/#clusters'] = 2

        # we define a protoype for our clustering
        g_clu['/my_clustering/prototype/average/name'] = 'My Random Prototype'
        # for demonstration purposes, we use random data here. the first dimonsion is the number of clusters
        g_clu['/my_clustering/prototype/average/root'] = np.random.normal(size=(2, 32, 32)).astype(np.float32)


def main():
    make_group_example()
    make_dataset_example()


if __name__ == '__main__':
    main()
