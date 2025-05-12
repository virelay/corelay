"""Contains functions, which generate example dataset, attribution, and analysis HDF5 files. These functions are mainly for documentation purposes to
show the structure of the different HDF5 files that can be generated using CoRelAy for the usage in ViRelAy projects.
"""

import h5py
import numpy


def make_group_example() -> None:
    """Generates example dataset, attribution, and analysis HDF5 files were the dataset samples and the attribution data are stored in HDF5 groups
    instead of HDF5 datasets. This is mainly used when the samples (and therefore the corresponding attribution data) do not all have the same shape
    and therefore cannot be stored in a single dataset. Instead they are stored in a group and the mapping between the group keys and the sample
    indices are stored in a separate group called "index".
    """

    # Input file with groups with different sizes
    with h5py.File('grouped.input.h5', 'w') as dataset_file:

        # Input data group
        keys = ('a', 'b', 'c', 'd', 'e')
        shapes = (4, 5, 6, 5, 4)
        channels = 3
        samples_group = dataset_file.require_group('data')
        for key, shape in zip(keys, shapes):

            # Each sample is its own HDF5 dataset
            samples_group[key] = numpy.random.normal(size=(channels, shape, shape)).astype(numpy.float32)

        # True label group
        labels = (0, 1, 0, 0, 1)
        labels_group = dataset_file.require_group('label')
        for key, label in zip(keys, labels):

            # Here each sample is just a single number, alternatively, we could use a 1-dimensional-array of type bool for multi-label data
            labels_group[key] = numpy.uint8(label)

        # We supply a custom ordering of our samples
        indices = (0, 2, 1, 3, 4)
        indices_group = dataset_file.require_group('index')
        for key, index in zip(keys, indices):

            # Each sample has only one index
            indices_group[key] = numpy.uint32(index)

    # Attribution file with groups with different sizes
    with h5py.File('grouped-attr_method-2.attribution.h5', 'w') as attributions_file:

        # We use attribute only subset of our data
        attribution_indices = (2, 3, 4)
        attributions_file['index'] = numpy.array(attribution_indices, dtype=numpy.uint32)

        # Attribution keys of our used subset
        attribution_keys = keys[2:]
        attribution_shapes = shapes[2:]
        attributions_group = attributions_file.require_group('data')
        for key, shape in zip(attribution_keys, attribution_shapes):
            attributions_group[key] = numpy.random.normal(size=(channels, shape, shape)).astype(numpy.float32)

        # Attribution labels are the assigned attribution in the output layer
        labels_group = attributions_file.require_group('label')
        attribution_labels = numpy.array([[0, 1], [0, 1], [0, 1]])
        for key, label in zip(attribution_keys, attribution_labels):

            # The output attributions can be any real number, and have the same shape as the output
            labels_group[key] = attribution_labels.astype(numpy.float32)

        # Predictions are the model output logits
        attribution_predictions = numpy.array([[0, 1], [0.5, 0.5], [1, 0]])
        predictions_group = attributions_file.require_group('prediction')
        for key, prediction in zip(attribution_keys, attribution_predictions):
            predictions_group[key] = prediction.astype(numpy.float32)

    # Analysis file with groups for embeddings and clusterings
    with h5py.File('grouped-attr_method-ana_topic.analysis.h5', 'w') as analysis_file:

        # We call this analysis 'My First Analysis'
        analysis_file['/my_first_analysis/name'] = 'My First Analysis'

        # The used indices of the analysis, here we use all 3 in the attribution file
        analysis_file['/my_first_analysis/index'] = numpy.array(attribution_indices, dtype=numpy.uint32)

        # For shorter references
        embeddings_group = analysis_file.require_group('/my_first_analysis/embedding')
        clusterings_group = analysis_file.require_group('/my_first_analysis/clustering')

        # Verbose name of the spectral embedding
        embeddings_group['/spectral/name'] = 'Spectral Embedding'

        # Spectral embedding (eigenvalue decomposition) with key 'spectral', 2 eigenvalues, here just random data
        number_of_eigenvalues = 2
        embeddings_group['/spectral/root'] = numpy.random.normal(
            size=(len(attribution_indices), number_of_eigenvalues)
        ).astype(numpy.float32)

        # The corresponding eigenvalues, specific to spectral embedding
        embeddings_group['/spectral/eigenvalue'] = numpy.random.normal(size=number_of_eigenvalues).astype(numpy.float32)

        # Verbose name of T-SNE
        embeddings_group['/tsne/name'] = 'T-SNE'

        # T-SNE embedding payload
        embeddings_group['/tsne/root'] = numpy.random.normal(size=(len(attribution_indices), 2)).astype(numpy.float32)

        # This T-SNE embedding is based on the spectral embedding
        embeddings_group['/tsne/base'] = embeddings_group['/spectral']

        # Both feature dimensions of the eigenvectors are used, but for demonstration purpose, we give the regionref
        embeddings_group['/tsne/region'] = embeddings_group['/spectral/root'].regionref[:, [0, 1]]

        # We call our random clustering 'my_clustering'
        clusterings_group['/my_clustering/name'] = 'My Random Clustering'

        # Clustering labels
        clusterings_group['/my_clustering/root'] = numpy.random.randint(0, 2, size=len(attribution_indices))

        # We specify this clustering to be based on 'spectral'
        clusterings_group['/my_clustering/base'] = embeddings_group['/spectral']

        # We use both feature dimensions for the spectral clustering
        clusterings_group['/my_clustering/region'] = embeddings_group['/spectral/root'].regionref[:, [0, 1]]

        # We chose 2 clusters
        clusterings_group['/my_clustering/#clusters'] = 2

        # We define a prototype for our clustering
        clusterings_group['/my_clustering/prototype/average/name'] = 'My Random Prototype'

        # For demonstration purposes, we use random data here. the first dimension is the number of clusters
        clusterings_group['/my_clustering/prototype/average/root'] = numpy.random.normal(size=(2, 32, 32)).astype(numpy.float32)


def make_dataset_example() -> None:
    """Generates example dataset, attribution, and analysis HDF5 files were the dataset samples and the attribution data are stored in HDF5 datasets
    instead of HDF5 groups. This is mainly used when the samples (and therefore the corresponding attribution data) all have the same shape and
    therefore can be stored in a single dataset. Instead of having a separate "index" group which maps the keys to the indices of the samples and
    attribution data, the HDF5 datasets can be directly indexed.
    """

    # Input file with datasets with identical sizes
    number_of_samples = 5
    shape = 7
    channels = 3
    labels = (0, 1, 0)
    with h5py.File('dataset.input.h5', 'w') as dataset_file:

        # Data samples have no identifier here and have implicit indices
        dataset_file['data'] = numpy.random.normal(size=(number_of_samples, channels, shape, shape))
        dataset_file['label'] = numpy.array(labels).astype(numpy.uint8)

    # Attribution file with datasets
    with h5py.File('dataset-attr_method-2.attribution.h5', 'w') as attributions_file:

        # We use attribute only subset of our data
        attribution_indices = (2, 3, 4)
        attributions_file['index'] = numpy.array(attribution_indices, dtype=numpy.uint32)

        # Attribution we only use a subset of our data
        attributions_file['attribution'] = numpy.random.normal(
            size=(len(attribution_indices), channels, shape, shape)
        ).astype(numpy.float32)

        # Attribution labels are the assigned attribution in the output layer
        attribution_labels = numpy.array([[0, 1], [0, 1], [0, 1]])

        # The output attributions can be any real number, and have the same shape as the output
        attributions_file['label'] = attribution_labels.astype(numpy.float32)

        # Predictions are the model output logits
        attribution_predictions = numpy.array([[0, 1], [0.5, 0.5], [1, 0]])
        attributions_file['prediction'] = attribution_predictions.astype(numpy.float32)

    # Using datasets in the input/attribution does not change the analysis file structure
    with h5py.File('dataset-attr_method-ana_topic.analysis.h5', 'w') as analysis_file:

        # We call this analysis 'My First Analysis'
        analysis_file['/my_first_analysis/name'] = 'My First Analysis'

        # The used indices of the analysis, here we use all 3 in the attribution file
        analysis_file['/my_first_analysis/index'] = numpy.array(attribution_indices, dtype=numpy.uint32)

        # For shorter references
        embeddings_group = analysis_file.require_group('/my_first_analysis/embedding')
        clusterings_group = analysis_file.require_group('/my_first_analysis/clustering')

        # Verbose name of the spectral embedding
        embeddings_group['/spectral/name'] = 'Spectral Embedding'

        # Spectral embedding (eigenvalue decomposition) with key 'spectral', 2 eigenvalues, here just random data
        number_of_eigenvalues = 2
        embeddings_group['/spectral/root'] = numpy.random.normal(
            size=(len(attribution_indices), number_of_eigenvalues)
        ).astype(numpy.float32)

        # The corresponding eigenvalues, specific to spectral embedding
        embeddings_group['/spectral/eigenvalue'] = numpy.random.normal(size=number_of_eigenvalues).astype(numpy.float32)

        # Verbose name of T-SNE
        embeddings_group['/tsne/name'] = 'T-SNE'

        # T-SNE embedding payload
        embeddings_group['/tsne/root'] = numpy.random.normal(size=(len(attribution_indices), 2)).astype(numpy.float32)

        # This T-SNE embedding is based on the spectral embedding
        embeddings_group['/tsne/base'] = embeddings_group['/spectral']

        # Both feature dimensions of the eigenvectors are used, but for demonstration purpose, we give the regionref
        embeddings_group['/tsne/region'] = embeddings_group['/spectral/root'].regionref[:, [0, 1]]

        # We call our random clustering 'my_clustering'
        clusterings_group['/my_clustering/name'] = 'My Random Clustering'

        # Clustering labels
        clusterings_group['/my_clustering/root'] = numpy.random.randint(0, 2, size=len(attribution_indices))

        # We specify this clustering to be based on 'spectral'
        clusterings_group['/my_clustering/base'] = embeddings_group['/spectral']

        # We use both feature dimensions for the spectral clustering
        clusterings_group['/my_clustering/region'] = embeddings_group['/spectral/root'].regionref[:, [0, 1]]

        # We chose 2 clusters
        clusterings_group['/my_clustering/#clusters'] = 2

        # We define a prototype for our clustering
        clusterings_group['/my_clustering/prototype/average/name'] = 'My Random Prototype'

        # For demonstration purposes, we use random data here. the first dimension is the number of clusters
        clusterings_group['/my_clustering/prototype/average/root'] = numpy.random.normal(size=(2, 32, 32)).astype(numpy.float32)


def main() -> None:
    """The entrypoint to the :py:mod:`hdf5_structure` script, which generates two sets of sample HDF5 databases, one were the dataset samples and
    their corresponding attributions are stored in HDF5 groups, and one were the dataset samples and their corresponding attributions are stored in
    HDF5 datasets.
    """

    make_group_example()
    make_dataset_example()


if __name__ == '__main__':
    main()
