# Specifications for storing data

## Legend

- `keys` or `files`
- *shape*
- **data type**
- `<variable>`

- HDF5 structure
  - `key`: **group**, or **dtype** if members have same shape and no identifiers
    - if **group**, `<subkey-variable>`: **dtype** variable subkey without type
    - if **dtype**: *shape* same shape for all members
  - `other-key`: **group** `<clustering>` **dtype** *shape* definition for all group members
    - `specific-member`: properties for some specific member of group

## General Data Specification

- all data is stored on `vca-gpu-211-01` at `<data-root>` = `/data/shared/sprincl/`
- file names with underscores for spaces and dashes as key-separators
- everything is HDF5
- all keys are in singular
- each data instance is one large file
  - input data is only one per dataset and model
  - attribution is one per dataset, model and attribution method
  - analysis is one per dataset, model, attribution method and analysis topic

## Model Input Data

- stored at `<data-root>/<dataset>/<model>.input.h5`
- shape for image data is *samples x channel x height x width*
- since preprocessing depends on the model, we supply a file `<model>.input.h5` with all preprocessing steps applied
- HDF5 structure
  - `data`: **group**, or **float32** if every sample has same dimensions
    - if **group**: `<data-id>` **float32** *channel x height x width* if group, `<data-id>` can be a filename or an identifier
    - if **float32**: **float32** *samples x channel x height x width*
  - `label`: **group**, or **uint** if data is **group**
    - if **group**: `<data-id>` **uint** *1* for single label, or **bool** *classes* for multi label
    - if **uint**: *samples* for single label, **bool** *samples x classes* for multi label if `data` is **float32**
  - `index`: **group** `<data-id>` **uint** *1*, optional
    - if `data` is **group**, assign indices to keys
    - otherwise natural sort order of keys is assumed

## Attribution of Input Data

- stored at `<data-root>/attributions/<dataset>/<model>-<attribution-method>-<attribution-strategy>.attribution.h5`
- `<attribution-strategy>` can be:
  - `true`: for true label
  - `model`: for model prediction
  - `<integer>` for choosing a fixed label
  - `<else>` for something I did not think of
- HDF5 structure
  - `index`: **uint** *samples* indices of attributed input samples in the input file
  - `attribution`: **group** or **float32** attributions with full channel information
    - if **group**: `<data-id>` **float32** *channel x height x width*
    - if **float32**: *samples x channel x height x width*
  - `label` **group** or **float32**, attribution assigned for the model output, governed by `<attribution-strategy>`
    - if **group**: `<data-id>` *samples x {1, <classes>}*
    - if **uint**: *samples x {1, <classes>}*
  - `prediction`: **group** or **float32**
    - if **group**: `<data-id>` **float32** *classes*
    - if **float32**: *samples x classes*

## Analysis Output Data

- stored at `<data-root>/analysis/<dataset>/<model>-<attribution-method>-<attribution-strategy>-<analysis-topic>.analysis.h5`
  - `<analysis-topic>` is an identifier for the analysis approach, e.g. different distance metrics, graph methods etc.
- HDF5 structure
  - `<analysis-identifier>` **group** with name of the analysis as subkeys (not necessarily classes!, wordnet-id for class-wise ImageNet Analysis)
    - `name`: **string** verbose name of analysis
    - `index`: **uint32** *samples* sample indices in the input attribution file
    - `embedding`: **group** `<embedding-id>`
      - `spectral`: **group**
        - `name`: **string**, verbose name of embedding
        - `root`: **float32** *samples x eigenvalues* Eigenvectors of Eigen Decomposition
        - `base`: **link**, if not model input, link to the embedding used
        - `region`: **regionref**, if not model input or not full embedding, regionref to the features used as input
        - `eigenvalue` **float32** *eigenvalues* Eigenvalues for the spectral embedding
      - `tsne`: **group**
        - `name`: **string**, verbose name of embedding
        - `root`: **float32** *samples x 2* t-SNE Embedding
        - `base`: **link**, if not model input, link to the embedding used
        - `region`: **regionref**, if not model input or not full embedding, regionref to the features used as input
    - `clustering`: **group**
      - `<clustering>`: **group** label for clusters on an embedding
        - `name`: **string**, verbose name of embedding
        - `root`: **uint32** *samples* labels for clustering on embedding
        - `base`: **link** link to the embedding used for clustering
        - `region`: **regionref**, if not model input or not full embedding, regionref to the features used as input
        - `#clusters`: **int**, optional if not applying, number of clusters for this clustering
        - `prototype`: **group** multiple prototypes for each cluster
          - `average`: **group** member average prototypes for all clusters
            - `name`: **string**, verbose name of prototype
            - `root`: **float32** *<#clusters> x channel x height x width* prototype payload
