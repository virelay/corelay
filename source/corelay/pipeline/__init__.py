"""A sub-package that contains pipelines, which perform a set of tasks represented by a special :py:class:`~corelay.plugboard.Slot` type called
:py:class:`~corelay.pipeline.base.Task`. :py:class:`~corelay.pipeline.base.Task` slots can be filled with a special
:py:class:`~corelay.plugboard.Plug` type called :py:class:`~corelay.pipeline.base.TaskPlug`, that ensure that the values they hold are instances of
:py:class:`~corelay.processor.base.Processor`. Furthermore, the sub-package contains :py:class:`~corelay.pipeline.base.Pipeline` implementations for
spectral embeddings, :py:class:`~corelay.pipeline.spectral.SpectralEmbedding`, and spectral clustering,
:py:class:`~corelay.pipeline.spectral.SpectralClustering`. These are specific to
`Spectral Relevance Analysis (SprAy) <https://www.nature.com/articles/s41467-019-08987-4>`_, an explainable artificial intelligence (XAI) method for
bridging the gap between local and global XAI.
"""
