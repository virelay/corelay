=====================
CoRelAy Documentation
=====================

.. image:: ../../design/corelay-logo-with-title.png
    :alt: CoRelAy Logo

Welcome to the documentation of CoRelAy, a library designed for composing efficient, single-machine data analysis pipelines. CoRelAy enables the rapid implementation of pipelines that can be used to analyze and process data. CoRelAy is primarily meant for the use in explainable artificial intelligence (XAI), often with the goal of producing output suitable for visualization in tools like `ViRelAy <https://github.com/virelay/virelay>`_.

At the core of CoRelAy are **pipelines**, which consist of a series of **tasks**. Each task is a modular unit that can be populated with operations to perform specific data processing tasks. These operations, known as **processors**, can be customized by assigning new processor instances to the tasks.

Tasks in CoRelAy are highly flexible and can be tailored to meet the needs of your analysis pipeline. By leveraging a wide range of built-in configurable processors with their respective **parameters**, you can easily adapt and optimize your data processing workflow.

.. note::

    If you come from a previous version of CoRelAy before the 1.0.0 release, please refer to the :doc:`migration guide <migration-guide/migrating-from-v0.2-to-v1.0>` for information on how to transition to the latest version. Some breaking changes have been introduced, and the migration guide will help you adapt your existing code to the new version.

Contents
========

This documentation is organized into four main sections:

* **Getting Started** -- In-depth information about all features of CoRelAy, including installation instructions and usage examples.
* **Migration Guide** -- A guide for users of previous versions of CoRelAy to help them transition to the latest version.
* **Contributors Guide** -- Guidelines for contributing to the project, from reporting bugs and creating feature requests to contributing to the code base and documentation.
* **API Reference** -- Detailed descriptions of the modules, classes, methods, and functions included in CoRelAy.

.. toctree::
    :maxdepth: 2

    getting-started/index
    migration-guide/index
    contributors-guide/index
    api-reference/index
    bibliography

Indices
=======

* :ref:`genindex`
* :ref:`modindex`

Citing
======

We encourage you to cite our related paper :cite:p:`anders2021software` if CoRelAy has been useful for your research. To make it easier, we've included the relevant citation information below.

.. code-block:: bibtex

    @article{anders2021software,
      author  = {Anders, Christopher J. and
                 Neumann, David and
                 Samek, Wojciech and
                 Müller, Klaus-Robert and
                 Lapuschkin, Sebastian},
      title   = {Software for Dataset-wide XAI: From Local Explanations to Global Insights with {Zennit}, {CoRelAy}, and {ViRelAy}},
      year    = {2021},
      volume  = {abs/2106.13200},
      journal = {CoRR}
    }
