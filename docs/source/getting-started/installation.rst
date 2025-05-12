============
Installation
============

To get started, you first have to install CoRelAy on your system. The recommended and easiest way to install CoRelAy is to use ``pip``, the Python Package Manager, which can install packages from the `Python package index (PyPI) <https://pypi.org/>`_:

.. code-block:: console

   $ pip install corelay

.. note::

   CoRelAy depends on the `metrohash-python <https://pypi.org/project/metrohash-python/>`_ library, which requires a C++ compiler to be installed. This may mean that you will have to install extra packages (GCC or Clang) for the installation to succeed. For example, on Fedora, you may have to install the ``gcc-c++`` package in order to make the ``c++`` command available, which can be done using the following command:

   .. code-block:: console

      $ sudo dnf install gcc-c++

To install CoRelAy with optional HDBSCAN and UMAP support, you can use the following command:

.. code-block:: console

    $ pip install corelay[umap,hdbscan]

If you'd like to try out the bleeding-edge development version or experiment with the included examples, you can also clone the Git repository and install it manually. The project uses the Python package and project manager `uv <https://github.com/astral-sh/uv>`_. You can find instructions on how to install and use ``uv`` in the `official documentation <https://docs.astral.sh/uv/>`_. To install and use the development version of CoRelAy, you can use the following commands:

.. code-block:: console

   $ git clone https://github.com/virelay/corelay.git
   $ cd corelay
   $ uv --directory source python install
   $ uv --directory source sync --all-extras
