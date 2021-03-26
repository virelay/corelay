#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name='corelay',
    use_scm_version=True,
    packages=find_packages(include=['corelay*']),
    install_requires=[
        'h5py>=2.9.0',
        'matplotlib>=3.0.3',
        'numpy>=1.16.3',
        'scikit-learn>=0.20.3',
        'scipy>=1.2.1',
        'Click>=7.0',
        'scikit-image>=0.18.0',
        'metrohash-python>=1.1.3.post2',
    ],
    setup_requires=[
        'setuptools_scm',
    ],
    extras_require={
        'umap': ['umap-learn>=0.3.9'],
        'hdbscan': ['hdbscan>=0.8.22']
    }
)
