#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="sprincl",
    version="0.1",
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        'h5py>=2.9.0',
        'matplotlib>=3.0.3',
        'numpy>=1.16.3',
        'scikit-learn>=0.20.3',
        'scipy>=1.2.1',
        'Click>=7.0',
        'bokeh>=1.2.0',
        'scikit-image>=0.15.0',
        'metrohash-python>=1.1.3.post2',
    ],
    entry_points={
        'console_scripts': [
            'sprincl = sprincl.cli:main',
            'vispr = vispr.cli:main',
        ]
    },
    extras_require={
        'umap': ['umap-learn>=0.3.9'],
        'hdbscan': ['hdbscan>=0.8.22']
    }
)
