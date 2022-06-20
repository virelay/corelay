#!/usr/bin/env python3
from setuptools import setup, find_packages


with open('README.md', 'r', encoding='utf-8') as fd:
    long_description = fd.read()


setup(
    name='corelay',
    use_scm_version=True,
    author='chrstphr',
    author_email='corelay@j0d.de',
    description='Quickly compose single-machine analysis pipelines.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/virelay/corelay',
    packages=find_packages(where='src', include=['corelay*']),
    package_dir={'': 'src'},
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
        'hdbscan': ['hdbscan>=0.8.22'],
        'docs': [
            'sphinx-copybutton>=0.4.0',
            'sphinx-rtd-theme>=1.0.0',
            'sphinxcontrib.datatemplates>=0.9.0',
            'sphinxcontrib.bibtex>=2.4.1',
        ],
        'tests': [
            'pytest',
            'pytest-cov',
        ],
    },
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ]
)
