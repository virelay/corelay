#!/usr/bin/env python
import logging
import os

import h5py
import numpy as np

from os import path
from argparse import ArgumentParser

from sklearn.manifold import TSNE
from sklearn.cluster import k_means

from .core import spectral_clustering
from .visualize import spray_visualize

logger = logging.getLogger(__name__)

def main():
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('attribution')
    parser.add_argument('output')
    parser.add_argument('-w', '--overwrite', action='store_true')
    parser.add_argument('--relabel', action='store_true')
    parser.add_argument('--retsne', action='store_true')
    parser.add_argument('--nneighbours', type=int, default=10)
    parser.add_argument('--nevals', type=int, default=10)
    parser.add_argument('--nclusters', type=int, default=10)
    args = parser.parse_args()

    with h5py.File(args.attribution, 'r') as fd:
        data  = fd['attribution'][:]
        label = fd['label'][:]

    data  = data.mean(1)
    shape = data.shape

    fname = args.output
    os.makedirs(path.dirname(fname), exist_ok=True)

    data = data.reshape(shape[0], np.prod(shape[1:]))
    if not path.exists(fname) or args.overwrite:
        logger.info('Computing {}'.format(fname))
        ew, ev, label = spectral_clustering(data, args.nneighbours, args.nevals, args.nclusters)
        tsne = TSNE().fit_transform(ev)
        with h5py.File(fname, 'w') as fd:
            fd['ew'] = ew
            fd['ev'] = ev
            fd['label'] = label
            fd['tsne'] = tsne
    else:
        logger.info('Loading {}'.format(fname))
        with h5py.File(fname, 'r') as fd:
            ew = fd['ew'][:]
            ev = fd['ev'][:]
            label = fd['label'][:]
            tsne = fd['tsne'][:]
        if args.relabel:
            _, label, _ = k_means(ev[:, -args.nevals:], args.nclusters)
        if args.retsne:
            tsne = TSNE().fit_transform(ev[:, -args.nevals:])

    belongs = (label[None] == np.arange(args.nevals)[:, None]).sum(1)
    logger.info('Samples in clusters: {}'.format(", ".join([str(n) for n in belongs])))

    logger.info('Visualizing {}'.format(fname))
    spray_visualize(data, ew, ev, label, tsne, args.output, args.nclusters, shape[1:])


if __name__ == '__main__':
    main()
