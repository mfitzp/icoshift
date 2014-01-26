#!/usr/bin/env python
# coding=utf-8
import sys
from copy import copy
#import distribute_setup
#distribute_setup.use_setuptools()

from setuptools import setup, find_packages

setup(

    name='icoshift',
    version='0.4',
    author='Martin Fitzpatrick',
    author_email='martin.fitzpatrick@gmail.com',
    url='https://github.com/mfitzp/icoshift',
    download_url='https://github.com/mfitzp/icoshift/zipball/master',
    description='icoshift: A versatile tool for the rapid alignment of 1D NMR spectra',
    long_description='Python (numpy+scipy) implementation of icoshift, an open source and highly efficient \
        program designed for solving signal alignment \
        problems in metabonomic NMR data analysis. The icoshift algorithm is based on correlation shifting \
        of spectral intervals and employs an FFT engine that aligns all spectra simultaneously. \
        Translated from MATLAB code using smop and manual adjustments. \
        ',

    packages = ['icoshift'],
    include_package_data = True,
    package_data = {
        '': ['*.txt', '*.rst', '*.md'],
    },
    exclude_package_data = { '': ['README.txt'] },

    install_requires = [
            'numpy>=1.7.1',
            'scipy>=0.12.0',
            ],

    keywords='bioinformatics metabolomics research analysis science',
    license='GPL',
    classifiers=['Development Status :: 4 - Beta',
               'Natural Language :: English',
               'Operating System :: OS Independent',
               'Programming Language :: Python :: 2',
               'License :: OSI Approved :: BSD License',
               'Topic :: Scientific/Engineering :: Bio-Informatics',
               'Intended Audience :: Science/Research',
               'Intended Audience :: Education',
              ],

    )