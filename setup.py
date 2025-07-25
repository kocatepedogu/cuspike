# SPDX-FileCopyrightText: 2025 Doğu Kocatepe
# SPDX-License-Identifier: GPL-3.0-or-later

from setuptools import setup, find_packages

VERSION = '0.1'
DESCRIPTION = 'cuSpike Python API'
LONG_DESCRIPTION = 'A high-performance spiking neural network simulator for CUDA with partial support for arbitrary neuron models'

setup(
        name="cuspike",
        version=VERSION,
        author="Dogu Kocatepe",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        python_requires=">=3.12",
        install_requires=['numpy', 'scikit-learn==1.6.1', 'scipy==1.15.2']
)
