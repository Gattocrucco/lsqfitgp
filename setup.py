# lsqfitgp/setup.py
#
# Copyright (c) 2020, 2022, Giacomo Petrillo
#
# This file is part of lsqfitgp.
#
# lsqfitgp is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# lsqfitgp is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with lsqfitgp.  If not, see <http://www.gnu.org/licenses/>.

import setuptools
import lsqfitgp

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lsqfitgp",
    version=lsqfitgp.__version__,
    author="Giacomo Petrillo",
    author_email="info@giacomopetrillo.com",
    description="Gaussian processes in nonlinear least-squares fits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Gattocrucco/lsqfitgp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'autograd',
        'gvar'
    ]
)
