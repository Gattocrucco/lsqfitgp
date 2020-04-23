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
    python_requires='>=3.1',
    install_requires=[
        'numpy',
        'scipy',
        'autograd',
        'gvar'
    ]
)
