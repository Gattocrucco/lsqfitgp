[![Documentation Status](https://readthedocs.org/projects/lsqfitgp/badge/?version=latest)](https://lsqfitgp.readthedocs.io/en/latest/?badge=latest)
![PyPI](https://img.shields.io/pypi/v/lsqfitgp)

# lsqfitgp

Python module for manipulating gaussian processes. Features:

  * Use [gvar](https://github.com/gplepage/gvar) to keep track transparently of
    correlations between prior, data and posterior.

  * Fit a latent gaussian process in a nonlinear model with
    [lsqfit](https://github.com/gplepage/lsqfit).
    
  * [autograd](https://github.com/HIPS/autograd)-friendly.
  
  * Supports multidimensional structured non-numerical input with named
    dimensions.
    
  * Apply arbitrary linear transformations to the process.
  
  * Use dictionaries to manipulate hyperparameters and hyperpriors. Use
    `gvar.BufferDict` to transparently apply transformations.
    
  * Get a covariance matrix for the optimized hyperparameters.
  
## Installation

Python >= 3.6 required. Then:

```
$ pip install lsqfitgp
```

## Documentation

The manual is available on
[readthedocs](https://lsqfitgp.readthedocs.io/en/latest/index.html). All the
code is documented with docstrings, so you can also use the Python help system
directly from the shell:

```python
>>> import lsqfitgp as lgp
>>> help(lgp)
>>> help(lgp.something)
```

or, in an IPython shell/Jupyter notebook/Spyder IDE, use the question mark
shortcut:

```
In [1]: lgp?

In [2]: lgp.something?
```

## Development

Create a virtual environment and install the requirements:

```sh
$ python -m venv myenv
$ . myenv/bin/activate
(myenv) $ pip install -r requirements.txt
```

The `Makefile` in the root directory contains targets to build the
documentation, run the tests, and prepare a release. Run `make` without
arguments to show the available targets:

```sh
$ make
available targets: upload release tests examples docscode docs
release = tests examples docscode docs (in order)
$ make tests # or make examples, or ...
```
