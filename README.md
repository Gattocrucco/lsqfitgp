[![Documentation Status](https://readthedocs.org/projects/lsqfitgp/badge/?version=latest)](https://lsqfitgp.readthedocs.io/en/latest/?badge=latest)

# lsqfitgp

Module for manipulating gaussian processes. Features:

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

```
pip install lsqfitgp
```

## Documentation

All the code is documented with docstrings, so you can use the Python help
system directly from the shell:

```python
>>> import lsqfitgp as lgp
>>> help(lgp)
>>> help(lgp.something)
```

or, in an IPython shell/Jupyter notebook/Spyder IDE, use the question mark
shortcut:

```python
In [1]: lgp?

In [2]: lgp.something?
```

I'm also writing a manual available on
[readthedocs](https://lsqfitgp.readthedocs.io/en/latest/index.html), but it is
not complete yet. In the meantime, in the directory `examples` there are
various scripts named with single letters (sorry for this nonsense notation).
In an IPython shell, you can run `examples/RUNALL.ipy` to run all the examples
and save the figures on files.

To build the html manual from source, do:

```sh
pip install sphinx
cd docs
python3 kernelsref.py
python3 runcode.py
make html
```

## Tests

The test code is in `tests`. Launch `pytest` in the repository to run all the
tests. `pytest` can be installed with `pip install pytest`.
