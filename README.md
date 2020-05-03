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

### Building the manual from source

```sh
pip install sphinx<2
cd docs
make html
```

If you add kernels, run `kernelsref.py` to regenerate `kernelsref.rst`.

If you add a documentation page with code examples, use `runcode.py` to run
all the code found in code sections in the rst file.

## Examples

In the directory `examples` there are various scripts named with single letters
(sorry for this nonsense notation). In an IPython shell, you can run
`examples/RUNALL.ipy` to run all the examples and save the figures on files.

## Tests

The test code is in `tests`. Launch `pytest` in the repository to run all the
tests. `pytest` can be installed with `pip install pytest`.
