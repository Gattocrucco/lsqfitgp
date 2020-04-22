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

I've still not packaged the program, so copy the directory `lsqfitgp` in the
same directory of the python script that needs to import it, or use the
environment variable `PYTHONPATH` or `sys.path` to tell Python where you have
placed `lsqfitgp`.

## Examples

In the directory `examples` there are various scripts named with single letters
(sorry for this nonsense notation). In an IPython shell, you can run
`examples/RUNALL.ipy` to run all the examples and save the figures on files.

## Documentation

There's no manual, only docstrings in the code.

## Tests

The test code is in `test`. Launch `pytest` in the repository to run all the
tests. `pytest` can be installed with `pip install pytest`.
