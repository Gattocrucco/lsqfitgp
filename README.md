[![PyPI](https://img.shields.io/pypi/v/lsqfitgp)](https://pypi.org/project/lsqfitgp/)

# lsqfitgp

Python module for manipulating Gaussian processes. Features:

  * Use [gvar](https://github.com/gplepage/gvar) to keep track transparently of
    dependencies between prior, data and posterior.
  * Fit a latent Gaussian process in a nonlinear model with
    [lsqfit](https://github.com/gplepage/lsqfit).
  * [JAX](https://github.com/google/jax)-friendly.
  * Supports multidimensional structured non-numerical input with named
    dimensions.
  * Apply arbitrary linear transformations to the process, finite and
    infinite.
  * Use dictionaries to manipulate hyperparameters and their priors. Use
    `gvar.BufferDict` to transparently apply transformations to implement
    Gaussian copulas.
  * Get a covariance matrix for the optimized hyperparameters.
  * Many covariance functions, including one for BART (Bayesian Additive
    Regression Trees).
  
See [this report](https://www.giacomopetrillo.com/scuola/gppdf.pdf) for the
theory behind lsqfitgp.

## Installation

Python >= 3.8 required. Then:

```
$ pip install lsqfitgp
```

## Documentation

The complete manual is available online at
[gattocrucco.github.io/lsqfitgp/docs](https://gattocrucco.github.io/lsqfitgp/docs).
All the code is documented with docstrings, so you can also use the Python help
system directly from the shell:

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

To access the manual for older versions, use the index at
[gattocrucco.github.io/lsqfitgp](https://gattocrucco.github.io/lsqfitgp).

## Development

Clone the repository, create a virtual environment and install the requirements:

```sh
$ git clone https://github.com/Gattocrucco/lsqfitgp.git
$ cd lsqfitgp
$ make resetenv
$ . pyenv/bin/activate
```

The `Makefile` in the root directory contains targets to build the
documentation, run the tests, and prepare a release. Run `make` without
arguments to show the available targets:

```sh
$ make
available targets: [...]
$ make tests # or make examples, or ...
```

The tests are run on each push and the resulting coverage report is published
online at
[gattocrucco.github.io/lsqfitgp/htmlcov](https://gattocrucco.github.io/lsqfitgp/htmlcov/).
To browse it locally after `make tests` etc., do `make covreport` and open
`htmlcov/index.html` in your browser.

## Similar libraries

  * [stheno](https://github.com/wesselb/stheno)
  * [tinygp](https://github.com/dfm/tinygp)
  * [GPJax](https://github.com/JaxGaussianProcesses/GPJax)

See also [Comparison of Gaussian process Software](https://en.wikipedia.org/wiki/Comparison_of_Gaussian_process_software)
on Wikipedia.

## License

This software is released under the [GPL](https://www.gnu.org/licenses/).
Amongst other things, it implies that, if you release an adaptation of this
software, *[or even a program just importing it as external
library](https://www.gnu.org/licenses/gpl-faq.html.en#IfLibraryIsGPL)*, you
have to release its code as open source with a license at least as strong as
the GPL.

This software contains code adapted from the following sources:

  * [TOEPLITZ_CHOLESKY](http://people.sc.fsu.edu/~jburkardt/py_src/toeplitz_cholesky/toeplitz_cholesky.html)
    by John Burkardt (LGPL license)
  * [SuperGauss](https://cran.r-project.org/package=SuperGauss) by
    Yun Ling and Martin Lysy (GPL license)
  * [gvar](https://github.com/gplepage/gvar) by Peter Lepage (GPL license)
