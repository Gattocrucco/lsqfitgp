[![PyPI](https://img.shields.io/pypi/v/lsqfitgp)](https://pypi.org/project/lsqfitgp/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13930792.svg)](https://doi.org/10.5281/zenodo.13930792)


# lsqfitgp

Python module to do inference with Gaussian processes. Features:

  * Based on [JAX](https://github.com/google/jax).
  * Interoperates with [gvar](https://github.com/gplepage/gvar) and
    [lsqfit](https://github.com/gplepage/lsqfit) to facilitate inexpert users.
  * Recursively structured covariates.
  * Apply arbitrary linear transformations to the processes, finite and
    infinite.
  * Small [PPL based on Gaussian
    copulas](https://gattocrucco.github.io/lsqfitgp/docs/reference/copula.html)
    to specify the hyperparameters prior.
  * [Rich
    collection](https://gattocrucco.github.io/lsqfitgp/docs/reference/kernelsref.html)
    of covariance functions.
  * Good GP versions of
    [BART](https://gattocrucco.github.io/lsqfitgp/docs/reference/bayestree.html#lsqfitgp.bayestree.bart)
    (Bayesian Additive Regression Trees) and
    [BCF](https://gattocrucco.github.io/lsqfitgp/docs/reference/bayestree.html#lsqfitgp.bayestree.bcf)
    (Bayesian Causal Forests).
  
See [this report](https://www.giacomopetrillo.com/scuola/gppdf.pdf) for the
theory behind lsqfitgp.

## Installation

Python >= 3.10 required. Then:

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

## Similar libraries

  * [stheno](https://github.com/wesselb/stheno)
  * [tinygp](https://github.com/dfm/tinygp)
  * [GPJax](https://github.com/JaxGaussianProcesses/GPJax)

See also [Comparison of Gaussian process Software](https://en.wikipedia.org/wiki/Comparison_of_Gaussian_process_software)
on Wikipedia.

## License

This software is released under the [GPL](https://www.gnu.org/licenses/).
Amongst other things, it implies that, if you release an adaptation of this
software, [or even a program just importing it as external
library](https://www.gnu.org/licenses/gpl-faq.html.en#IfLibraryIsGPL), you
have to release its code as open source with a license at least as strong as
the GPL.
