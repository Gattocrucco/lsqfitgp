<!--- lsqfitgp/docs/changelog.md

  Copyright (c) 2023, Giacomo Petrillo

  This file is part of lsqfitgp.

  lsqfitgp is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  lsqfitgp is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with lsqfitgp.  If not, see <http://www.gnu.org/licenses/>.
-->

<!--- This changelog is written in Markdown and without line splits to make it
  copy-pastable to github releases. -->

<!--- TODO: have sphinx process the dollars as latex --->

# Changelog


## 0.20. Kernels, kernels, kernels. All you can think about is covariance functions. I am sick of your kernels. I am sick of cleaning the leftover eigenspaces you leave on the carpet when you come home, late in the night, after spending all day hewing positive semidefinite operators. I am sick of hearing your breath, the stale unique acrid smell that fills the algebraists' workrooms, wafting through the linen to my nostrils. Go away. Go away from here and rest with your beloved kernels! (2023-08-29)

### Release highlights

  * Improvements to the kernel system.
  * `GP` objects are immutable.

### Kernels

  * Coherent logic to determine the class of the result of operations on kernels.
  * Each generic class has it own `Cross*` superclass representing crosskernels.
  * Decorators to make crosskernels.
  * Transformation system with clear semantics for linear operators and algebraic operations.
  * Exponentiation `scalar ** kernel`.
  * Replaced `diff` and `where` with linear operators.
  * Hardcoded non-fuzzy derivability check. May yield false positives when additional derivatives are taken by an outer context, but disabling it is easy.
  * Fix derivation on crosskernels when one input is scalar and the other is structured.

### GP

  * GP objects are immutable: operations can only return a new, different object. This makes them more compatible with jax and reduces chances of errors.
  * Change name and signature of methods to define processes: the prefix is `def` instead of `add`, and the first argument is always the key of the new process.
  * GP methods that return dictionaries use ordinary Python `dict`s instead of `gvar.BufferDict`.

### gvar-related

  * Context manager `switchgvar` to avoid accumulating gvars in memory in long running loops.


## 0.19. Miss the Forest for Two Forests and an Auxiliary Regression Term (2023-08-22)

### Release highlights
  
  * GP version of BCF (Bayesian Causal Forests).
  * Extension of the `copula` submodule to a full PPL (Probabilistic Programming Language) to specify the hyperparameter priors.

### `bayestree` submodule

* The new class `bcf` implements the infinite trees limit of the BCF model for causal inference in deconfounded observational studies. Additionally, it allows specifying an arbitrary auxiliary regression term.
* `bart` adds a parameter `marginalize_mean` to avoid having the mean as hyperparameter.

### `copula` submodule

The new class `Distr` replaces `CopulaFactory` and can be instantiated. It represents a probability distribution over a tensor, much like PyMC. `Distr` objects can be used as parameters to other `Distr` objects to define a model. The distribution is always parametrized to be a multivariate standard Normal.

### Improvements to `StructuredArray`

 * `StructuredArrays` created from numpy structured arrays with same names and types, but different padding, will now  compare as equal instead of different.
 * `StructuredArray.from_dict` to create a `StructuredArray` from a dict of arrays.
 * Define `len(StructuredArray)`.
 * Support `numpy.concatenate`.
 * Support `numpy.lib.recfunctions.append_fields`.

### `gvar`-related

 * Solve bug that could make `jax.jit` fail if a `gvar.BufferDict` was passed as argument.
 * Decorator `gvar_gufunc` to make a function support gvars.


## 0.18. Our code, hallowed be thy dependencies, but deliver us from pins and torment (2023-08-02)

Recent versions of `lsqfitgp` would impose version 0.4.6 of `jax` and `jaxlib` due to incompatibilities with newer releases. This limitation is now gone. Thanks to the experimental Windows support added to `jax`, `lsqfitgp` can be `pip install`ed on Windows. The unit tests currently have some failures and crash midway, so I guess it's not fully usable.

### New decomposition system

The covariance matrix decomposition system has been scrapped and rewritten. Consequences:

  * All solvers but `'chol'` are gone, they were never useful in practice anyway. I may add new solvers in the future. Consequently, it is not necessary to specify `GP(..., solver='chol')` to speed up the fit.
  * It is not possible to pass a decomposition in place of `givencov` to use the Woodbury matrix formula.
  * Operations with decompositions can not be differentiated; instead, there is a method that computes the Normal density and its derivatives with custom code. This is faster because it avoids redundant calculations.
  * `GP.marginal_likelihood` loses the `separate` option.
  * `lsqfitgp.raniter` fails if the matrix is not p.s.d. instead of issuing a warning and then working around the problem.
  * The `'fisher'` method in `empbayes_fit` now works decently enough that it can be used in practice, same for `covariance='fisher'`.

### `empbayes_fit`

  * `empbayes_fit(..., minkw=dict(method='trust-constr')` did not work, fixed.
  * Now `jit=True` by default.


## 0.17. And God spoke through the interface, and the exception was raised KeyError('fagiolino') (2023-07-17)

This release mostly features improvements to the BART simplified interface.

### `bayestree.bart`
 * Example on the ACIC data challenge
 * Error weights
 * New methods `bart.gp()`, `bart.data()`, `bart.pred()`
 * All methods can sample hyperparameters with Laplace
 * The prior mean is a free hyperparameter
 * The prior on the error variance is scaled with the data, in particular `bart` now works well with non-standardized data

### `raniter`
 * Accept user provided random number generator
 * Solved bug with scalar elements in dictionaries

### `empbayes_fit`
 * The arguments in `gpfactorykw` are now passed through a function call instead of being closed over. This implies that, if `jit=True`, large arrays can be passed efficiently without being hardcoded in the compiled code, and that arguments not understood by JAX can not be passed through `gpfactorykw`.


## 0.16. If fitting a nonparametric regression became so easy, many professional statisticians would be left without a job! (2023-04-24)

This release adds a simplified interface that packages a Gaussian process regression with the BART kernel.


## 0.15. EUROCIM special edition, 10 exclusive bugs for the first 500 users, download now (2023-04-17)

This release mostly features further improvements to the BART kernel. According to the benchmarks I am going to show at EUROCIM 2023, GP regression with this kernel and hyperparameter tuning beats the original cross-validated BART MCMC both in statistical and computational performance. Following the ACIC challenge 2022 results, this implies that it's probably SoTA on the standard causal inference in decounfounded observational study setting with $n<10000$, although that is still to be verified.

### BART
  * Strongly optimized implementation, both on memory usage and running time.
  * Computation is batched by default to avoid creating large intermediate matrices when there are many covariates.
  * Good choices for technical parameters are documented.
  * `BART.correlation` also accepts total splits and indices w.r.t. the grid instead of split count separations.
  * Example script with full BART inference.

### Hyperparameters fit (`empbayes_fit`)
  * New parameters:
      - `covariance` to set how to estimate the posterior covariance matrix
      - `fix` to fix some hyperparameters to the initial values
      - `mlkw` to pass additional keyword arguments to `GP.marginal_likelihood`
      - `forward` to use forward autodiff for the likelihood gradient; it uses a partially reverse autodiff calculation to avoid the $O(n^3d)$ complexity of full-forward mode
  * Removed the `'hessian'` and `'hessmod'` values of parameter `method`. Use `'fisher'` instead. **<-- BREAKING CHANGE**
  * New attributes:
      - `prior`: the prior on the hyperparameters, as gvars
      - `initial`: the starting values for the MAP search
      - `fix`: the mask indicating locked hyperparameters
      - `gpfactory`: the `gpfactory` argument
  * Additional logging: per task timing and function calls count.
  * Copula factories for the Beta and Inverse Gamma distributions, in the submodule `copula`.

### `StructuredArray`
  * New function `unstructured_to_structured` to create a `StructuredArray` from an unstructured array.
  * Arrays are now read-only. Setting the fields requires a JAX-like syntax with `.at[field].set(value)`. **<-- BREAKING CHANGE**
  * Added methods `squeeze` and `astype`, although the latter refuses to change the type.
  * Pytree unflattening tries to infer shape and dtype if the leaves are not compatible with the original structure. The inference is incomplete and ambiguous. **<-- BREAKING CHANGE**
  * Implemented numpy functions `ix`, `empty_like`, and `empty`.

### Kernels
  * New `CrossKernel` parameter `batchbytes` to batch the calculation, and corresponding method `batch`. Requires jax-traceability of the kernel implementation.
  * The `Decaying` kernel gets an explicit real exponent to indicate infinite divisibility.

### GP
  * By default, even if `checksym=False`, the full covariance matrix is computed for diagonal blocks, unless the additional new option `halfmatrix=True` is set. **<-- BREAKING CHANGE**

### Dependencies
  * Pinned jax and jaxlib to 0.4.6. Earlier versions severely break the optimization of the BART kernel, while later versions break the `jit` option in `empbayes_fit`. Sorry for this pin, I do know that dependency pins in libraries are evil. **<-- BREAKING CHANGE**
  * Tested on Python 3.11


## 0.14.1. It wasn't me! (2023-03-28)

Limits the maximum supported jax version to 0.4.6 due to the new `jax.jit` implementation triggering a bug I have not been able yet to fix.


## 0.14. Bayes Area Rapid Transit (2023-02-16)

Improvements to the BART kernel:
 * nonuniform splitting probability along covariates
 * interpolation between lower and upper bound
 * recursion reset to increase accuracy

A good but fast approximation to the exact BART kernel can now be obtained with `BART(..., maxd=4, reset=2, gamma=0.95)`. Other features:
 * `StructuredArray.from_dataframe` converts pandas and polars dataframes to a jax-compatible columnar format
 * Added LOBPCG to the decompositions, it computes a low-rank approximation of the covariance matrix, less accurate but faster than 'lowrank', renamed Lanczos
 * Faster and coherent positivity check of GP objects. Only the highest and lowest eigenvalues, estimated with LOBPCG, are checked, which stays fast even with large matrices. The check is repeated as many times as necessary on the minimal set of covariance blocks touched by any operation.
 * The Woodbury formula, triggered by passing a pre-decomposed error covariance matrix, now works well for ill-conditioned priors (but not still for the error covariance).
 * `GP.add(proc)lintransf` define the default process if not otherwise specified.
 * The `GP` solvers now require two separate parameters `epsrel` and `epsabs` instead of `eps` to specify the regularization. `epsabs` may be useful to fix the regularization independently of the hyperparameters when using `empbayes_fit`.
 * Decompositions produced by `GP.decompose` have additional operations and properties `(de)correlate(..., transpose=True)`, `matrix()`, `eps`, `m`

Fixed many bugs, in particular all JAX JIT incompatibilities are gone.

New requirements:
 * scipy >= 1.5
 * jax, jaxlib >= 0.4.1


## 0.13. The mark of Riemann (2022-11-01)

Major changes:

 - The `Fourier` kernel has been replaced with `Zeta`, which is the generalization to a continuous order parameter.

New features:

- New method `GP.decomp` to decompose any positive semidefinite matrix.
- `GP.pred` and `GP.marginal_likelihood` now accept a `Decomposition` produced by `GP.decomp` as value to the `ycov` parameter, indicating that the decomposition of the prior covariance matrix shall be done with Woodbury's formula. This is computationally convenient if the subset of process values is a low rank linear transformation of some other values, and `ycov` is a fixed matrix that can be computed and decomposed once and for all. The low rank is automatically detected and exploited, although only to one level of depth in the transformations defined with `GP.addtransf` or `GP.addlintransf`. The current implementation of Woodbury does not work properly with degenerate or ill-conditioned matrices.
- `GP.addcov` accepts a new parameter `decomps` to provide precomputed decompositions of the diagonal blocks of the user-provided covariance matrix. These decompositions are pre-loaded in the cache used by `GP.pred` and `GP.marginal_likelihood`.
- New function `sample`, equivalent of `gvar.sample`.
- New parameter `initial` to `empbayes_fit` to set the starting point of the minimization.
- New parameter `verbosity` to `empbayes_fit` to set console logging level.
- New attributes `pmean`, `pcov` and `minargs` to `empbayes_fit`. `minargs` provides the complete argument list passed to the minimizer.
- New parameter `MA(norm=...)` to standardize the moving average covariance kernel to unit variance.
- New method `BART.correlation` to compute the BART correlation with more options than the standard kernel interface.
- Optimization in the implementation of BART to make the computational cost linear with two levels of depth.

Bugs fixed

- Some compatibility issues with new JAX versions.


## 0.12. The wealth of a man is measured not in gold, neither in cattle; but in his most treasured covariance functions (2022-07-16)

This release improves the built-in collection of covariance functions.

New kernels:
- `Cauchy`: generalized Cauchy kernel, replaces and extends `RatQuad`
- `HoleEffect`
- `Bessel`
- `Sinc`, `Pink`, `Color`: kernels with power-law spectra
- `StationaryFracBrownian`: a stationary analogue of `FracBrownian`
- `Decaying`: a kernel for exponentially decaying functions
- `CausalExpQuad`
- `Log`
- `Circular`: a periodic kernel, less smooth than `Periodic`
- `AR`, `MA`: discrete autoregressive and moving average processes
- `BART`: Bayes Additive Regression Trees

Improved kernels:
- `Constant`, `White`: support non-numerical input
- `Matern`: all derivatives work for any value of `nu`; extended to `nu=0` as white noise
- `GammaExp`: fixed problems with second derivatives
- `Gibbs`: support multidimensional input
- `FracBrownian`: the fractional Brownian motion is now bifractional with an additional index K; extended to negative values
- `Fourier`: support arbitrarily large `n`
- `Wendland`: renamed `PPKernel`, now accepts real exponent parameter

Deleted kernels:
- `RatQuad`: replaced by `Cauchy`
- `PPKernel`: replaced by `Wendland`
- `Matern12`, `Matern32`, `Matern52`: replaced by `Maternp`

New general feature: kernels have a `maxdim` option to set a limit of the dimensionality of input.

Improvements to `StructuredArray`:
- now supported numpy functions can be applied from `numpy` instead of using their duplicates in `lsqfitgp`
- a `StructuredArray` can be converted back to a numpy array
- a `StructuredArray` can be converted to an unstructured array with `numpy.lib.recfunctions.structured_to_unstructured`


## 0.11. With great linearity comes great responsibility (2022-06-11)

New features:

* Two more generic GP methods to define linear transformations, `addproclintransf` and `addlintransf`, which take arbitrary callables
* Support for jax's just in time compiler
* Second order methods in `empbayes_fit`, included Fisher scoring

And many bugfixes as usual.


## 0.10. A rose by any other name would be differentiable (2022-06-01)

Ported from autograd to jax. Consequences:

  * Does not naively pip-install anymore on Windows
  * Internal bug: backward derivatives of bilinear forms are broken
  * Scalar-Kernel multiplication in autodiff context now works
  * Problems with second derivatives of some kernels are now solved


## 0.9. Free as in "you can peek at the prior" (2022-05-15)

 * Lifted the restriction that points can't be added any more to a `GP` object after getting the prior in `gvar` form (thanks to G. P. Lepage for implementing the necessary gvar feature)
 * New `GP` parameter `posepsfac` to change the tolerance of the prior covariance matrix positive definiteness check
 * The positiveness check is done when avoiding `gvar`s too


## 0.8. Fourier (2022-05-05)

Fixes a lot of bugs. New features and improvements:
  * Fourier series, experimental and half-broken (`loc` and `scale` won't work)
  * New process-transformation methods `GP.addkernelop`, `GP.addprocderiv`, `GP.addprocxtransf`, `GP.addprocrescale`
  * User-defined prior covariance matrices
  * Cache of prior covariance matrices decompositions, to get the posterior for different variables separately
  * New kernel class `StationaryKernel`
  * Improved the numerical accuracy of the `Fourier` kernel
  * Option `saveargs` in `Kernel` to remember the initialization arguments
  * New regularizations `svdcut-` and `svdcut+` that preserve the sign of the eigenvalues
  * Now `BlockDecomp` implements `correlate` and `decorrelate` with the block Cholesky decomposition


## 0.7. Little washing bear (2022-04-22)

  * With the new family of methods `GP.addproc*` one can define independent processes and combinations of them.
  * Fixed error which arose sometimes when taking the gradient of `GP.marginal_likelihood` with multiple derivatives taken on the process.
  * The derivability checking system is now correct but fuzzy: it emits warnings if the derivatives *may* be illegal, and raises an exception only when they are surely illegal.
  * Added `axes: int` parameter to `GP.addtransf` to contract more indices.


## 0.6.4. gvar fix^2 (2022-03-13)

A fix to adapt to a gvar fix.


## 0.6.3. gvar fix (2022-03-10)

A fix to adapt to a change in `gvar` internals.


## 0.6.2. ar kernel (2020-05-19)

  * Renamed the AR2 kernel "Celerite", because it did not really cover all the AR(2) cases.

  * The Harmonic kernel now supports taking a derivative w.r.t Q even at Q=1.


## 0.6.1. Night fix (2020-05-19)

  * Simplified the parametrization of AR2 and Harmonic kernels.

  * Fixed numerical accuracy problems in the Harmonic kernel.


## 0.6. ARRH (2020-05-18)

  * New kernels AR2, BrownianBridge, Harmonic.


## 0.5.1. Flat prior (2020-05-17)

  * Fixed bug in GP.prior(key, raw=True) which would return a flat array whatever the original shape


## 0.5. Fastiter (2020-05-16)

  * New generator lsqfitgp.raniter; like gvar.raniter but takes the mean and covariance matrix separately.

  * Solved bug in GP.prior when raw=True


## 0.4. Classy (2020-05-15)

  * Converted empbayes_fit to a class and optimized it a bit.


## 0.3. Quad (2020-05-12)

  * Improved numerical accuracy.

  * New kernel OrnsteinUhlenbeck.


## 0.2.1. No SVD (2020-05-09)

  * Use a gvar 11.5 feature to replace SVD with diagonalization in gvar.raniter.


## 0.2. Series (2020-05-08)

  * New kernels Taylor and Fourier.


## 0.1.4. Forcekron (2020-05-05)

  * Fixed kernels Cos, Periodic, Categorical and FracBrownian on multidimensional input. Now Cos and Periodic are not incorrectly considered isotropic.


## 0.1.3. Readonly (2020-05-04)

  * Non-field indexing of StructuredArray now returns a read-only StructuredArray that can not be assigned even on whole fields. Wrapping it again with StructuredArray removes the limitation.

  * New parameters `raises` of empbayes_fit to disable raising an exception when the minimization fails.


## 0.1.2. Asarray (2020-05-03)

  * Exposed StructuredArray-compatible numpy-like functions

  * Improved docstrings

  * empbayes_fit raises an exception when the minimization fails


## 0.1.1. Vector (2020-05-01)

  * Support decorating a np.vectorize function as kernel


## 0.1. Wiener (2020-04-30)

  * Added kernel WienerIntegral

  * Support scalars in GP.addx


## 0.0.2. Some fixes (2020-04-27)

  * Improved docstrings

  * Improved error messages

  * Optimized gvar priors for transformed processes


## 0.0.1. Initial release (2020-04-23)

No comment.
