# TODO

Other TODOs are scattered in the code. Search for `TODO`.

## Documentation

Write a manual.

## Fixes and tests

Stabilize Matern kernel near r == 0, then Matern derivatives for real nu
(quick partial fix: larger eps in `IsotropicKernel.__init__`).

Check that float32 is respected.

Test recursive dtype support.

## New functionality

Long-term: move to a variable-oriented approach like gvar instead of the
monolithic GP object I'm doing now. It should be doable because what I'm
doing now with the keys is quite similar to a set of variables, but I have
not clear ideas on the interface. It could be based on underlying default
GP object, like gvar does with its hidden covariance matrix of all primary
gvars.

Invent a simpler alternative to Where/Choose and GP.addtransf for the case
of adding various kernels and getting the separate prediction for each one.

Accept xarray.DataSet and pandas.DataFrame as inputs. Probably I can't use
these as core formats due to autograd.

Support taking derivatives in arbitrarily nested dtypes.

Is there a smooth version of the Wiener process? like, softmin(x, y)?

Experiment on using second order corrections in empbayes_fit. I can
probably half-use the least squares derivative estimation on the residuals
term and a normal hessian on the logdet term.

## Optimization

`gvar.raniter` uses `gvar.evalcov_blocks` which is optimized for sparse
covariance matrices, while I tipically use it with dense covariance matrices.
For example, in `examples/w.py`, out of 10 seconds of `gvar.raniter`, the
actual matrix decomposition takes only 0.3 s. I should write my own sampling
function. Use numpy conventions instead of emulating `gvar.raniter`, so a
single output array where the first axes run along samples. Possible name:
`lgp.sample`. It has a single positional argument that is a scalar/array/dict
of gvars, or a scalar/array/dict representing a covariance matrix. It has a
`solver` argument like `GP.__init__` (move the solver name mapping to
`_linalg.py`).

Kronecker optimization: subclass GPKron where addx has a parameter `dim` and
it accepts only non-structured arrays. Or, more flexible: make a class
Lattice that is structured-array-like but different shapes for each field,
and a field _kronok in Kernel update automatically when doing operations with
kernels. Also, take a look at the pymc3 implementation. Can I use the
kronecker optimization when the data covariance is non-null? -> Yes with a
reasonable approximation of the marginal likelihood, but the data covariance
must be diagonal. Other desiderata: separation along arbitrary subsets of
the dimensions, it would be important when combining different keys with
addtransf (I don't remember why). Can I implement a "kronecker" numpy array
using the numpy internal interfaces so that I can let it roam around without
changing the code and use autograd? Something like pydata/sparse (if that
works).

Sparse algorithms. Make a custom minimal CSR class that allows an autograd
box as values buffer with only kernel operations implemented (addition,
multiplication, matrix multiplication, power). Make two decompositions
specifically for sparse matrices, sparselu and sparselowrank. Finite support
kernels have a parameter sparse=True to return a sparse matrix. Operations
between a sparse and a dense object should raise an error while computing
the kernel if the result is dense, but not while making prediction.
Alternative: make pydata/sparse work with autograd. I hope I can inject the
code into the module so I don't have to rely on a fork. Probably I have to
define some missing basic functions and define the vjp of the constructors.

DiagLowRank for low rank matrix + multiple of the identity (multiple rank-1
updates to the Cholesky factor? Would it be useful anyway?)

Option to compute only the diagonal of the output covariance matrix, and
allow diagonal-only input covariance for data (will be fundamental for
kronecker). For the output it already works implicitly when using gvars.

Make everything opt-in except numpy. There's already a numpy submodule for
doing this with scipy.linalg (numpy.dual).
autograd can be handled by try-except ImportError and defining a variable
has_autograd. With gvar maybe I can get through quickly if I define
gvar.BufferDict = dict and other things NotImplemented. (Low priority).

Fourier kernels. Look at Celerite's algorithms.
