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

### `gvar`-related issues

#### `evalcov_blocks`

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

#### `svd`

I made a pull request to gvar with a faster `evalcov_blocks`. Now by profiling
`examples/w.py` I see the bottleneck is in `gvar.svd.__init__`. It is not the
actual SVD decomposition (why on earth is it doing a SVD instead of
diagonalizing?), it is just the code in `gvar.svd.__init__`. What is it doing?
I'm in doubt if trying to optimize `gvar.svd` or just optimizing `gvar.raniter`
by writing her own decomposition routine. Or maybe `gvar.svd` computes optional
things that I can disable in `gvar.raniter`.

#### `evalcov`

Possible optimization for `evalcov`: if the sparsity is more than something,
also depending on the absolute size, use a dense matrix multiplication. I tried
it on a completely dense case with 100 variables and it gave a 16x improvement,
keeping also into account the conversion of the matrices. Steps:

  * Build the mask as usual reading the indices from `d` attributes, while also
    counting the total number of elements.
  
  * Get the count of primary gvars involved by summing the mask.
  
  * Count the number of nonzero entries in `cov` with the mask. Probably
    requires a new `smat` method.
    
  * Using the counts and the number of primary gvars, determine if the dense
    algorithm would be convenient.
    
  * Extract a dense submatrix from `cov` using the mask. Probably requires a
    new `smat` method, and probably it is convenient to first convert the
    mask to a mapping, i.e. replace the 1s with the cumsum of the 1s, they then
    point to the destination index with an offset of +1.
    
  * Write `d` elements into a dense matrix, using the mapping as above.
    Probably requires a new svec method, which could be used by the `smat`
    method above.
    
  * Perform the matrix multiplication. Check if LAPACK has routines specialized
    for matrix multiplication with symmetric inputs/outputs. There's a LAPACK
    wrapper in scipy.

#### Global covariance matrix

In general gvar could benefit some core optimizations. The global covariance
matrix is in LIL (list of lists) format, and the full matrix is stored despite
being symmetrical. There are two main optimizations that can be implemented
separately: using a compressed sparse format, and storing only the lower
triangular part.

If only the lower triangular part is stored, computing covariance matrices is
still reasonably simple. Let `A` be the covariance matrix. Say `A = L + L^T`,
where `L` is lower triangular, so `L_ij = A_ij` if `i < j`, `A_ii/2` if `i = j`,
`0` if `i > j`. Let `B` be the linear transformation from the primary `gvar`s
to the `gvar`s we want to compute the covariance of, i.e. `C = B A B^T`. So:
`C = B A B^T = B (L + L^T) B^T = H + H^T`, where `H = B L B^T`.

Using a compressed sparse format is not particularly inefficient because,
although gvar needs to add entries to it, the new entries are always added as
a diagonal block on new rows, so the buffers can be extended like `std:vector`
bringing a global factor of 2. The possibility to implement the functionality
of adding primary gvars correlated with existing primary gvars can be preserved
if only the lower triangular part is stored.

All this can be implemented using `scipy.sparse`. The core functionality of
`scipy.sparse` can be easily copy-pasted if depending on scipy is to be avoided.

Storing only the lower part should only have benefits. Using a compressed
sparse format may not be worth it.

### Solvers

Fourier kernels. Look at Celerite's algorithms.

DiagLowRank for low rank matrix + multiple of the identity (multiple rank-1
updates to the Cholesky factor? Would it be useful anyway?)

#### Kronecker

Subclass GPKron where addx has a parameter `dim` and it accepts only
non-structured arrays. Or, more flexible: make a class Lattice that is
structured-array-like but different shapes for each field, and a field _kronok
in Kernel update automatically when doing operations with kernels. Also, take a
look at the pymc3 implementation. Can I use the kronecker optimization when the
data covariance is non-null? -> Yes with a reasonable approximation of the
marginal likelihood, but the data covariance must be diagonal. Other
desiderata: separation along arbitrary subsets of the dimensions, it would be
important when combining different keys with addtransf (I don't remember why).
Can I implement a "kronecker" numpy array using the numpy internal interfaces
so that I can let it roam around without changing the code and use autograd?
Something like pydata/sparse (if that works).

Option to compute only the diagonal of the output covariance matrix, and
allow diagonal-only input covariance for data (will be fundamental for
kronecker). For the output it already works implicitly when using gvars.

#### Sparse

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

### Other

Make everything opt-in except numpy. There's already a numpy submodule for
doing this with scipy.linalg (numpy.dual). `autograd` can be handled by
try-except ImportError and defining a variable has_autograd. With gvar maybe I
can get through quickly if I define gvar.BufferDict = dict (Low priority).
