# TODO

Other TODOs are scattered in the code. Search for `TODO`.

## Documentation

From numpy's guidelines: avoid the colon in the lists when there's nothing
after the column. Use optional this way: type, optional. Fixed set of values:
use braces with default argument first. In returned values list, if there's no
meaningful name, just write the type with no colon.

Add Kernel or IsotropicKernel in the kernels list.

Add a Raises section to GP methods that can trigger covariance matrix checks.

Add Examples sections.

## Fixes and tests

Stabilize Matern kernel near r == 0, then Matern derivatives for real nu
(quick partial fix: larger eps in `IsotropicKernel.__init__`).

Check that float32 is respected.

Test recursive dtype support.

Fix the second derivatives kernel bugs (aagh)

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

Also awkward arrays may be interesting as input format.

Support taking derivatives in arbitrarily nested dtypes. Switch from the
mess-tuple format to dictionaries dim->int. Dim can be (a tuple of) str, int,
ellipsis, slices, the same recursive format I should support in
`Kernel.__init__`.

Is there a smooth version of the Wiener process? like, softmin(x, y)?

Experiment on using second order corrections in empbayes_fit. I can
probably half-use the least squares derivative estimation on the residuals
term and a normal hessian on the logdet term.

## Optimization

### `gvar`-related issues

#### sparse `evalcov`

My `evalcov_blocks` code didn't make it into `gvar`, but I could recycle the
`_evalcov_sparse` function, since now `gvar` depends on `scipy` so using
`scipy.sparse.cs_matrix` publicly is possible. Interface:

    evalcov_sparse(g, lower=None, halfdiag=False):
        """
        Computes the covariance matrix of GVars as a sparse matrix.
        
        Parameters
        ----------
        g : GVar or array-like
            A collection of GVars.
        lower : None or bool, optional
            If None (default), the full covariance matrix is returned. If True,
            only the lower triangular part. If False, only the upper triangular
            part.
        halfdiag : bool, optional
            If True, the diagonal of the matrix (the variances) is divided by 2,
            such that, if lower=True or lower=False, the full covariance
            matrix can be recovered by C + C.T. Default is False.
        
        Returns
        -------
        C : scipy.sparse.cs_matrix
            The (possibly lower/upper triangular part of) covariance matrix of
            `g`.
        
        Raises
        ------
        ValueError
            If `lower` is None and `halfdiag` is True.
        """

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
