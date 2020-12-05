# TODO

Other TODOs are scattered in the code. Search for `TODO`.

## Documentation

Add a Raises section to GP methods that can trigger covariance matrix checks.

Add Examples sections.

Interlinks with gvar and lsqfit docs.

Mention that numpy.lib.recfunctions.unstructured_to_structured may be used
for euclidean multidimensional input.

In the manual add an automatically generated index of the examples, with links
to github, with the docstrings of examples.

Separate the index of kernels by class (after adding StationaryKernel).

## Fixes and tests

Stabilize Matern kernel near r == 0, then Matern derivatives for real nu
(quick partial fix: larger eps in `IsotropicKernel.__init__`).

Check that float32 is respected.

Test recursive dtype support.

Lower thresholds in linalg tests.

gvar.BufferDict should have a prettier repr on the IPython shell. Is there a
standard way to configure a pretty print?

## New functionality

Long-term: move to a variable-oriented approach like gvar instead of the
monolithic GP object I'm doing now. It should be doable because what I'm
doing now with the keys is quite similar to a set of variables, but I have
not clear ideas on the interface. It could be based on underlying default
GP object, like gvar does with its hidden covariance matrix of all primary
gvars.

Invent a simpler alternative to Where/Choose and GP.addtransf for the case
of adding various kernels and getting the separate prediction for each one.
Possibly an helper function.

Accept xarray.DataSet and pandas.DataFrame as inputs. Probably I can't use
these as core formats due to autograd.

Also awkward arrays may be interesting as input format.

Support taking derivatives in arbitrarily nested dtypes. Switch from the
mess-tuple format to dictionaries dim->int. Dim can be (a tuple of) str, int,
ellipsis, slices, the same recursive format I should support in
`Kernel.__init__`. Add a class just for parsing the subfield specification.

Is there a smooth version of the Wiener process? like, softmin(x, y)? I tried
smoothing it with a gaussian but an undoable integral comes up.

Non-gaussian likelihoods, accessible with a string optional parameter in
GP.marginal_likelihood.

Student-t processes. There are some limitations. Possible interface: a nu
parameter in addx. Can I mix different nu in the same fit? Intepreting the
result would be a bit obscure, raniter will always sample normal distributions.

From GPy: periodic Matérn up to a certain k. Graph kernels from pyGPs:
https://www.cse.wustl.edu/~m.neumann/pyGPs_doc/Graph.html.

Are there interesting discontinuous processes apart from the white noise and
its integrals?

Can I do something helpful for Riemann manifolds? Like, the wikipedia page for
processes has a nice brownian motion on a sphere. How could I implement that
in a generic way which would allow the user to put any process on any manifold?

Bayesian classification: I found only one decent quick-and-lean python package,
bayesian-optimization, but I'm not satisfied with it nor with its
documentation. If I don't find out something better I'd like to study the
matter and write something.

### Fourier

The fourier transform is a linear operator, can I use it like I'm doing with
derivatives? I can't do it for the spectrum of a stationary process because
as soon as I put data in the posterior is non-stationary. In other words, the
correlation between a point and a frequency component is 0.

I can do the following:

  * Apply a DTFT to a specific key, can be done in addtransf or similar.
  
  * Compute the Fourier series of a periodic process (surely doable for the
    Fourier kernel).
    
  * Apply a Fourier transform or a DFT on a kernel which represents a
    transient process (should be doable for the gaussian kernel rescaled with
    gaussians).

Each kernel would need to have its handwritten transformation, with an
interface like `diff()`. For the Fourier series it makes sense to make a new
class `PeriodicKernel`, for the transform and DTFT dunno.

I should also add convolutions. Maybe a method `GP.addconv`.

### Classification

I'd like to avoid introducing a different likelihood and Laplace and EP. In the
user guide I did text classification in my own crappy but quick and still
statistically well-defined way, well it turns out that technique has been
studied and it works almost as well, it's in GPML section 6.5 "Least-squares
Classification". (A blog post on this would be interesting.)

I can improve it in the following way that won't require significant new code
but only explanations to the user: for each datapoint give the probability for
each class instead of the exact class, and map it back to a latent GP datapoint
with the inverse of the Gaussian CDF. Then the posterior class probability on
another point is given by Phi(mu/sqrt(1 + sigma^2)), where mu and sigma are the
mean an sdev of the posterior for the latent GP and Phi is the Gaussian CDF.

Proof: let uppercase letters be the class of each point. Let

    P(A|g_A) = Phi(g_A),

Where g_A is the latent GP variable. Then the posterior for the class on
another point B is

    P(B|P(A) = p_A) = int dg_B P(B|g_B,p_A) p(g_B|p_A)

Now we assume that B is conditionally independent of p_A, i.e. the dependencies
are completely expressed by the latent GP, so

    = int dg_B P(B|g_B) p(g_B|g_A) =
    = int dg_B Phi(g_B) N(g_B; mu, sigma)

Now I use a formula I found on [wikipedia]
(https://en.wikipedia.org/wiki/Error_function#Integral_of_error_function_with_Gaussian_density_function):

    = Phi(mu / sqrt(1 + sigma^2)).

This does not give the joint probability of various points being in the same
class. I think it is doable but extending the formula to more than one
variable is not immediate. I could at least obtain it for two points, so that I
can compute a sort of "correlation matrix".

## Optimization

### `gvar`-related issues

Lepage has done sone updates after my requests, I should look into it and update the manual accordingly.

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

DiagLowRank for low rank matrix + multiple of the identity (multiple rank-1
updates to the Cholesky factor? Would it be useful anyway? Maybe it's easier
with QR)

#### Evenly spaced input

With stationary kernels, an evenly spaced input produces a toeplitz matrix,
which requires O(N) memory and can be solved in O(N^2). If the data has
uniform independent errors it's still toeplitz.

Add a Toeplitz decomposition to _linalg. Since it's O(N^2) I can save the
matrix and rerun the solve in every routine, but maybe for numerical accuracy
and to use gvars it is better to explicitly save a cholesky decomposition.
Wikipedia says it can be done but it's not in scipy.linalg, let's hope it is in
LAPACK. This decomposition class does not check the input matrix, it just reads
the first row.

Add a toeplitz kw to Kernel. Values: False, True, 'auto'. True will complain if
a non-evenly spaced input comes in. 'auto' will check if the input is compliant
(before applying forcebroadcast), otherwise proceed as usual. Specific kernels
need not be aware of this, `__call__` passes them only the necessary points and
then packs up the result as a toeplitz array-like.

This only applies to stationary kernels and isotropic kernels in 1D. I can
implement it this way: make an intermediate private Kernel subclass
_ToeplitzKernel which implements `__call__`, StationaryKernel and
IsotropicKernel are then subclasses of _ToeplitzKernel.

Add a 'toeplitz' solver to GP. It fails if the matrix is non-toeplitz,
accepting both toeplitz array-likes and normal matrices to be checked.
Non-toeplitz solvers print a warning if they receive a toeplitz array-like.

#### Markovian processes

Example: the Wiener kernel is a Markov process, i.e. on the right of a
datapoint you don't need the datapoints on the left. This means it can be
solved faster. See how to do this and if it can be integrated smoothly with the
rest of the library.

#### Autoregressive processes

Like what Celerite does, it solves 1D GPs in O(N). Reference article to cite:
FAST AND SCALABLE GAUSSIAN PROCESS MODELING WITH APPLICATIONS TO ASTRONOMICAL
TIME SERIES, Daniel Foreman-Mackey, Eric Agol, Sivaram Ambikasaran, and Ruth
Angus.

I should make the Celerite kernel a handwritten subclass since it's a
subalgebra.

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

Idea: make an array-like class to represent the tensor product of arrays
without actually computing explicitly the product. Use this class to represent
a lattice (may not be trivial since the input can be structured), then the
kernels will operate on it like a numpy array and automatically the output will
still be a tensor product if all the operations were separable. I think that
ExpQuad is the only possible case where an apparently non-separable formula is
actually separable, I could make a quick fix by using forcekron=True, or
implement an ufunc that is understood by the tensor product class. A more
elegant but somewhat complicated solution would be to have a sum of arrays
array-like class that on any ufunc does explicitly the sum but for the
exponential, and a sum of a tensor product array-like produces a sum of arrays
array-like.

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

#### Using multiple solvers

I have already implemented block matrix solving. This requires a solver for
each key, so I will add a solver parameter to addx and addtransf.

However there's the need for more fine-grained settings. Say I have a kronecker
product of a normal matrix and a toeplitz matrix. The toeplitz matrix needs its
specialized algorithm. Since each special matrix will have its specialized
algorithm(s), the thing can be done this way: solver can be a tuple of str.
Solvers are tried from the left until one succedes. If specialized solvers
appear first in the list, they will be applied on specialized matrices, while
they will fail on normal matrices which will fall back to another solver. Raise
warnings if a generic algorithm is applied on a special matrix.
