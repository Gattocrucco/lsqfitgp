# TODO

Other TODOs are scattered in the code. Search for `TODO`.

## Documentation

Add a Raises section to GP methods that can trigger covariance matrix checks.

Add more 'Examples' sections in docstrings

Mention that `numpy.lib.recfunctions.unstructured_to_structured` may be used
for euclidean multidimensional input.

In the nonlinear fit example, use the gaussian cdf instead of the hyperbolic
arctangent to make a uniform prior over the interval. => Or maybe explain
what distribution `tanh` entails.

A chapter on propagation, from simple GP-only to nonlinear with multiple fits.

mi sono accorto che lepage già praticamente faceva la stessa cosa che faccio io
nell'esempio "y has no errors, marginalization". Comunque si chiama "regola di
Matheron". I wonder if Lepage knows that what he's doing is a special case of
Gaussian processes, maybe he's reinventing everything himself because he's
smart.

Aggiungere che il manuale richiede conoscenze di base di algebra lineare e
analisi.

Check again the raniter example because gvar.raniter seems to have gotten much
faster. => Actually it's lgp.raniter who's gotten slower! The cholesky
decomposition now takes only 10 ms on 1000x1000 compared to 50 ms on my old
laptop. However next(lgp.raniter(,)) takes 270 ms. Profiling shows that it's
indeed the cholesky decomposition who is taking up most of the time. WTF??
Moreover profiling one single execution shows it takes about 15 ms, while
profiling a cycle shows about 250 ms per cycle. WTF????? A cycle with two
executions is sufficient to jump from 15 ms to 250 ms. It works this way even
if I call lgp.raniter one or two times between two time calls. gvar.raniter
does not show this behaviour, it's linear as expected. => Update: the cholesky
decomp seems to take 250 ms in gvar.raniter too, but already in the first
call. Idea: my laptop has 2 "efficiency cores", maybe it's using them in a
weird way I can't predict. But why wouldn't this happen when I just call
linalg.cholesky? => Activity monitor shows I'm using all cores when running
a timeit of lgp.raniter, so nope.

Chapters for the advanced guide: -kernel operations -fourier series -taylor
series (taylor afterward because it has a richer set of kernels implementing but
they must be obtained with transformations) -user defined kernels => Also
something on pseudoinverses and projectors, for when the data is incompatible
with the model.

In the kernels reference show random samples in 2D when maxdim > 1

Add an example with regression + gp

Add a brief head "Example" section in the README

fare in modo che quando sto preparando una release non ci sia un periodo in cui
la documentazione online è rotta perché i source link puntano al nuovo tag che
non esiste. La cosa migliore sarebbe mostrare di default l'ultima versione
rilasciata anziché master.

Rename all the single letter example scripts and gift them a communicative
description

## Fixes and tests

Stabilize Matern kernel near r == 0, then Matern derivatives for real nu
(quick partial fix: larger eps in `IsotropicKernel.__init__`).

Check that float32 is respected. => may not be worth it since GP needs accuracy.

Test recursive dtype support in CrossKernel

Check that conditioning multiple times with zero errors does not change the
result.

In the chapter on integration the samples are very noisy. Some trials with
playground/integrals.py told me that the problem is not gvar.raniter but the
solver used in the prediction. Using eigcut- with a high enough eps makes the
samples smooth, but the result is very sensitive on the choice of cut, and the
samples change shape macroscopically. From this I guess that the problem is
that important eigenvectors of the posterior covariance matrix are not smooth.

Does pred works with empty given? It should allow it and behave like prior. I
should take the occasion of the new linalg system to allow 0x0 matrices.

Add jit tests to test_GP. Not just for error conditions, also nontrivial
successful computations with pred. Maybe do this in test_pred.

Use `jax.experimental.checkify` instead of `_patch_jax.skipifabstract`, then
handle checking in `empbayes_fit`, with options to check errors or not (default
checks everything and raises mid-minimization). Maybe wait for `checkify` to
become non-experimental? => I tried using checkify and it was a disaster,
definitely wait for improvements.

Check that lsqfitgp installs from scratch on a fresh system. I expect that it
will fail on a python venv due to the need to compile gvar and lsqfit, but
succeed under spyder because it uses conda. => convince lepage to do the wheels.

## Implementation details

Fare una github action che carica la release su PyPI quando faccio la release
su github. => Non è che però è un rischio? In questo modo il mio account github
controllerebbe cosa finisce su PyPI perché dovrei metterci le chiavi di accesso

In defining abstract methods, use the single line syntax:

    @abc.abstractmethod
    def cippa(*_): pass

such that there is not need to add # pragma: no cover (I hope, try it)

## New functionality

A function to extract the sdev/var from a covariance dict/tensor.

### Bayesian optimization

Bayesian optimization: I found only one decent quick-and-lean python package,
bayesian-optimization, but I'm not satisfied with it nor with its
documentation. If I don't find out something better I'd like to study the
matter and write something. => see the recent book "Surrogates"

### New interfaces

#### GP in-language

This could be implemented with a class that uses `GP` behind the scenes,
imitating Stheno:

```python
m = MGP()
f = m.gp(ExpQuad())
g = m.gp(ExpQuad())
h = f + 2 * g.diff(1)
y = f(x)
z = g(x)
q = y - 2 * z
m2 = m | (q == [1, 2, 3], z[0] == 7)
yc = y << m2
yc.sample()
yc.mean()
yc.cov()
y.mean()
a = m.norm(cov)
...
```

The variable names would be set automatically internally, and be carried by
the objects as attributes. The object does not refer directly to the GPs because
it changes since it's immutable.

### Efficient sampling

se la matrice di covarianza a priori dei punti dove voglio calcolare la
predizioni è speciale, riesco a fare qualcosa? Ad esempio penso a una griglia
fitta... forse LOVE (nei preprint salvati) fa qualcosa del genere visto che
diceva di fare fast sampling delle predizioni? => vedi quaderno 7 aprile 2022,
c'è anche il caso di dati assunti indipendenti date le predizioni che è
super-potente. => This thing could also solve the rough samples problem.

la cosa che uso per fare il gaussian process con gvar si chiama Matheron's rule
(https://avt.im/publications/2020/07/09/GP-sampling) Questa cosa forse è un
modo diverso alla fine di scrivere quello che ho ricavato con schur e woodsbury
direttamente sulle matrici, che se so invertire velocemente il priore sui test
point allora sono a posto

To sample efficiently from transformations, apply the transformations to the
sample instead of decomposing the covariance matrix, only if the starting
covariance matrix is smaller than the transformed one. More complex case:
parts of these matrices have been already decomposed and are in cache.

### Nonlinear fit

voglio poter risolvere la seguente situazione: metto un *posteriore*
di gp dentro a lsqfit e poi voglio prolungare. Mi sa che predfromfit non
funziona mica. Forse funziona magicamente a culo usando i gvar che si ricordano
tutte le correlazioni.

dovrei riimplementare lsqfit.nonlinear_fit per usare le mie decomposizioni e
cooperare meglio con i processi gaussiani latenti senza passare dalle gvar.
Quindi devo anche fare una funzione condivisa che mappa il nome della
decomposizione a una classe Decomp.

I think the second order correction would be useful, however it can be
computationally heavy in general. Maybe if I can write the bias correction as
the gradient of something then I can use backprop and it becomes efficient? The
formula is similar to a gradient but actually has indices in the wrong places
to be one.

When there are many parameters, it may be more convenient to compute the
jacobian backward for the model function instead of forward for the whole
residuals function, and then apply manually the whitening.

When I do minimization over a nonlinear minimization, is it feasible to compute
the derivative w.r.t. the covariance matrix? Otherwise I would have to resort
to dicrete derivatives (very very slow with many hyperparameters!) => It is
probably fine if I compute the derivative with the local linearization in the
minimum, like I'm doing in lsqfitgp.empbayes_fit.

Is there a reasonable way to support zero errors in prior/data? =>
scipy.optimize supports constraints.

### Hyperparameters

In empbayes_fit per fare la propagazione devo mettermi lì e calcolare le
derivate prime del risultato rispetto ai dati con la funzione implicita quando
minimizzo la marginal likelihood.

Basak et al 2021 spiega che MLE per gli iperparametri dei GP funziona male,
cosa di cui mi ero accorto con lsqfitgp. Però non danno delle gran soluzioni,
bisogna scegliere bene il cambio di variabile sui parametri, e regolarizzare di
forza la matrice di covarianza a priori (GPy usa fino alla 10^-2 relativo alla
sigma dei dati). Come si potrebbe risolvere per davvero questo problema? Con
Bayesian optimization sulla loglikelihood? Però a quel punto va fatta in modo
robusto o è un serpente che si morde la coda. Andrebbe trovato un modello
semplice che si fa quasi tutto analitico... ma se sbaglio es. la scala poi non
becco il minimo, magari ci passo attraverso. Oppure usare un algoritmo ibrido
che usa minimi quadrati per la parte con i residui e poi qualcos'altro per la
parte con il logaritmo? L'errore lo posso impostare da una stima di algebra
lineare numerica dell'errore, c'è in Basak. La sigma e la mu le ottimizzo
analiticamente come spiegato in Basak.

Posso ottenere il posteriore GP con Laplace quando ci sono gli iperparametri,
posto che anche sugli iperparametri ho usato Laplace? (I think not)

In lsqfitgp.empbayes_fit, calcolare la marginal likelihood usando laplace al
livello 2 come ho fatto in BartGP

The marginal likelihood minimization often complains about "loss of precision".
I suspect this is due to the log determinant term. It is not well defined
because the regularization I apply to the matrix changes its value, maybe
that's the problem? How should I regularize to have a smooth logdet without
adding a swamp of white noise like GPy and pymc3? Maybe I could add white noise
only for the logdet term. Most decompositions should support adding a multiple
of the identity.

Consider PyBADS (minimization) and PyVBMC (variational posterior) for inference.
They are mantained python packages implementing algorithms designed for
expensive (> 0.1s) and inaccurate likelihood evaluations, with < 20 parameters.
They won't be optimal because I also have the gradient, but at least they won't
hang in places where the linalg error is large.

### Weird input

Awkward arrays may be interesting as input format.

Support taking derivatives in arbitrarily nested dtypes. Switch from the
mess-tuple format to dictionaries dim->int. Dim can be (a tuple of) str, int,
ellipsis, slices, the same recursive format I should support in
`Kernel.__init__`. Add a class just for parsing the subfield specification.
Also, call these classes Dim and Dims such that they can be used for other
things that need a dimension specification, like Fourier transforms.

Can I do something helpful for Riemann manifolds? Like, the wikipedia page for
processes has a nice brownian motion on a sphere. How could I implement that in
a generic way which would allow the user to put any process on any manifold? =>
has been done now, see Terenin's web page. For vector fields they didn't manage
more than the naive embedding, but for scalar fields they solved the problem.
=> Wait, Terenin's doing an *input* manifold, random walk on sphere is *output*
manifold, it's different.

### Non-Gaussianity

Non-gaussian likelihoods, accessible with a string optional parameter in
GP.marginal_likelihood.

Student-t processes. There are some limitations. Possible interface: a nu
parameter in addx. Can I mix different nu in the same fit? Intepreting the
result would be a bit obscure, raniter will always sample normal distributions.
=> It's not really useful because it's equivalent to an additional variance
scale hyper.

### Kernel operations

In general I can apply to a kernel (or more kernels) any function which has
a Taylor series with nonnegative coefficients. (Got the idea from a seminar by
Hensman, I think this thing should be standard anyway, although I don't
remember reading it in Rasmussen's book.)

When doing kernel base power, or in general applying functions which have a
narrow domain, the kernel must be within bounds. For weakly stationary kernels,
or in general for bounded variance kernels, it is always possible to rescale
the kernel into the domain, so it is not really a problem.

Turning bands operator (Gneiting 2002, p. 501). Applies to isotropic kernels,
introduces negative correlations. Preserves support and derivability.

Replace `forcekron` with `forcesep` with possible values None, 'prod', 'sum'.

### New specific kernels/kernel options

Is there a smooth version of the Wiener process? like, softmin(x, y)? I tried
smoothing it with a gaussian but an undoable integral comes up.

From GPy: periodic Matérn up to a certain k.

Graph kernels from pyGPs:
https://www.cse.wustl.edu/~m.neumann/pyGPs_doc/Graph.html. I also collected
some articles somewhere.

il kernel Taylor posso ricavarlo analiticamente anche per funzioni
che vanno giù più velocemente? Tipo (1/k!)^2n si riesce a fare? Scommetterei di
sì visto che (1/k!)^2 si può
=> WolframAlpha says:
    `sum_k=0^oo x^k/(k!)^n = 0_F_(n-1)([], [1]*(n-1), x)`
But scipy.special does not implement the generalized hypergeometric function.

Add order: int >= 0 parameter to Gibbs that rescales the kernel by
(s(x)s(y))^order, useful when taking derivatives to have the derivative of
order `order` with constant variance (check this is actually the case). =>
This is very generic so maybe it does not belong to the Gibbs? But the Gibbs
is the only one accepting directly the scale instead of the corresponding
transformation.

Implement the kernels from Smola and Kondor (2003) (graphs).

Kernels on graphs from Nicolentzos et al. (2019): see their GraKeL library.
Also there was a library for graphs in python, I don't remember the name.

Classes: RandomWalkKernel, PeriodicKernel (che non è una sottoclasse di
StationaryKernel, perché in linea di principio può anche essere non
stazionario, lo aggiungo con ereditarietà multipla solo come flag).

Look in the kernels reference I highlighted in the GPML bibliography
(Abrahamsen 1997). => Spherical (p. 40) Gives formulas in dimensions 1, 2, 3,
5. Appears generalizable for odd n, but maybe not for even n, is that why it
doesn't give n=4? Is it a special case of Wendland? => I think not. Cubic (p.
41) is similarly a polynomial, but works in all dimensions I think.

Multi-fractional brownian motion, see Lim and Teo (2009) => imperscrutable math

Kernel with the Dirichlet function (scipy.special.diric) =>
https://en.wikipedia.org/wiki/Dirichlet_kernel,
https://francisbach.com/information-theory-with-kernel-methods/

ARIMA, and something to flatten a VARIMA and back, and continuous equivalent.

splines (somewhere on Rasmussen's)

truncated power (Gneiting 2002, p. 501).

compact poly-harmonic (Gneiting 2002, p. 504).

The autocorrelation of any distribution is pos def stationary, in particular
the intersection volume of a solid and a translated copy of it.

Generalized Wendland and its Matérn-limit reparametrization (Bevilaqua 2019).

erfc transformation of a variogram (Stein 2005, eq. 3)

F-covariance on the sphere (Alegria 2021)

Generalized Gibbs nonstationary transformation (Paciorek 2006).

Bernoulli (my old Fourier) (Guinness 2016).

The covariance functions from Porcu (2018, p. 370).

Kernels on spheres with option to use chordal or great arc distance. New class?

Dagum (Gneiting 2013, p. 1346)

Gneiting class eq. 14

Should I standardize all periodic kernels to period 1 or period 2pi? The
problem is that machine pi is not accurate, so depending on how the function is
written, using 2 pi as period can be less accurate than using 1. However, see
https://www.gnu.org/software/gsl/doc/html/specfunc.html#restriction-functions.

https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/kernels/#KernelFunctions.RationalKernel

truncated linear form GPy
https://gpy.readthedocs.io/en/deploy/GPy.kern.src.html#module-GPy.kern.src.trunclinear

### Transformations

#### Pointwise infinite transformations

It would be convienent to be able to do the following: define a tensor-shaped
process. Example: take a gradient. Currently this must be managed manually by
the user, putting the tensor indices in the domain type, and then evaluating
the process on an appropriately shaped array of points, for each separate set
of points. I think it is appropriate that the kernels work on scalar processes
only and that the tensor indices must be represented explicitly in the domain
because this avoids shape-related bugs and is the most general and easier to
think about way when writing the kernel. So the functionality should be added
at the level of `GP`. To work with non-independent components, however, it
would be necessary that the `GP` method forcibly sets some fields of x based on
a rule. A simpler alternative would be to add a `shape` parameter to `addproc`,
assuming independent processes, and add the possibility of (constant) matrices
as factors in `addproctransf`.

#### Finite transformations

It could be convenient to change the matrix multiplication order based on
efficiency, like when using backprop vs. forward. Now I'm always doing
forward. It should be feasible to do backward at least, for when the output
has less axes than the inputs. The concrete recurring case would be
sums/integrals constraints. => Maybe opt_einsum has algorithms to do more
complicated optimization. => I may use a qinfo library for tensor contraction.

`pred` should notice when the conditioning is on a transformed variable that
starts from a lower dimensional set of variables and thus the prior covariance
matrix is rank deficient. => Actually, it is sufficient to find a "bottleneck"
somewhere along the matrix multiplications, not necessarily in the starting
matrix.

I can not currently use directly a fft because it returns complex output. The
user needs to manually separate the real and imaginary parts. (It's still quite
usable)

Discrete derivatives. They can be done manually, but they are common with ARIMA
so it may be convenient to have a method.

#### Fourier
  
Apply a Fourier transform or a DTFT on a kernel which represents a transient
process (should be doable for the gaussian kernel rescaled with gaussians).

Leggere l'articolo "Efficient Fourier representations of families of Gaussian
processes".

Should I combine someway derivatives, multiplications by functions and Fourier?
It would be complicated in full generality. I could just implement someway
Fourier(Derivative) = Multiplication(Fourier), I bet this would turn out
useful. Also Derivative(Taylor) = Multiplication(Taylor), etc.

#### Taylor

In general a kernel derived from a white Taylor coefficients series can be
written as

    k(x, y) = sum_k=0^oo c(k)x^k c(k)y^k =
            = sum_k=0^oo (xy)^k c^2(k)

Thus any function with nonnegative Taylor coefficients is a valid kernel if
evaluated in `xy`. This also follows from the fact that `x,y->xy` is positive
definite and positive coefficients functions produce other positive definite
kernels. So any univariate kernel operation applied to the 1D dot product
kernel that has analytically known Taylor coefficients is a valid candidate to
implement the Taylor transformation.

Thus I should redefine the Taylor kernel to be `xy` and make it such that the
Taylor transformation remains properly defined as kernel operations are applied
to it.

How would this work in multiple dimensions? Does it amount just to start from
the dot product kernel?

It may be possible to recycle jax.experimental.jet to do these things.

Can I compose efficiently multiple positive-series Taylor functions, given the
generator of the series?

#### Other infinite transformations

Mellin transform?

Any orthogonal complete set that admits an analytically summable series with a
family of random access sequences of coefficients is ok.

In general I could write a method "diagonalization" that generates
cross-covariances with a numerable basis that diagonalizes a kernel. This would
be useful for sparse approximations. => Actually what I need is any
symmetric decomposition of the covariance operator, need not be a full blown
diagonalization.

RatQuad is a gamma scale mixture of ExpQuad, so I could implement the
corresponding transformation easily. In general series of kernels support this.
=> Wait, nope: being an integral, the individual expquad processes have zero
variance.

Lamperti transformation (stationary to self similar), see Lim and Li (2006).

#### Nonlinear transformations

Shteno supports analytical multiplications of GPs with moment matching. Can I
do that for general transformations by taking hessians with JAX? The problem I
expect with multiplications is that if the mean is zero (as I'm doing) then it
won't work well because the linearization is crap. I should really support mean
functions at some point.

#### Replace `BufferDict` with pytrees

Useful functions: `jax.flatten_util.ravel_tree`, `jax.tree_util.tree_transpose`.

### Discrete likelihoods

I'd like to avoid introducing a different likelihood and Laplace and EP. In the
user guide I did text classification in my own crappy but quick and still
statistically well-defined way, well it turns out that technique has been
studied and it works almost as well, it's in GPML section 6.5 "Least-squares
Classification".

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

Per chiarire la questione della classificazione esatta con i processi
Gaussiani, posso raccontare una storiella così: ci sono tre personaggi, il Sage
classifier, lo Human classifier e il Machine classifier. Il Sage è la verità
personificata ma nessuno lo vede da tempo. Sa il valore logico delle
proposizioni (x,i) = "l'oggetto i appartiene alla categoria i". Lo Human
classifier cerca di indovinare cosa sa il Sage, quindi tira fuori delle
probabilità p(x,i), e anche quelle congiunte p((x,i), (y,j)), etc. Il Machine
cerca di indovinare cosa direbbe lo Human, e usa il modello con il GP latente
mappato sulle p(x,i) con la CDF gaussiana. La macchina si fida del giudiziono
dell'uomo, quindi la probabilità che assegna all'output del saggio, data
l'opinione umana, è p(x,i|p_H(x,i))=p_H(x,i). Quindi la macchina prende il suo
posteriore sulle p_H e calcola p(x,i) marginalizzato. Se il GP ha gli
iperparametri diventa più complicato perché marginalizzare p_H non si riesce.

Maybe this appears as "linear probability modeling" in the literature?

=> After trying this out in a more realistic context, I can say this does not
work well in practice. It tends to either overfit or underfit. The problem is
that if the error variance is large, then you can't get extreme probabilities,
while if it's small, then you get overfitting. So in the well-separated text
classification example it works because overfitting is fine. More importantly, I
now have to actually implement non-Gaussian likelihoods. Methods: Laplace, EP,
Polya-Gamma. Reading Rasmussen, Laplace does not seem to work well, I should
consider only EP instead. Maybe Polya-Gamma is better though. I should think
about integrating this in the homemade replacement to lsqfit.

For Poisson I can take the square root. How good is it? Ref. Casella-Berger,
pag. 563, ex. 11.1-11. I also wrote something in gppdf.tex.

## Optimization

### `gvar`-related issues

gvar: gvar.dump(..., add_dependencies=True) per essere efficiente
dovrebbe salvare una sottomatrice della matrice di covarianza sparsa interna
anziché usare evalcov_blocks.

gvar.make_fake_data seems to be using evalcov instead of evalcov_blocks,
making it inefficient with many variables with a diagonal covariance matrix,
which is a common case with data.

Make a jax-based rewrite of gvar, centered on an array-like which uses numpy
protocols.

#### Global covariance matrix

The global covariance matrix is in LIL (list of lists) format, and the full
matrix is stored despite being symmetrical. There are two main optimizations
that can be implemented separately: using a compressed sparse format, and
storing only the lower triangular part.

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
with QR) => Useful with Woodsbury, would be used in combined gp-regression.

Can I use the LDL decomposition? On 1000x1000 with scipy.linalg on my laptop,
it is 5x slower than cholesky and 5x faster than diagonalization. It would
avoid positivity problems compared to cholesky. Is it less numerically stable?
Why is it slower? => Pivoting.

Tentative interface for optimizations: addx and addtransf have a flags: str
parameter that accepts options either in short form (single uppercase letters)
or long form (comma separated words) specifying assumed properties of the
corresponding covariance matrix. A more generic method addflags can set the
flags of both diagonal and out of diagonal blocks. Flags by default are None,
when set they become immutable, even if set to an empty string, and calling
addflags raises an exception.

An interface to be designed allows to specify a DAG between the defined
variables, defining directed links and conditional properties (conditional
independence within, deterministic links). Maybe this would require more
hands-on control by the user instead of trying to all optimizations
automatically, I'm not sure.

When a covariance block flag or a DAG assumption is not used, emit a warning.
Can be implemented by flagging the individual objects with "done", false by
default, and then checking if something was not used in the blocks involved.

In GP, allow an option "rank" to fix the rank of eigenvalue truncation instead
of thresholding. It would be useful that the decomposition has a property
"rank" such that the user can get it for a case where it should be the lowest
possible by using GP.decomp.

#### Stationary processes (toeplitz)

With stationary kernels, an evenly spaced input produces a toeplitz matrix,
which requires O(N) memory and can be solved in O(N^2). If the data has
uniform independent errors it's still toeplitz.

The kinds of solvers/decompositions are:
1) Levinson O(n^2)  -> already implemented
1a) Levinson-Trench-Zohar O(n^2), a bit faster for solve  -> see SuperGauss
2) Generalized Schur O(n log^2 n)  -> see SuperGauss
3) PCG O(n log n)  -> see Chan and SuperGauss
4) Schur O(n^2)  -> already implemented
5) Toeplitz Bareiss O(n^2) + O(n^2) space, but stable (see ??)

Book about PCG: Chan 2007, An Introduction to Iterative Toeplitz Solvers
Review of Levinson and Schur: Heinig 2011, Fast algorithms for Toeplitz and Hankel matrices

See https://en.wikipedia.org/wiki/Levinson_recursion for more references.

In general SuperGauss has all these methods implemented.

How to take samples with GSchur:
1) obtain the semencul decomposition V^-1 = AA^T - BB^T with A, B triangular
   toeplitz   O(n log^2 n)
2) take two iid normal vectors za and zb   O(n)
3) y = A za + B zb has then covariance matrix V^-1   O(n log n)
4) x = V y has covariance V   O(n log n)
=> Nope wait, if I do that I get covariance AA^T + BB^T, with a plus sign. How
could I get the minus?

Way to sample in O(n log n) that doesn't always work: use the standard
circulant matrix embedding. If it turns out pos def (can be checked quickly
with FFT), use it for sampling (Graham 2018, alg. 2). A sufficient condition of
pos def (Dietrich 1997, th. 2) is that the first row should be convex,
decreasing and nonnegative (the latter can be relaxed a bit).

If I can compute V^-1 x and V x, can I sample from V? I think not, but it
would allow to use PCG which is O(n log n)

A n-dimensional regular grid with a stationary kernel (even not separable) is
not toeplitz but is nested block toeplitz with n levels, can still be solved
fast probably by adapting toeplitz algorithms to a matrix field. See Graham et
al. 2018.

Since toeplitz-by-vector is O(n log n), I should write a low-rank toeplitz
solver.

##### Periodic stationary processes (circulant)

If furthermore the process is periodic and the grid of points is aligned with
the period, the covariance matrix is circulant and diagonalized by the DFT.

#### Markovian processes (sparse inverse)

Example: the Wiener kernel is a Markov process, i.e. on the right of a
datapoint you don't need the datapoints on the left. This means it can be
solved faster. See how to do this and if it can be integrated smoothly with the
rest of the library.

I think what you get is a Kalman filter.

In GPstuff they also approximate any stationary 1D process with a sufficient
number of hidden Markov processes. => This doesn't work well for long memory
processes, i.e., if the covariance function decays slower than exponentially.

#### Autoregressive processes

Like what Celerite does, it solves 1D GPs in O(N). Reference article to cite:
FAST AND SCALABLE GAUSSIAN PROCESS MODELING WITH APPLICATIONS TO ASTRONOMICAL
TIME SERIES, Daniel Foreman-Mackey, Eric Agol, Sivaram Ambikasaran, and Ruth
Angus.

I should make the Celerite kernel a handwritten subclass since it's a
subalgebra.

#### Kronecker

Can I use the kronecker optimization when the data covariance is non-null? ->
Yes with a reasonable approximation of the marginal likelihood, but the data
covariance must be diagonal. Other desiderata: separation along arbitrary
subsets of the dimensions, it would be important when combining different keys
with addtransf (I don't remember why).

Option to compute only the diagonal of the output covariance matrix, and
allow diagonal-only input covariance for data (will be fundamental for
kronecker). For the output it already works implicitly when using gvars.

Per Kronecker, fare come l'ho pensata per Toeplitz: cioè se ne occupa
totalmente l'oggetto `GP`. C'è un metodo `addgrid` che sa che poi può tenere il
kernel fattorizzato. Anche in questo caso è l'utente che deve preoccuparsi che
il kernel sia separabile. In questo modo funziona anche se ho un kernel che non
è scritto separato ma che sappiamo essere separabile, ad esempio quello
gaussiano, allo stesso modo in cui posso assumere che un array sia equispaziato
anche se non lo è alla precisione numerica dei float. Dovrei una gerarchia con
base InputPoints. Le classi concrete sono `AnyPoints`, `EvenlySpaced`, `Grid`.
`EvenlySpaced` ha un'opzione `stationary:bool`, `Grid` ha `separable:bool`.
Queste opzioni vanno messe nell'oggetto `InputPoints` anziché nel `GP` perché
ad esempio devo poter applicare le opzioni solo a un asse in una griglia.

Yet another interface: add directly the kronecker product as transformation,
something like `addtransf`.

#### Sparse

`GP.addcov` accepts directly sparse matrices (JAX or scipy format), while
`GP.addx`, `GP.addtransf` and `GP.addlintransf` get an additional parameter
`sparsity=None` that accepts a matrix used to represent the sparsity pattern,
with boolean interpretation. This matrix may be dense or sparse. The `Points`
block building method uses the sparsity pattern to make indices to select two
1d arrays of x values, and to shape the result back into a sparse matrix.
`LinTransf` should work as is if jax operations support sparse matrices (or
should I apply `sparsify` automatically if all inputs are sparse?). Then
add some solvers with sparse algorithms, lanczos and lobpcg.

#### Sparse inverse (DAG)

gli inducing points li posso indicare scegliendo una variabile
precedentemente definita con addx/addtransf.

#### Matérn 1D

è uscito sull'arxiv un articolo sull'inferenza esatta 1D con matern
half-integer ("kernel packet"), l'ho scaricato

#### Block diagonal

Use the information on zero blocks. Happens with split components. I could also
search for diagonal blocks in a dense matrix since it's O(n^2).

#### Using multiple solvers

To be more general I could write a "positive definite optimizer", such that it
treats not just inversion but also conditioning. The positive-preserving
operations, starting from a p.d. matrix divided in blocks, are

- submatrix
- schur complement
- sum
- sandwich
- hadamard product
- kronecker product
- inversion

Assumptions on diagonal blocks:

- diagonal
- circulant
- (nested block) toeplitz
- banded
- sparse
- sparse inverse

Assumptions on off-diagonal blocks:

- zero

Can I do this using kanren? https://github.com/pythological/kanren

#### Quantum stuff

See Quimb. For example, they have a fast rank revealing SVD.
