# TODO

Other TODOs are scattered in the code. Search for `TODO`.

## Documentation

Add a Raises section to GP methods that can trigger covariance matrix checks.

Add 'Examples' sections in docstrings

Interlinks with gvar and lsqfit docs.

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

In lsqfit il risultato del fit dovrebbe rappresentare una probabilità
condizionata sui dati. Quindi cosa rappresentano bayesianamente le correlazioni
con i dati? E inoltre: se faccio due fit sugli stessi dati, qual è
l'interpretazione bayesiana della correlazione tra i risultati? Forse queste
cose è meglio se le studio prima su lsqfitgp che è equivalente a un fit lineare
e le formule sono analitiche e scritte chiaramente.

Explain somewhere how to combine processes defined on different variables. I
don't know if this actually works seamlessly, derivatives may break. Surely it
is a bit unelegant due to addx enforcing dtype uniformity. => I should stop
enforcing uniformity after I make sure everything works with derivatives.

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

Add an "Andvanced guide" after the User guide, the first chapters would be
-kernel operations -fourier series -taylor series (taylor afterward because
it has a richer set of kernels implementing but they must be obtained with
transformations) -user defined kernels

In the kernel reference add automatically flags for all the supported
transformations. For loc, scale and other standard stuff striket' them if not
supported. To do this leanly, I need to have the transformation method raise
NotImplementedError right away instead of letting the core callable do it: thus
the implementation methods shall return NotImplemented in place of a callable
to disable the transformations.

The way I specify the hyperprior with transformations of gvars from the point
of view of a statistician is a gaussian copula.

In the kernels reference, generate automatically:

  * random samples in 2D when maxdim > 1 (after removing default forcekron)
  
  * in the index, a flag D0, D1, ..., D∞, D? for the derivability with the
    default parameters

Explain somewhere the jax 32/64 bit thing.

## Fixes and tests

Stabilize Matern kernel near r == 0, then Matern derivatives for real nu
(quick partial fix: larger eps in `IsotropicKernel.__init__`).

Check that float32 is respected.

Test recursive dtype support

In general add extensive tests for StructuredArray.

The minimum derivability warnings are annoying because there are a lot of them
when doing nontrivial things, maybe I should put a warnings filter in GP.pred
such that at most only one warning is emitted per call. => Or maybe not => Add
initialization parameter `verbose` that sets this kind of things. I imagine
that the automatical solving strategy algorithm will emit lots of useful
messages/warnings about what it has decided to do.

Check that conditioning multiple times with zero errors does not change the
result.

In the chapter on integration the samples are very noisy. Some trials with
playground/integrals.py told me that the problem is not gvar.raniter but the
solver used in the prediction. Using eigcut- with a high enough eps makes the
samples smooth, but the result is very sensitive on the choice of cut, and the
samples change shape macroscopically. From this I guess that the problem is
that important eigenvectors of the posterior covariance matrix are not smooth.

Does pred works with empty given? It should allow it and behave like prior.

According to the coverage report, empbayes_fit.__init__ is not executed while
running examples. This is wrong because, for example, pdf4.py uses it. What's
up?

Currently the positivity check is done only on the first call to predfromdata,
on all non-transformed blocks. This means that if I add other blocks afterward
they are not checked. Maybe a more coherent behaviour would be to check all
blocks involved in the conditioning, on every conditioning.

Add jit tests to test_GP

## Implementation details

Chiamare GP -> _GPBase e poi fare una sottoclasse GP e mettere tutti i metodi
di convenienza che non accedono a cose interne in GP.

Fare una github action che carica la release su PyPI quando faccio la release
su github. => Non è che però è un rischio? In questo modo il mio account
github controllerebbe cosa finisce su PyPI perché dovrei metterci le chiavi di
accesso. Dovrei attivare l'autenticazione a due fattori. => Vantaggio: potrei
testare anche l'installazione da PyPI dopo aver caricato la release.

In defining abstract methods, use the single line syntax:

    @abc.abstractmethod
    def cippa(*_): pass

such that there is not need to add # pragma: no cover.

## New functionality

Add mean functions.

### More generic conditioning interface

Other newer projects similar to mine (JuliaGaussianProcesses, tinygp) have a
conditioning method returning a new GP object instead of returning directly the
prediction. I hadn't done something like that because I want to keep the GP
object monolithic to be able to apply nontrivial optimizations with multiple
conditionings, but to add the sampling methods it would be very convenient to
have something like that instead of duplicating everything.

I could have a method addcond analogous to pred. Compared to caching, this
solves the problem of not caching when the data errors are not defined in the
GP. Then pred and sample can accept a cond label instead of a dictionary as
given which points to a store with a cache. Internally both sample and pred
would call an internal part of addcond that does not define an explicit
variable.

Or, once I have caching (see below in "optimization"), sample and pred could
share a lot of internals and the cache would avoid decomposing matrices twice.

### Bayesian optimization

Bayesian optimization: I found only one decent quick-and-lean python package,
bayesian-optimization, but I'm not satisfied with it nor with its
documentation. If I don't find out something better I'd like to study the
matter and write something.

### Low-rank/regression

Se ho un processo somma di due processi di cui uno ha basso rango, in altre
parole gp + minimi quadrati lineare, che credo sia quello che si fa di solito
nel kriging, posso risolvere più efficientemente che gp-style senza modifiche?
E vorrei anche avere i valori dei parametri minimi quadrati. E voglio che
funzioni anche se il priore sui parametri non è già diagonalizzato. E anche se
il priore è uniforme (però rompe la marginal likelihood). (Ci dovrebbe essere
su GPML.)

aggiungere l'opzione per fittare roba di basso rango a parte usando woodsbury.
Deve funzionare sia per regressione lineare con covariate e priore arbitrari,
sia per una componente del kernel di basso rango... forse è chiedere troppo,
per cominciare accontentiamoci di regressione. Interfaccia?? Visto che voglio
mettere addkernel, potrei mettere addregres, e poi sommo con addtransf

### New interfaces

Long-term: move to a variable-oriented approach like gvar instead of the
monolithic GP object I'm doing now. It should be doable because what I'm
doing now with the keys is quite similar to a set of variables, but I have
not clear ideas on the interface. It could be based on underlying default
GP object, like gvar does with its hidden covariance matrix of all primary
gvars.

Fare delle interfacce di alto livello che siano più specifiche, alcune
verrebbero bene come sottoclassi di GP, altre come megafunzioni. Magari potrei
fare anche l'interfaccia plugin per scitkit-learn.

per implementare delle variabili smart gp basandomi su un GP globale,
posso fare un kernel che si aspetta un array strutturato in cui la prima
componente è un indice che seleziona il kernel mentre la seconda contiene il
tipo che serve all'utente. Wait il tipo può cambiare in base al gp... uff... ok
basta mettere l'opzione in addx per non imporre che le x siano tutte dello
stesso tipo. Poi bisogna avere il modo per cancellare/ridefinire le variabili
in un GP per quando faccio assegnamenti temporanei. Le espressioni di gp
chiamano addtransf. Aspetta: potrei voler fare espressioni sia su processi non
valutati che valutati. Allora avrei bisogno di una sorta di abstractgp che si
concretizza quando gli passo le x. Per condizionare uso una funzione globale
perché devo poter mettere insieme vari gp.

If I do the interface with a hidden global GP, use a context manager like
pymc3. Maybe this would also be useful for gvar.

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

forse devo aggiungere a GP stesso i metodi per campionare da un
posteriore/priore per usare le decomposizioni efficienti. Forse dovrei
aggiungere un metodo "addcondition" che definisce una nuova variabile come
condizionata? Pensarci. => O forse per non annegare nei metodi è meglio
resistuire un oggetto "distribuzione" che è in grado di campionare e fare tante
altre cose... però così ho solo spostato il problema perché già GP e le gvar
dovrebbero svolgere questo ruolo.

To sample efficiently from transformations, apply the transformations to the
sample instead of decomposing the covariance matrix, only if the starting
covariance matrix is smaller than the transformed one. More complex case:
parts of these matrices have been already decomposed and are in cache.

### Nonlinear fit

Per fare fit non lineari con lsqfitgp senza lsqfit potrei fare così: uso
empbayes_fit e in makegp inserisco il jacobiano della trasformazione non
lineare con addtransf. => No: dovrei sapere il jacobiano in funzione del
risultato.

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

Is there a reasonable way to support zero errors in prior/data?

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

Option to set starting point (would be useful for Bayesian optimization to
start from previous parameter values)

summary method in empbayes_fit like lsqfit.nonlinear_fit

last gp computed in minimization as meangp attribute in empbayes_fit

The marginal likelihood minimization often complains about "loss of precision".
I suspect this is due to the log determinant term. It is not well defined
because the regularization I apply to the matrix changes its value, maybe
that's the problem? How should I regularize to have a smooth logdet without
adding a swamp of white noise like GPy and pymc3? Maybe I could add white noise
only for the logdet term. Most decompositions should support adding a multiple
of the identity.

Can I use Fisher scoring instead of the actual hessian and jacobian? => This
might be known as "natural gradients" in the machine learning literature but
I'm not sure.

### JAX

I could write a function that converts a JAX function to a gvar function. Then
I could do a version of lsqfit.nonlinear_fit that provides the gvar version of
the jax function provided by the user.

### Weird input

Accept xarray.DataSet and pandas.DataFrame as inputs. Probably I can't use
these as core formats due to autograd.

Also awkward arrays may be interesting as input format.

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

### Kernel operations

In general I can apply to a kernel (or more kernels) any function which has
a Taylor series with nonnegative coefficients. (Got the idea from a seminar by
Hensman, I think this thing should be standard anyway, although I don't
remember reading it in Rasmussen's book.)

I can do (positive scalar) ** kernel (=> implement Kernel.__rpow__). Also
(positive kernel) ** kernel I think.

Standard functions with positive Taylor coefficients: tan, 1/sin, 1/cos, asin,
acos, 1/(1-x), exp, -log(1-x), sinh, cosh, atanh, I_a (real a > -1).

When doing kernel base power, or in general applying functions which have a
narrow domain, the kernel must be within bounds. For weakly stationary kernels,
or in general for bounded variance kernels, it is always possible to rescale
the kernel into the domain, so it is not really a problem.

How do I implement this? I could overload numpy operations, but not for
example for 1/(1-x) or -log(1-x). I wouldn't like to entrust the user to do
operations in the right sequence with a positive Taylor series at the end.
Simplest way: add operations as Kernel methods. Some numpy ufuncs would
even recognize this and so for example np.tan(kernel) would work.

Turning bands operator (Gneiting 2002, p. 501). Applies to isotropic kernels,
introduces negative correlations. Preserves support and derivability.

Look at what the Schoenberg theorem is (seen in seminar by Théo Galy-Fajou)

Tentative coherent implementation of transformations and their compositions:
Each transformation is implemented by a method of the kernel. Kernel provides
three decorators for methods: unarytransf, binarytransf, outputtransf,
inputtransf. The marked method shall return a callable that is used as the core
kernel for the new Kernel object returned. unarytransf is for transformations
that only act on the kernel output, as for example exp(k(x, y)). binarytransf
likewise is for operations on two kernels at a time, like k(x, y) + q(x, y).
outputtransf is for operations that correspond to linear transformations of the
process defined by the kernel, like f(x)k(x, y)f(y). inputtransf is for
transformations of the arguments of the kernel, k(f(x), f(y)). The difference
between unary/binary/input and output is that the latter category actually
needs two (three for cross-kernels) core implementations, acting only on one
argument or on both arguments; so in practice the corresponding decorators
actually need to be a family of decorators similar to @property. Example:

```python
class Cippa(Kernel):
    @outputtransf
    def mytransf(self):
        f = lambda x: x ** 2
        k = self._kernel
        return lambda x, y: f(x) * f(y) * k(x, y)
    @mytransf.left # raises if the method name is different from mytransf
    def mytransf(self):
        f = lambda x: x ** 2
        k = self._kernel
        return lambda x, y: f(x) * k(x, y)
    # now transf.right is defined implicitly in terms of left if not given, but
    # only on kernels and not on crosskernels
```

See `playground/methdec.py` for a sketch of the implementation. The next issue
then is changing the transformations themselves as other transformations are
applied. Example: apply a translation and then fourier. The translation must
modify the fourier implementation to take into account the translation. Other
example: non-linear + fourier. In this case the non-linear transf. should in
some way disable the fourier transformation. To this end, I would define a
class method that can be used within the implementation of the transformations
to list the transformations defined and possibly wrap and redefine them. The
class method would list through the methods and search for a store attribute
set by the decorators, then fetch from it the implementations. The
generic implementations can go in `_KernelBase`, then the basic versions are
instantiated in `CrossKernel`. `Kernel` just acts as a "middleman" used to
flag symmetric covariance functions.

=> Actually, this interface is still crap. It does not support order and
variable specifications like `diff`. So the method must take in the two
arguments, and the decorator has a keyword argument for a class (or callable in
general) used to preprocess the arguments and then do the no-op check (in any
case None works). To support dimensions in `fourier` it is sufficient to use in
the computation the variable we are acting on, assuming the kernel is separable
(currently true for `Fourier`), and keep the other kernel factors unaltered.
Logically, it should be the implementation of the `forcekron` inputtransf that
modifies the `fourier` implementation to act separately. In general this works
for any linear transformation that has a concept of applying to a subvariable,
so maybe it should be a more specific category than `outputtransf`, maybe
`diffliketransf`.

Maybe for elegance and encapsulation the user-provided implementation function
should take in the core kernel instead of `self`, and always be passed the
keyword-only arguments used in the initialization of the kernel, updating the
`__kwdefaults__` of the core. Tentative interface:

```python
class Cippa(Kernel):
    @diffliketransf
    def rescale(kernel, fx, fy, **kw):
        if fy is None:
            return lambda x, y: fx(x) * kernel(x, y)
        if fx is None:
            return lambda x, y: fy(y) * kernel(x, y)
        return lambda x, y: fx(x) * fy(y) * kernel(x, y)
    @diffliketransf(swap=True) # swap = if one is None, pass first the non-None
    def rescale(kernel, fx, fy, **kw):
        if fy is None:
            return lambda x, y: fx(x) * kernel(x, y)
        return lambda x, y: fx(x) * fy(y) * kernel(x, y)
    @diffliketransf(incls=lambda f: (lambda x: 1) if f is None else f)
    # incls = callable used to preprocess the arguments
    def rescale(kernel, fx, fy, **kw):
        return lambda x, y: fx(x) * fy(y) * kernel(x, y)
```

Instead of forbidding Kernel-CrossKernel operations, make Kernel a subclass
of CrossKernel, implement a generic subclass permanence system in _binary
and then let GP raise an error when it receives a CrossKernel. Maybe binary
should be another decorator which I apply to __add__, __mult__ and __pow__.

I can use `__kwdefaults__` to get the default values of kernel parameters
when I decorate them, such that callable derivable and initargs would always
know all the parameters.

Replace `forcekron` with `forcesep` with possible values None, 'prod', 'sum'.
This because I can make diff-like transformations act separately automatically
in both cases.

### New specific kernels/kernel options

Is there a smooth version of the Wiener process? like, softmin(x, y)? I tried
smoothing it with a gaussian but an undoable integral comes up.

From GPy: periodic Matérn up to a certain k.

Graph kernels from pyGPs:
https://www.cse.wustl.edu/~m.neumann/pyGPs_doc/Graph.html. I also collected
some articles somewhere.

Are there interesting discontinuous processes apart from the white noise? =>
I guess they would be something + white noise, so not interesting => nope,
it's sufficient that there is a cusp for r->0, the gammaexp does this for
gamma < 1, matern too for nu < 1/2

il kernel Taylor posso ricavarlo analiticamente anche per funzioni
che vanno giù più velocemente? Tipo (1/k!)^2n si riesce a fare? Scommetterei di
sì visto che (1/k!)^2 si può
=> WolframAlpha says:
    `sum_k=0^oo x^k/(k!)^n = 0_F_(n-1)([], [1]*(n-1), x)`
But scipy.special does not implement the generalized hypergeometric function.

aggiungere kernel BART con maxdepth, default beta=inf, maxdepth=1. Come input
prende gli splitting points, e deve avere un class method per ricavare gli
spitting points dalle X. Per aggiungere il class method usando comunque il
decoratore, definire _BART con il decoratore e poi sottoclassarlo a BART
esplicitamente.

fare anche il BART con la formula esplicita per maxdepth = 2 (quaderno 2022 MAR
12) perché se no è troppo inefficiente. In generale quello con maxdepth
andrebbe calcolato su pochi punti e poi interpolato, l'interpolazione va fatta
con attenzione quando beta è piccolo.

Fare la versione del kernel Fourier con solo il seno. Nel caso n=1 viene il
Brownian bridge. => no, dovrebbe funzionare prendendo la parte dispari
con addproclintransf, fare un esempio di prova

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
stazionario, lo aggiungo con ereditarietà multipla solo come flag)

Look in the kernels reference I highlighted in the GPML bibliography
(Abrahamsen 1997). => Spherical (p. 40) Gives formulas in dimensions 1, 2, 3,
5. Appears generalizable for odd n, but maybe not for even n, is that why it
doesn't give n=4? Is it a special case of PPKernel? => I think not. Cubic (p.
41) is similarly a polynomial, but works in all dimensions I think.

Multi-fractional brownian motion, see Lim and Teo (2009) => imperscrutable math

Can I do a kernel with the Dirichlet function? (scipy.special.diric)

ARIMA, and something to flatten a VARIMA and back

splines

truncated power (Gneiting 2002, p. 501).

compact poly-harmonic (Gneiting 2002, p. 504).

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
complicated optimization.

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

#### Other infinite transformations

Mellin transform?

Any orthogonal complete set that admits an analytically summable series with a
family of random access sequences of coefficients is ok.

In general I could write a method "diagonalization" that generates
cross-covariances with a numerable basis that diagonalizes a kernel. This would
be useful for sparse approximations.

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

### Discrete likelihoods

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

For Poisson I can take the square root. How good is it? Ref. Casella-Berger,
pag. 563, ex. 11.1-11.

## Optimization

### `gvar`-related issues

gvar: gvar.dump(..., add_dependencies=True) per essere efficiente
dovrebbe salvare una sottomatrice della matrice di covarianza sparsa interna
anziché usare evalcov_blocks.

gvar.make_fake_data seems to be using evalcov instead of evalcov_blocks,
making it inefficient with many variables with a diagonal covariance matrix,
which is a common case with data.

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

Memory problem: gvar doesn't allow old correlation matrices which are not used
anymore to be garbage collected, because they are all stored in the single
global covariance matrix. switch_gvar doesn't solve this because it keeps a
chronology of all the matrices, and anyway it can't be used if you have stopped
using all gvars created to that point. => If one cares about efficiency, he
probably won't use gvar in critical paths that are repeated over and over, so
the point may be moot.

### Solvers

DiagLowRank for low rank matrix + multiple of the identity (multiple rank-1
updates to the Cholesky factor? Would it be useful anyway? Maybe it's easier
with QR) => Useful with Woodsbury, would be used in combined gp-regression.

Can I use the LDL decomposition? On 1000x1000 with scipy.linalg on my laptop,
it is 5x slower than cholesky and 5x faster than diagonalization. It would
avoid positivity problems compared to cholesky. Is it less numerically stable?
Why is it slower?

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

#### Sparse inverse (DAG)

gli inducing points li posso indicare scegliendo una variabile
precedentemente definita con addx/addtransf.

#### Matérn 1D

è uscito sull'arxiv un articolo sull'inferenza esatta 1D con matern
half-integer ("kernel packet"), l'ho scaricato

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

L'algoritmo che decide in che ordine risolvere i sistemi lineari a blocchi
metterlo in una funzione a parte visto che è di applicabilità generale. Prima
cerco le componenti connesse, cioè decompongo in diagonale a blocchi la
supermatrice. Poi boh. Sarebbe carino farlo fare a un tesista di informatica,
perché c'è la questione di ottimizzare una specie di roba ad albero con i pesi
dati da polinomi convessi vari.

### Hyperparameters

Try to use empbayes_fit without recreating the GP object each time.
