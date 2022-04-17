# TODO

Other TODOs are scattered in the code. Search for `TODO`.

## Documentation

Add a Raises section to GP methods that can trigger covariance matrix checks.

Add 'Examples' sections.

Interlinks with gvar and lsqfit docs.

Mention that `numpy.lib.recfunctions.unstructured_to_structured` may be used
for euclidean multidimensional input.

Separate the index of kernels by class (after adding `StationaryKernel`).

In the nonlinear fit example, use the gaussian cdf instead of the hyperbolic
arctangent to make a uniform prior over the interval.

A chapter on propagation, from simple GP-only to nonlinear with multiple fits.

mi sono accorto che lepage già praticamente faceva la stessa cosa che
faccio io nell'esempio "y has no errors, marginalization". Comunque si chiama
"regola di Matheron".

Non usare x e y per il kernel perché non si capisce, usare x_1 e x_2.
Aggiungere che il manuale richiede conoscenze di base di algebra lineare e
analisi.

In lsqfit il risultato del fit dovrebbe rappresentare una probabilità
condizionata sui dati. Quindi cosa rappresentano bayesianamente le correlazioni
con i dati? E inoltre: se faccio due fit sugli stessi dati, qual è
l'interpretazione bayesiana della correlazione tra i risultati? Forse queste
cose è meglio se le studio prima su lsqfitgp che è equivalente a un fit lineare
e le formule sono analitiche e scritte chiaramente.

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

## Fixes and tests

Go through the coverage and add tests to cover untested lines.

Stabilize Matern kernel near r == 0, then Matern derivatives for real nu
(quick partial fix: larger eps in `IsotropicKernel.__init__`).

Check that float32 is respected.

Test recursive dtype support.

Lower thresholds in linalg tests.

gvar.BufferDict should have a prettier repr on the IPython shell. Is there a
standard way to configure a pretty print?

per ovviare al bug della moltiplicazione per scalare del kernel con
autograd, aggiungere dei metodi pubblici espliciti per comporre i kernel

## Implementation details

Usare l'interfaccia numpy `__array_function__` per `StructuredArray`.

Chiamare GP -> _GPBase e poi fare una sottoclasse GP e mettere tutti i metodi
di convenienza che non accedono a cose interne in GP.

Usare i template descritti nell'articolo di scikit-hep che era uscito
sull'arxiv, in particolare per configurare i test con le github actions.

## New functionality

Dovrebbe essere possibile per l'utente inserire i suoi blocchi della matrice di
covarianza senza che nessuno gli rompa i coglioni. Anche dopo aver chiamato
`addx`.

Add mean functions.

### Bayesian optimization

Bayesian optimization: I found only one decent quick-and-lean python package,
bayesian-optimization, but I'm not satisfied with it nor with its
documentation. If I don't find out something better I'd like to study the
matter and write something.

To use gvar it would be necessary to remove the limitation of not being able to
create new primary gvars correlated with old ones. Ask Lepage about this. => I
opened an issue on gvar, let's see.

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

### Efficient sampling

posso implementare correlate/decorrelate per BlockMatrix usando una
sorta di cholesky a blocchi, l'ho scritto sul quaderno il 5 aprile 2022

se la matrice di covarianza a priori dei punti dove voglio calcolare la
predizioni è speciale, riesco a fare qualcosa? Ad esempio penso a una griglia
fitta... forse LOVE (nei preprint salvati) fa qualcosa del genere visto che
diceva di fare fast sampling delle predizioni? => vedi quaderno 7 aprile 2022,
c'è anche il caso di dati assunti indipendenti date le predizioni che è
super-potente.

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
posto che anche sugli iperparametri ho usato Laplace?

In lsqfitgp.empbayes_fit, calcolare la marginal likelihood usando laplace al
livello 2 come ho fatto in BartGP

### Port to JAX

In sostanza l'unico motivo per cui non posso passare da autograd a JAX in
lsqfitgp è che voglio lasciare che l'utente usi le funzioni di gvar quando
scrive la funzione `makegp` in `empbayes_fit`. Se faccio delle classi
array-like posso usare JAX anche se non posso aggiungere la mia classe a quelle
supportate chiamando la funzione di JAX in `__array_ufunc__` sull'array interno
(ad esempio per Toeplitz). (Ma alla fine mi sa che non faccio le classi
array-like.)

Generazione di codice, così dopo che ho testato un fit lo posso mettere da
qualche parte. Forse in realtà se uso jax c'è già il jit che gira su gpu e tpu
e quindi sticazzi.

quando passo a jax e posso usare le hessiane, potrei usare fisher
scoring per l'ottimizzazione cioè prendere il valore atteso dell'hessiana

per far passare JAX attraverso BufferDict in empbayes_fit, se uso il
parametro buf posso passare l'array di JAX senza pluggare jax.np dentro il
codice di gvar. Il problema è come supportare le trasformazioni, quelle invece
richiedono una funzione che si succhi array di oggetti (no jax) però vorrei che
supportasse gli array di JAX. Forse bisogna rompere a Lepage, però bisogna che
venga un meccanismo generico anziché un paciugo che supporta solo JAX.

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
processes has a nice brownian motion on a sphere. How could I implement that
in a generic way which would allow the user to put any process on any manifold?
=> has been done now, see Terenin's web page

### Non-Gaussianity

Non-gaussian likelihoods, accessible with a string optional parameter in
GP.marginal_likelihood.

Student-t processes. There are some limitations. Possible interface: a nu
parameter in addx. Can I mix different nu in the same fit? Intepreting the
result would be a bit obscure, raniter will always sample normal distributions.

### New kernels

Is there a smooth version of the Wiener process? like, softmin(x, y)? I tried
smoothing it with a gaussian but an undoable integral comes up.

From GPy: periodic Matérn up to a certain k.

Graph kernels from pyGPs:
https://www.cse.wustl.edu/~m.neumann/pyGPs_doc/Graph.html. I also collected
some articles somewhere.

Are there interesting discontinuous processes apart from the white noise and
its integrals?

il kernel "cauchy" (1 + r^alpha)^-beta, 0 < alpha <= 2, beta > 0, è
ottenibile a partire da quelli definiti? in realtà Cauchy standard sarebbe come
la Cauchy, cioè alpha=2, beta=1, che equivale al "rational quadratic" per
alpha=1. Comunque direi che manca, aggiungere GenCauchy. Credo sia isotropico.
Citare l'articolo che lo introduce.

il kernel Taylor posso ricavarlo analiticamente anche per funzioni
che vanno giù più velocemente? Tipo (1/k!)^2n si riesce a fare? Scommetterei di
sì visto che (1/k!)^2 si può

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
Brownian bridge.

Circular kernel: 
https://docs.pymc.io/pymc-examples/examples/gaussian_processes/GP-Circular.html

Kernel stazionario sinc. Qual è la sua diagonalizzazione? (Ho scoperto che
diagonalizzare il kernel si chiama teorema di Mercer, o decomposizione di
tizio-caio-nonricordo.)

Kernel rumore rosa, troncato tra due frequenze.

è uscito sull'arxiv un articolo sull'inferenza esatta 1D con matern
half-integer ("kernel packet"), l'ho scaricato

### Transformations

Tentative summary of the plan:

  * GP.addproc to add a kernel. Processes are assumed independent between each
    other. GP.addx requires a process key as input, by default uses the kernel
    specified at GP initialization (which can be omitted).
  
  * GP.addproctransf to sum processes and multiply them by functions of x,
    like addtransf but for the whole function.
  
  * GP.addprocderiv to add a derivative of the process.
  
To build the kernel pieces, Kernel.diff and Kernel.rescale (latter to be
implemented) are applied in sequence recursively. This means that I should
probably take out the logic out of the kernel core built by Kernel.diff.
All this would provide 1) simple way of defining sum of components 2) clear
order of multiplication/derivation.

#### Finite transformations

In `GP.addtransf` vorrei poter moltiplicare element-wise gli array con
broadcasting anziché fare la contrazione. E vorrei poter fare robe tipo la FFT.
E la FFT forse non conviene calcolare il jacobiano ma applicarla in qualche
modo alla matrice di covarianza che devo trasformare. Potrei fare una gerarchia
di oggetti che rappresentano trasformazioni, e un parametro `defaulttransf` che
specifica `TensorMul` di default, e un valore del dizionario di input è
l'argomento dell'`__init__` della trasformazione di default. Probabilmente
queste classi mi conviene metterle in un nuovo file perché possono diventare
tante.

`addtransf` deve anche supportare una trasformazione `Reshape`, può venire
comoda per fare uno stack di cose con il broadcasting, tipo se voglio impilare
delle derivate in un gradiente. Potrei anche supportare broadcast.

in addtransf può mangiarsi un'espressione costruita con operazioni di
numpy su placeholder gp['roba']. Però forse è overkill, questa cosa dovrebbe
risiedere a un livello più alto. => A ben pensarci alla fine viene più
semplice che ricostruire tutte le operazioni di numpy con una gerarchia di
classi come scritto sopra.

#### Fourier

The fourier transform is a linear operator, can I use it like I'm doing with
derivatives? I can't do it for the spectrum of a stationary process because
as soon as I put data in the posterior is non-stationary. In other words, the
correlation between a point and a frequency component is 0.

I can do the following:

  * Apply a DFT to a specific key, can be done in addtransf or similar.
  
  * Compute the Fourier series of a periodic process (surely doable for the
    Fourier kernel).
    
  * Apply a Fourier transform or a DTFT on a kernel which represents a
    transient process (should be doable for the gaussian kernel rescaled with
    gaussians).

Each kernel would need to have its handwritten transformation, with an
interface like `diff()`. For the Fourier series it makes sense to make a new
class `PeriodicKernel`, for the transform and DTFT dunno.

I should also add convolutions. Maybe a method `GP.addconv`.

Leggere l'articolo "Efficient Fourier representations of families of Gaussian
processes".

#### Taylor

Define custom transformations for the "Taylor" kernel.

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

Lepage has done sone updates after my requests, I should look into it and
update the manual accordingly.

gvar: gvar.dump(..., add_dependencies=True) per essere efficiente
dovrebbe salvare una sottomatrice della matrice di covarianza sparsa interna
anziché usare evalcov_blocks.

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
with QR) => Useful with Woodsbury, would be used in combined gp-regression.

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

Checking if input is evenly spaced is nonexact and the user should have control
over this. Possible interface: GP.addx(..., evenlyspaced=True), convenienced
to GP.addlinspace(start, stop, num, ...). In some way the x array is marked
as a linspace.

This only applies to stationary kernels and isotropic kernels in 1D. I can
implement it this way: make an intermediate private Kernel subclass
_ToeplitzKernel which implements `__call__`, StationaryKernel and
IsotropicKernel are then subclasses of _ToeplitzKernel.

Add a 'toeplitz' solver to GP. It fails if the matrix is non-toeplitz,
accepting both toeplitz array-likes and normal matrices to be checked.
Non-toeplitz solvers print a warning if they receive a toeplitz array-like.

Scipy.linalg now has a function to multiply toeplitz matrices.

#### Markovian processes

Example: the Wiener kernel is a Markov process, i.e. on the right of a
datapoint you don't need the datapoints on the left. This means it can be
solved faster. See how to do this and if it can be integrated smoothly with the
rest of the library.

I think what you get is a Kalman filter.

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
