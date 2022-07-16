# lsqfitgp/_GP.py
#
# Copyright (c) 2020, 2022, Giacomo Petrillo
#
# This file is part of lsqfitgp.
#
# lsqfitgp is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# lsqfitgp is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with lsqfitgp.  If not, see <http://www.gnu.org/licenses/>.

import itertools
import sys
import abc
import warnings

import gvar
import numpy
import jax
from jax import numpy as jnp
from scipy import linalg

from . import _Kernel
from . import _linalg
from . import _array
from . import _Deriv
from . import _patch_jax
from . import _patch_gvar

__all__ = [
    'GP',
]

def _concatenate(alist):
    """
    Decides to use numpy.concatenate or jnp.concatenate depending on the
    input to support gvars.
    """
    if any(a.dtype == object for a in alist):
        return numpy.concatenate(alist)
    else:
        return jnp.concatenate(alist)

def _triu_indices_and_back(n):
    """
    Return indices to get the upper triangular part of a matrix, and indices to
    convert a flat array of upper triangular elements to a symmetric matrix.
    """
    indices = jnp.triu_indices(n)
    q = jnp.empty((n, n), indices[0].dtype)
    a = jnp.arange(len(indices[0]))
    q = q.at[indices].set(a)
    q = q.at[indices[::-1]].set(a)
    return indices, q

def _isarraylike_nostructured(x):
    return numpy.isscalar(x) or jnp.isscalar(x) or isinstance(x, (list, tuple, numpy.ndarray, jnp.ndarray))

def _isarraylike(x):
    return _isarraylike_nostructured(x) or isinstance(x, _array.StructuredArray)

def _isdictlike(x):
    return isinstance(x, (dict, gvar.BufferDict))

def _compatible_dtypes(d1, d2): # pragma: no cover
    """
    Function to check x arrays datatypes passed to GP.addx. If the dtype is
    structured, it checks the structure of the fields is the same, but allows
    casting of concrete dtypes (like, in one array a field can be int, in
    another float, as long as the field name and position is the same).
    Currently not used.

    May not be needed in numpy 1.23 (to be released yet).
    """
    if d1.names != d2.names or d1.shape != d2.shape:
        return False
    if d1.names is not None:
        for name in d1.names:
            if not _compatible_dtypes(d1.fields[name][0], d2.fields[name][0]):
                return False
    else:
        try:
            numpy.result_type(d1, d2) # TODO not strict enough!
        except TypeError:
            return False
    return True

class _Element(metaclass=abc.ABCMeta):
    """
    Abstract class for an object holding information associated to a key in a
    GP object.
    """
    
    @property
    @abc.abstractmethod
    def shape(self): # pragma: no cover
        """Output shape"""
        pass
    
    @property
    def size(self):
        return numpy.prod(self.shape, dtype=int)

class _Points(_Element):
    """Points where the process is evaluated"""
    
    def __init__(self, x, deriv, proc):
        assert _isarraylike(x)
        assert isinstance(deriv, _Deriv.Deriv)
        self.x = x
        self.deriv = deriv
        self.proc = proc
    
    @property
    def shape(self):
        return self.x.shape

class _LinTransf(_Element):
    """Linear transformation of other _Element objects"""
    
    shape = None
    
    def __init__(self, transf, keys, shape):
        self.transf = transf
        self.keys = keys
        self.shape = shape

class _Cov(_Element):
    """User-provided covariance matrix block(s)"""
    
    shape = None
    
    def __init__(self, blocks, shape):
        """blocks = dict (key, key) -> matrices"""
        self.blocks = blocks
        self.shape = shape

class _Proc(metaclass=abc.ABCMeta):
    """
    Abstract base class for an object holding information about a process
    in a GP object.
    """
    
    @abc.abstractmethod
    def __init__(self): # pragma: no cover
        pass
    
class _ProcKernel(_Proc):
    """An independent process defined with a kernel"""
    
    def __init__(self, kernel, deriv=0):
        assert isinstance(kernel, _Kernel.Kernel)
        self.kernel = kernel
        self.deriv = deriv
    
class _ProcTransf(_Proc):
    """A process defined as a linear transformation of other processes"""
    
    def __init__(self, ops, deriv):
        """ops = dict proc key -> callable"""
        self.ops = ops
        self.deriv = deriv

class _ProcLinTransf(_Proc):
    
    def __init__(self, transf, keys, deriv):
        self.transf = transf
        self.keys = keys
        self.deriv = deriv

class _ProcKernelOp(_Proc):
    """A process defined by an operation on the kernel of another process"""
    
    def __init__(self, proc, method, arg):
        """proc = proc key, method = Kernel method, arg = argument to method"""
        self.proc = proc
        self.method = method
        self.arg = arg
        
class _SingletonMeta(type):
    
    def __repr__(cls):
        return cls.__name__

class _Singleton(metaclass=_SingletonMeta):
    
    def __new__(cls):
        raise NotImplementedError(f"{cls.__name__} can not be instantiated")

class DefaultProcess(_Singleton):
    """Key for the default process in GP objects"""
    pass

_zerokernel = _Kernel.Zero()

class GP:
    """
    
    Object that represents a Gaussian process over arbitrary input.
    
    Methods
    -------
    addx
        Add points where the Gaussian process is evaluated.
    addlintransf
        Define a finite linear transformation of the evaluated process.
    addtransf
        Define a finite linear transformation of the evaluated process with
        explicit coefficients.
    addcov
        Introduce a set of user-provided prior covariance matrix blocks.
    addproc
        Define a new independent component of the process.
    addproctransf
        Define a pointwise linear transformation of the process with explicit
        coefficients.
    addproclintransf
        Define a pointwise linear transformation of the process.
    addkernelop
        Define a transformation of the process through a kernel method.
    addprocderiv
        Define a derivative of the process.
    addprocxtransf
        Define a process with transformed inputs.
    addprocrescale
        Define a rescaled process.
    prior
        Compute the prior for the process.
    pred
        Compute the posterior for the process.
    predfromfit
        Like `pred` with ``fromdata=False``.
    predfromdata
        Like `pred` with ``fromdata=True``.
    marginal_likelihood
        Compute the marginal likelihood, i.e., the unconditional probability of
        data.
    
    Parameters
    ----------
    covfun : Kernel or None
        An instance of `Kernel` representing the covariance kernel of the
        default process of the GP object. It can be left unspecified.
    solver : str
        The algorithm used to decompose the prior covariance matrix. See list
        below for the available solvers. Default is `eigcut+` which is slow but
        robust.
        
        'eigcut+'
            Promote small (or negative) eigenvalues to a minimum value.
        'eigcut-'
            Remove small (or negative) eigenvalues.
        'svdcut+'
            Promote small eigenvalues to a minimum value, keeping their sign.
        'svdcut-'
            Remove small eigenvalues.
        'lowrank'
            Reduce the rank of the matrix. The complexity is O(n^2 r) where
            `n` is the matrix size and `r` the required rank, while the other
            algorithms are O(n^3). Slow for small sizes.
        'chol'
            Cholesky decomposition after regularizing the matrix with a
            Gershgorin estimate of the maximum eigenvalue. The fastest of the
            O(n^3) algorithms.
    
    checkpos : bool
        If True (default), raise a `LinAlgError` if the prior covariance matrix
        turns out non positive within numerical error.
    checksym : bool
        If True (default), check that the prior covariance matrix is
        symmetric. If False, only half of the matrix is computed.
    checkfinite : bool
        If True (default), check that the prior covariance matrix does not
        contain infs or nans.
    checklin : bool
        If True (default), the method `addlintransf` will check that the
        given transformation is linear on a random input tensor.
    posepsfac : number
        The threshold used to check if the prior covariance matrix is positive
        definite is multiplied by this factor (default 1).
    **kw
        Additional keyword arguments are passed to the solver.
        
        eps : positive float
            For solvers 'eigcut+', 'eigcut-', 'svdcut+', 'svdcut-', 'chol'.
            Specifies the threshold for considering small the eigenvalues,
            relative to the maximum eigenvalue. The default is matrix size *
            float epsilon.
        rank : positive integer
            For the 'lowrank' solver, the target rank. It should be much
            smaller than the matrix size for the method to be convenient.

    """
    
    # TODO maybe checkpos=True is a too strong default. It can quickly grow
    # beyond doable even if the actual computations remain light. Also, it
    # does not really make sense that it is used only in case of gvar output.
    # Alternative 1: set checkpos=False by default and if True build all the
    # covariance blocks the first time I call _covblock and check everything.
    # Alternative 2: only check the matrices I need to invert. Alternative 3:
    # the default is to check only if the matrices are small enough that it is
    # not a problem. Alternative 4: check diagonal blocks.
    # Current behaviour: only the covariance of things added with addx is
    # checked, and only if gvars are used at some point
    # Alternative 5: do the check in _solver only on things to be solved
    # => Idea: do the full check even when not using gvars, only on
    # non-transformed variables, but issue a warning if the system is large
    
    # TODO maybe the default should be solver='eigcut-' (or 'svdcut-'?) because
    # it adds less noise. It is moot anyway if the plot is done sampling with
    # gvar.raniter or lgp.raninter because they do add their own noise. But this
    # will be solved when I add GP sampling methods.
    
    def __init__(self, covfun=None, solver='eigcut+', checkpos=True, checksym=True, checkfinite=True, checklin=True, posepsfac=1, **kw):
        self._elements = dict() # key -> _Element
        self._covblocks = dict() # (key, key) -> matrix
        self._priordict = gvar.BufferDict({}, dtype=object) # key -> gvar array (shaped)
        self._decompcache = dict() # tuple of keys -> Decomposition
        self._procs = dict() # proc key -> _Proc
        self._kernels = dict() # (proc key, proc key) -> _KernelBase
        if covfun is not None:
            if not isinstance(covfun, _Kernel.Kernel):
                raise TypeError('covariance function must be of class Kernel')
            self._procs[DefaultProcess] = _ProcKernel(covfun)
        decomp = {
            'eigcut+': _linalg.EigCutFullRank,
            'eigcut-': _linalg.EigCutLowRank,
            'svdcut+': _linalg.SVDCutFullRank,
            'svdcut-': _linalg.SVDCutLowRank,
            'lowrank': _linalg.ReduceRank,
            'chol'   : _linalg.CholGersh,
        }[solver]
        self._decompclass = lambda K, **kwargs: decomp(K, **kwargs, **kw)
        self._checkpositive = bool(checkpos)
        self._checkpositive_todo = True
        self._posepsfac = float(posepsfac)
        self._checksym = bool(checksym)
        self._checkfinite = bool(checkfinite)
        self._checklin = bool(checklin)
    
    def addproc(self, kernel=None, key=DefaultProcess, deriv=0):
        """
        
        Add an independent process.
        
        Parameters
        ----------
        kernel : Kernel
            A kernel for the process. If None, use the default kernel. The
            difference between the default process and a process defined with
            the default kernel is that, although they have the same kernel,
            they are independent.
        key : hashable
            The name that identifies the process in the GP object. If not
            specified, sets the kernel of the default process.
        deriv : Deriv-like
            Derivatives to take on the process defined by the kernel.
                
        """
        
        if key in self._procs:
            raise KeyError(f'process key {key!r} already used in GP')
        
        if kernel is None:
            kernel = self._procs[DefaultProcess].kernel
        
        deriv = _Deriv.Deriv(deriv)
        
        self._procs[key] = _ProcKernel(kernel, deriv)
    
    def addproctransf(self, ops, key, deriv=0):
        """
        
        Define a new process as a linear combination of other processes.
        
        Let f_i(x), i = 1, 2, ... be already defined processes, and g_i(x) be
        deterministic functions. The new process is defined as
        
            h(x) = g_1(x) f_1(x) + g_2(x) f_2(x) + ...
        
        Parameters
        ----------
        ops : dict
            A dictionary mapping process keys to scalars or scalar
            functions. The functions must take an argument of the same kind
            of the domain of the process.
        key : hashable
            The name that identifies the new process in the GP object.
        deriv : Deriv-like
            The linear combination is derived as specified by this
            parameter.
        
        """
        
        for k, func in ops.items():
            if k not in self._procs:
                raise KeyError(f'process key {k!r} not in GP object')
            if not jnp.isscalar(func) and not callable(func):
                raise TypeError(f'object of type {type(func)!r} for key {k!r} is neither scalar nor callable')
        
        if key in self._procs:
            raise KeyError(f'process key {key!r} already used in GP')
        
        deriv = _Deriv.Deriv(deriv)
        
        self._procs[key] = _ProcTransf(ops, deriv)
        
        # functions = [
        #     op if callable(op)
        #     else (lambda x: lambda _: x)(op)
        #     for op in ops.values()
        # ]
        # def equivalent_lintransf(*procs):
        #     def fun(x):
        #         out = None
        #         for fun, proc in zip(functions, procs):
        #             this = fun(x) * proc(x)
        #             out = this if out is None else out + this
        #         return out
        #     return fun
        # self.addproclintransf(equivalent_lintransf, list(ops.keys()), key, deriv, False)
    
    def addproclintransf(self, transf, keys, key, deriv=0, checklin=False):
        """
        
        Define a new process as a linear combination of other processes.
        
        Let f_i(x), i = 1, 2, ... be already defined processes, and T
        a linear map from processes to a single process. The new process is
        
            h(x) = T(f_1, f_2, ...)(x).
        
        Parameters
        ----------
        transf : callable
            A function with signature `transf(callable, callable, ...) -> callable`.
        keys : sequence
            The keys of the processes to be passed to the transformation.
        key : hashable
            The name that identifies the new process in the GP object.
        deriv : Deriv-like
            The linear combination is derived as specified by this
            parameter.
        checklin : bool
            If True, check if the transformation is linear. Default False.
        
        Notes
        -----
        The linearity check may fail if the transformation does nontrivial
        operations with the inner function input.
        
        """
        
        if key in self._procs:
            raise KeyError(f'process key {key!r} already used in GP')
    
        for k in keys:
            if k not in self._procs:
                raise KeyError(k)
        
        deriv = _Deriv.Deriv(deriv)
        
        if len(keys) == 0:
            self._procs[key] = _ProcKernel(_zerokernel)
            return
        
        if checklin is None:
            checklin = self._checklin
        if checklin:
            mockup_function = lambda a: lambda _: a
            # TODO this array mockup fails with jax functions
            class Mockup(numpy.ndarray):
                __getitem__ = lambda *_: Mockup((0,))
                __getattr__ = __getitem__
            def checktransf(*arrays):
                functions = [mockup_function(a) for a in arrays]
                return transf(*functions)(Mockup((0,)))
            shapes = [(11,)] * len(keys)
            self._checklinear(checktransf, shapes, elementwise=True)
        
        self._procs[key] = _ProcLinTransf(transf, keys, deriv)

    def addkernelop(self, method, arg, key, proc=DefaultProcess):
        """
        
        Define a new process as the transformation of an existing one.
        
        Parameters
        ----------
        method : str
            A method of `Kernel` taking two arguments which returns a
            transformed kernel.
        arg : object
            A valid argument to the method.
        key : hashable
            Key for the new process.
        proc : hashable
            Key of the process to be transformed. If not specified, use the
            default process.
        
        """
        
        if not hasattr(_Kernel.Kernel, method):
            raise ValueError(f'Kernel has not attribute {method!r}')
        if key in self._procs:
            raise KeyError(f'process key {key!r} already used in GP')
        if proc not in self._procs:
            raise KeyError(f'process {proc!r} not found')
                
        self._procs[key] = _ProcKernelOp(proc, method, arg)
    
    def addprocderiv(self, deriv, key, proc=DefaultProcess):
        """
        
        Define a new process as the derivative of an existing one.
        
        .. math::
            g(x) = \\frac{\\partial^n}{\\partial x^n} f(x)
        
        Parameters
        ----------
        deriv : Deriv-like
            Derivation order.
        key : hashable
            The key of the new process.
        proc : hashable
            The key of the process to be derived. If not specified, use the
            default process.
        
        """
        deriv = _Deriv.Deriv(deriv)
        self.addkernelop('diff', deriv, key, proc)
    
    def addprocxtransf(self, transf, key, proc=DefaultProcess):
        """
        
        Define a new process by transforming the inputs of another one.
        
        .. math::
            g(x) = f(T(x))
        
        Parameters
        ----------
        transf : callable
            A function mapping the new kind input to the input expected by the
            transformed process.
        key : hashable
            The key of the new process.
        proc : hashable
            The key of the process to be transformed. If not specified, use the
            default process.
        
        """
        assert callable(transf)
        self.addkernelop('xtransf', transf, key, proc)
    
    def addprocrescale(self, scalefun, key, proc=DefaultProcess):
        """
        
        Define a new process as a rescaling of an existing one.
        
        .. math::
            g(x) = s(x)f(x)
        
        Parameters
        ----------
        scalefun : callable
            A function from the domain of the process to a scalar.
        key : hashable
            The key of the new process.
        proc : hashable
            The key of the process to be transformed. If not specified, use the
            default process.
        
        """
        assert callable(scalefun)
        self.addkernelop('rescale', scalefun, key, proc)
    
    def addx(self, x, key=None, deriv=0, proc=DefaultProcess):
        """
        
        Add points where the Gaussian process is evaluated.
        
        The GP object keeps the various x arrays in a dictionary. If `x` is an
        array, you have to specify its dictionary key with the `key` parameter.
        Otherwise, you can directly pass a dictionary for `x`.
        
        To specify that on the given `x` a derivative of the process instead of
        the process itself should be evaluated, use the parameter `deriv`.
        
        `addx` never copies the input arrays if they are numpy arrays, so if
        you change their contents before doing something with the GP, the
        change will be reflected on the result. However, after the GP has
        computed internally its covariance matrix, the x are ignored.
        
        Parameters
        ----------
        x : array or dictionary of arrays
            The points to be added.
        key : hashable
            If `x` is an array, the dictionary key under which `x` is added.
            Can not be specified if `x` is a dictionary.
        deriv : Deriv-like
            Derivative specification. A :class:`Deriv` object or something that
            can be converted to :class:`Deriv`.
        proc : hashable
            The process to be evaluated on the points. If not specified, use
            the default process.
        
        """
        
        # TODO after I implement block solving, add per-key covariance matrix
        # flags.
        
        # TODO add `copy` parameter, default False, to copy the input arrays.
        
        deriv = _Deriv.Deriv(deriv)

        if proc not in self._procs:
            raise KeyError(f'process named {proc!r} not found')
                
        if _isarraylike(x):
            if key is None:
                raise ValueError('x is array but key is None')
            x = {key: x}
        elif _isdictlike(x):
            if key is not None:
                raise ValueError('can not specify key if x is a dictionary')
            if None in x:
                raise ValueError('None key in x not allowed')
        else:
            raise TypeError('x must be array or dict')
        
        for key in x:
            if key in self._elements:
                raise KeyError('key {!r} already in GP'.format(key))
            
            gx = x[key]
            
            # Convert to numpy array or StructuredArray.
            if not _isarraylike(gx):
                raise TypeError('x[{!r}] is not array-like'.format(key))
            gx = _array.asarray(gx)

            # Check it is not empty.
            if not gx.size:
                raise ValueError('x[{!r}] is empty'.format(key))
                # TODO it would be elegant to support empty arrays, but how
                # many things would break down from here?

            # Check dtype is compatible with previous arrays.
            # TODO since we never concatenate arrays we could allow a less
            # strict compatibility. In principle we could allow really anything
            # as long as the kernel eats it, but this probably would let bugs
            # through without being really ever useful. What would make sense
            # is checking the dtype structure matches recursively and check
            # concrete dtypes of fields can be casted.
            # TODO result_type is too lax. Examples: str, float -> str,
            # object, float -> object. I should use something like the
            # ordering function in updowncast.py.
            if hasattr(self, '_dtype'):
                try:
                    self._dtype = numpy.result_type(self._dtype, gx.dtype)
                    # do not use jnp.result_type, it does not support
                    # structured types
                except TypeError:
                    msg = 'x[{!r}].dtype = {!r} not compatible with {!r}'
                    msg = msg.format(key, gx.dtype, self._dtype)
                    raise TypeError(msg)
            else:
                self._dtype = gx.dtype

            # Check that the derivative specifications are compatible with the
            # array data type.
            if gx.dtype.names is None:
                if not deriv.implicit:
                    raise ValueError('x has not fields but derivative has')
            else:
                for dim in deriv:
                    if dim not in gx.dtype.names:
                        raise ValueError(f'deriv field {dim!r} not in x')
            
            self._elements[key] = _Points(gx, deriv, proc)
        
    def addtransf(self, tensors, key, axes=1):
        """
        
        Apply a linear transformation to already specified process points. The
        result of the transformation is represented by a new key.
        
        Parameters
        ----------
        tensors : dict
            Dictionary mapping keys of the GP to arrays/scalars. Each array is
            matrix-multiplied with the process array represented by its key,
            while scalars are just multiplied. Finally, the keys are summed
            over.
        key : hashable
            A new key under which the transformation is placed.
        axes : int
            Number of axes to be summed over for matrix multiplication,
            referring to trailing axes for tensors in `tensors`, and to
            heading axes for process points. Default 1.
        
        Notes
        -----
        The multiplication between the tensors and the process is done with
        np.tensordot with, by default, 1-axis contraction. For >2d arrays this
        is different from numpy's matrix multiplication, which would act on the
        second-to-last dimension of the second array.
        
        """        
        # Note: it may seem nice that when an array has less axes than `axes`,
        # the summation would be restricted only on the existing axes. However
        # this brings about the ambiguous case where only one of the factors has
        # not enough axes. How many axes do you sum over on the other?
        
        # Check axes.
        assert isinstance(axes, int), axes
        assert axes >= 1, axes
        
        # Check key.
        if key is None:
            raise ValueError('key can not be None')
        if key in self._elements:
            raise KeyError(f'key {key!r} already in GP')
        
        # Check keys.
        for k in tensors:
            if k not in self._elements:
                raise KeyError(k)
        
        # Check tensors and convert them to jax arrays.
        tens = {}
        for k, t in tensors.items():
            t = jnp.asarray(t)
            # no need to check dtype since jax supports only numerical arrays
            if self._checkfinite and _patch_jax.isconcrete(t):
                T = _patch_jax.concrete(t)
                if not numpy.all(numpy.isfinite(T)):
                    raise ValueError(f'tensors[{k!r}] contains infs/nans')
            rshape = self._elements[k].shape
            if t.shape and t.shape[-axes:] != rshape[:axes]:
                raise ValueError(f'tensors[{k!r}].shape = {t.shape!r} can not be multiplied with shape {rshape!r} with {axes}-axes contraction')
            tens[k] = t
        
        # Compute shape.
        arrays = tens.values()
        elements = (self._elements[k] for k in tens)
        shapes = (
            t.shape[:-axes] + e.shape[axes:] if t.shape else e.shape
            for t, e in zip(arrays, elements)
        )
        try:
            shape = jnp.broadcast_shapes(*shapes)
        except ValueError:
            msg = 'can not broadcast tensors with shapes ['
            msg += ', '.join(repr(t.shape) for t in arrays)
            msg += '] contracted with arrays with shapes ['
            msg += ', '.join(repr(e.shape) for e in elements) + ']'
            raise ValueError(msg)
        
        # TODO fail gracefully with empty input, but we still need to raise
        # since the shape is undetermined. Or we could create an exact scalar 0?
        # the latter would still break operations defined on tensors, so nope,
        # just fail.
        
        # Define linear transformation.
        def equiv_lintransf(*args):
            assert len(args) == len(tens)
            out = None
            for a, (k, t) in zip(args, tens.items()):
                if t.shape:
                    b = jnp.tensordot(t, a, axes)
                else:
                    b = t * a
                if out is None:
                    out = b
                else:
                    out = out + b
            return out
        keys = list(tens.keys())
        self.addlintransf(equiv_lintransf, keys, key, checklin=False)
    
    def addlintransf(self, transf, keys, key, checklin=None):
        """
        
        Define a finite linear transformation of the evaluated process.
        
        Parameters
        ----------
        transf : callable
            A function with signature `f(array1, array2, ...) -> array` which
            computes the linear transformation. The function must be
            jax-traceable, i.e., use jax.numpy instead of numpy.
        keys : sequence
            Keys of parts of the process to be passed as inputs to the
            transformation.
        key : hashable
            The key of the newly defined points.
        checklin : bool
            If True (default), check that the given function is linear in its
            inputs. The default can be overridden at initialization of the GP
            object. Note that an affine function (x -> a + bx) is not linear.
        
        Raises
        ------
        RuntimeError :
            The transformation seems not to be linear. To disable the linearity
            check, initialize the GP with `checklin=False`.
        
        """
        
        # TODO elementwise operations can be applied more efficiently to
        # primary gvars (tipical case), so the method could use an option
        # `elementwise`. What is the reliable way to check it is indeed
        # elementwise with a single random vector? Zero items of the tangent
        # at random with p=0.5 and check they stay zero? (And of course check
        # the shape is preserved.)
        
        # Check key.
        if key is None:
            raise ValueError('key can not be None')
        if key in self._elements:
            raise KeyError(f'key {key!r} already in GP')
        
        # Check keys.
        for k in keys:
            if k not in self._elements:
                raise KeyError(k)
        
        # Determine shape.
        class ArrayMockup:
            def __init__(self, elem):
                self.shape = elem.shape
                self.dtype = float
        inp = [ArrayMockup(self._elements[k]) for k in keys]
        out = jax.eval_shape(transf, *inp)
        shape = out.shape
        
        # Check that the transformation is linear.
        if checklin is None:
            checklin = self._checklin
        if checklin:
            shapes = [self._elements[k].shape for k in keys]
            self._checklinear(transf, shapes)
        
        self._elements[key] = _LinTransf(transf, keys, shape)
    
    def _checklinear(self, func, inshapes, elementwise=False):
        
        # Make input arrays.
        rkey = jax.random.PRNGKey(202206091600)
        inp = []
        for shape in inshapes:
            inp.append(jax.random.normal(rkey, shape))
            _, rkey = jax.random.split(rkey)
        
        # Put zeros into the arrays to check they are preserved.
        if elementwise:
            shape = jnp.broadcast_shapes(*inshapes)
            zeros = jax.random.bernoulli(rkey, 0.5, shape)
            for i, a in enumerate(inp):
                inp[i] = a.at[zeros].set(0)
            
        # Compute JVP and check it is identical to the function itself.
        out0, out1 = jax.jvp(func, inp, inp)
        if _patch_jax.isconcrete(out0, out1):
            out0, out1 = _patch_jax.concrete(out0, out1)
            if out1.dtype == jax.float0:
                cond = numpy.allclose(out0, 0)
            else:
                cond = numpy.allclose(out0, out1)
            if not cond:
                raise RuntimeError('the transformation is not linear')
        
            # Check that the function is elementwise.
            if elementwise:
                if out0.shape != shape or not (numpy.allclose(out0[zeros], 0) and numpy.allclose(out1[zeros], 0)):
                    raise RuntimeError('the transformation is not elementwise')
    
    def addcov(self, covblocks, key=None):
        """
        
        Add user-defined prior covariance matrix blocks.
        
        Covariance matrices defined with `addcov` represent arbitrary
        finite-dimensional zero-mean Gaussian variables, assumed independent
        from all other variables in the GP object.
                
        Parameters
        ----------
        covblocks : array or dictionary of arrays
            If an array: a covariance matrix (or tensor) to be added under key
            `key`. If a dictionary: a mapping from pairs of keys to the
            corresponding covariance matrix blocks. A missing off-diagonal
            block in the dictionary is interpreted as a matrix of zeros,
            unless the corresponding transposed block is specified.
        key : hashable
            If `covblocks` is an array, the dictionary key under which
            `covblocks` is added. Can not be specified if `covblocks` is a
            dictionary.
        
        Raises
        ------
        KeyError :
            A key is already used in the GP.
        ValueError :
            `covblocks` and/or `key` are malformed or inconsistent.
        
        """
        
        # TODO maybe allow passing only the lower/upper triangular part for
        # the diagonal blocks, like I meta-allow for out of diagonal blocks?
        
        # Check type of `covblocks` and standardize it to dictionary.
        if _isarraylike(covblocks):
            if key is None:
                raise ValueError('covblocks is array but key is None')
            covblocks = {(key, key): covblocks}
        elif _isdictlike(covblocks):
            if key is not None:
                raise ValueError('can not specify key if covblocks is a dictionary')
            if None in covblocks:
                raise ValueError('None key in covblocks not allowed')
        else:
            raise TypeError('covblocks must be array or dict')
        
        # Convert blocks to numpy arrays and determine shapes from diagonal
        # blocks.
        shapes = {}
        preblocks = {}
        for keys, block in covblocks.items():
            # TODO maybe check that keys is a 2-tuple
            for key in keys:
                if key in self._elements:
                    raise KeyError(f'key {key!r} already in GP')
            xkey, ykey = keys
            block = jnp.asarray(block)
            if xkey == ykey:
                if block.ndim % 2 == 1:
                    raise ValueError(f'diagonal block {key!r} has odd number of axes')
                half = block.ndim // 2
                head = block.shape[:half]
                tail = block.shape[half:]
                if head != tail:
                    raise ValueError(f'shape {block.shape!r} of diagonal block {key!r} is not symmetric')
                if self._checksym and _patch_jax.isconcrete(block):
                    B = _patch_jax.concrete(block)
                    if not numpy.allclose(B, B.T):
                        raise ValueError(f'diagonal block {key!r} is not symmetric')
                shapes[xkey] = head
            preblocks[keys] = block
        
        # Reshape blocks to square matrices and check that the shapes of out of
        # diagonal blocks match those of diagonal ones.
        blocks = {}
        for keys, block in preblocks.items():
            if self._checkfinite and _patch_jax.isconcrete(block):
                B = _patch_jax.concrete(block)
                if not numpy.all(numpy.isfinite(B)):
                    raise ValueError(f'block {keys!r} not finite')
            xkey, ykey = keys
            if xkey == ykey:
                size = numpy.prod(shapes[xkey], dtype=int)
                blocks[keys] = block.reshape((size, size))
            else:
                for key in keys:
                    if key not in shapes:
                        raise KeyError(f'key {key!r} from off-diagonal block {keys!r} not found in diagonal blocks')
                eshape = shapes[xkey] + shapes[ykey]
                if block.shape != eshape:
                    raise ValueError(f'shape {block.shape!r} of block {keys!r} is not {eshape!r} as expected from diagonal blocks')
                xsize = numpy.prod(shapes[xkey], dtype=int)
                ysize = numpy.prod(shapes[ykey], dtype=int)
                block = block.reshape((xsize, ysize))
                blocks[keys] = block
                revkeys = keys[::-1]
                blockT = preblocks.get(revkeys)
                if blockT is None:
                    blocks[revkeys] = block.T
        
        # Check symmetry of out of diagonal blocks.
        if self._checksym:
            for keys, block in blocks.items():
                xkey, ykey = keys
                if xkey != ykey:
                    blockT = blocks[ykey, xkey]
                    if _patch_jax.isconcrete(block, blockT):
                        B, BT = _patch_jax.concrete(block, blockT)
                        if not numpy.allclose(B.T, BT):
                            raise ValueError(f'block {keys!r} is not the transpose of block {revkeys!r}')
        
        # Create _Cov objects.
        for key, shape in shapes.items():
            self._elements[key] = _Cov(blocks, shape)

    def _crosskernel(self, xpkey, ypkey):
        
        # Check if the kernel is in cache.
        cache = self._kernels.get((xpkey, ypkey))
        if cache is not None:
            return cache
        
        # Compute the kernel.
        xp = self._procs[xpkey]
        yp = self._procs[ypkey]
        
        if isinstance(xp, _ProcKernel) and isinstance(yp, _ProcKernel):
            kernel = self._crosskernel_kernels(xpkey, ypkey)
        elif isinstance(xp, _ProcTransf):
            kernel = self._crosskernel_transf_any(xpkey, ypkey)
        elif isinstance(yp, _ProcTransf):
            kernel = self._crosskernel_transf_any(ypkey, xpkey)._swap()
        elif isinstance(xp, _ProcLinTransf):
            kernel = self._crosskernel_lintransf_any(xpkey, ypkey)
        elif isinstance(yp, _ProcLinTransf):
            kernel = self._crosskernel_lintransf_any(ypkey, xpkey)._swap()
        elif isinstance(xp, _ProcKernelOp):
            kernel = self._crosskernel_op_any(xpkey, ypkey)
        elif isinstance(yp, _ProcKernelOp):
            kernel = self._crosskernel_op_any(ypkey, xpkey)._swap()
        else:
            raise TypeError(f'unrecognized process types {type(xp)!r} and {type(yp)!r}')
        
        # Save cache.
        self._kernels[xpkey, ypkey] = kernel
        self._kernels[ypkey, xpkey] = kernel._swap()
        
        return kernel
        
    def _crosskernel_kernels(self, xpkey, ypkey):
        xp = self._procs[xpkey]
        yp = self._procs[ypkey]
        
        if xp is yp:
            return xp.kernel.diff(xp.deriv, xp.deriv)
        else:
            return _zerokernel
    
    def _crosskernel_transf_any(self, xpkey, ypkey):
        xp = self._procs[xpkey]
        yp = self._procs[ypkey]
        
        kernelsum = _zerokernel
        
        for pkey, factor in xp.ops.items():
            kernel = self._crosskernel(pkey, ypkey)
            if kernel is _zerokernel:
                continue
            
            if not callable(factor):
                factor = (lambda f: lambda _: f)(factor)
            kernel = kernel.rescale(factor, None)
            
            if kernelsum is _zerokernel:
                kernelsum = kernel
            else:
                kernelsum += kernel
                
        return kernelsum.diff(xp.deriv, 0)
    
    def _crosskernel_lintransf_any(self, xpkey, ypkey):
        xp = self._procs[xpkey]
        yp = self._procs[ypkey]
        
        kernels = [self._crosskernel(pk, ypkey) for pk in xp.keys]
        kernel = _Kernel.CrossKernel._nary(xp.transf, kernels, _Kernel.CrossKernel.side.LEFT)
        kernel = kernel.diff(xp.deriv, 0)
        
        return kernel
    
    def _crosskernel_op_any(self, xpkey, ypkey):
        xp = self._procs[xpkey]
        yp = self._procs[ypkey]
        
        if xp is yp:
            basekernel = self._crosskernel(xp.proc, xp.proc)
            # In principle I could avoid handling this case separately but it
            # will probably allow simplifications in how I implement nontrivial
            # transformations in Kernel: I won't need to support taking a
            # transformation in two steps.
        else:
            basekernel = self._crosskernel(xp.proc, ypkey)
        
        if basekernel is _zerokernel:
            return _zerokernel
        elif xp is yp:
            return getattr(basekernel, xp.method)(xp.arg, xp.arg)
        else:
            return getattr(basekernel, xp.method)(xp.arg, None)
    
    def _makecovblock_points(self, xkey, ykey):
        x = self._elements[xkey]
        y = self._elements[ykey]
        
        assert isinstance(x, _Points)
        assert isinstance(y, _Points)
        
        kernel = self._crosskernel(x.proc, y.proc)
        if kernel is _zerokernel:
            # TODO handle zero cov block efficiently
            return jnp.zeros((x.size, y.size))
        
        kernel = kernel.diff(x.deriv, y.deriv)
        
        if x is y and not self._checksym:
            indices, back = _triu_indices_and_back(x.size)
            ax = jnp.broadcast_to(x.x.reshape(-1)[:, None], (x.x.size, y.x.size))[indices]
            ay = jnp.broadcast_to(y.x.reshape(-1)[None, :], (x.x.size, y.x.size))[indices]
            halfcov = kernel(ax, ay)
            cov = halfcov[back]
        else:
            ax = x.x.reshape(-1)[:, None]
            ay = y.x.reshape(-1)[None, :]
            cov = kernel(ax, ay)
        
        return cov
    
    def _makecovblock_lintransf_any(self, xkey, ykey):
        x = self._elements[xkey]
        y = self._elements[ykey]
        assert isinstance(x, _LinTransf)
        
        # Gather covariance matrices to be transformed.
        covs = []
        for k in x.keys:
            elem = self._elements[k]
            cov = self._covblock(k, ykey)
            assert cov.shape == (elem.size, y.size)
            cov = cov.reshape(elem.shape + (y.size,))
            covs.append(cov)
        
        # Apply transformation.
        t = jax.vmap(x.transf, -1, -1)
        cov = t(*covs)
        assert cov.shape == x.shape + (y.size,)
        return cov.reshape((x.size, y.size)) # don't leave out the ()!
        # the () probably was an obscure autograd bug, I don't think it will
        # be a problem again with jax
            
    def _makecovblock(self, xkey, ykey):
        x = self._elements[xkey]
        y = self._elements[ykey]
        if isinstance(x, _Points) and isinstance(y, _Points):
            cov = self._makecovblock_points(xkey, ykey)
        elif isinstance(x, _LinTransf):
            cov = self._makecovblock_lintransf_any(xkey, ykey)
        elif isinstance(y, _LinTransf):
            cov = self._makecovblock_lintransf_any(ykey, xkey)
            cov = cov.T
        elif isinstance(x, _Cov) and isinstance(y, _Cov) and x.blocks is y.blocks and (xkey, ykey) in x.blocks:
            cov = x.blocks[xkey, ykey]
        else:
            # TODO handle zero cov block efficiently
            cov = jnp.zeros((x.size, y.size))
        
        if _patch_jax.isconcrete(cov):
            C = _patch_jax.concrete(cov)
            if self._checkfinite and not numpy.all(numpy.isfinite(C)):
                raise RuntimeError('covariance block {!r} is not finite'.format((xkey, ykey)))
            if self._checksym and xkey == ykey and not numpy.allclose(C, C.T):
                raise RuntimeError('covariance block {!r} is not symmetric'.format((xkey, ykey)))

        return cov

    def _covblock(self, row, col):
        
        if self._checkpositive and self._checkpositive_todo:
            self._checkpositive_todo = False
            keys = [
                k for k, v in self._elements.items()
                if isinstance(v, (_Points, _Cov))
            ]
            fullcov = self._assemblecovblocks(list(keys))
            self._checkpos(fullcov)
            
        if (row, col) not in self._covblocks:
            block = self._makecovblock(row, col)
            if row != col:
                if self._checksym:
                    if _patch_jax.isconcrete(block):
                        blockT = self._makecovblock(col, row)
                        assert _patch_jax.isconcrete(blockT)
                        B, BT = _patch_jax.concrete(block, blockT)
                        if not numpy.allclose(B.T, BT):
                            msg = 'covariance block {!r} is not symmetric'
                            raise RuntimeError(msg.format((row, col)))
                self._covblocks[col, row] = block.T
            self._covblocks[row, col] = block
        
        return self._covblocks[row, col]
        
    def _assemblecovblocks(self, rowkeys, colkeys=None):
        if colkeys is None:
            colkeys = rowkeys
        blocks = [
            [self._covblock(row, col) for col in colkeys]
            for row in rowkeys
        ]
        return jnp.block(blocks)
    
    def _solver(self, keys, ycov=None, **kw):
        """
        Return a decomposition of the covariance matrix of the keys in `keys`
        plus the matrix ycov.
        """
        
        # TODO Block matrix solving. Example: solve a subproblem with kronecker,
        # another plain.
        
        # TODO Check if the matrix is block diagonal. It's O(n^2). It is often
        # not needed but when it is it makes a difference. A faster and less
        # thorough check can be done only on off-diagonal key-key blocks being
        # zero, which may be useful with multi-output or split components.
        
        # TODO cache ignores **kw.
        
        keys = tuple(keys)
        
        # Check if decomposition is in cache.
        if ycov is None:
            cache = self._decompcache.get(keys)
            if cache is not None:
                return cache
            # TODO use frozenset(keys) instead of tuple(keys) to make cache
            # work when order changes, but I have to permute the decomposition
            # to make that work. Needs an ad-hoc class in _linalg.
        
        # Compute decomposition.
        Kxx = self._assemblecovblocks(keys)
        if ycov is not None:
            Kxx = Kxx + ycov
        decomp = self._decompclass(Kxx, **kw)
        
        # Cache decomposition.
        if ycov is None:
            self._decompcache[keys] = decomp
        
        return decomp
        
    def _checkpos(self, cov):
        # TODO faster but less interpretable ways of checking positivity:
        # 1) cholesky + eps, if fails it's not positive
        # 2) ldlt, check each 2x2 block     <--- probably best? is it stable?
        # 3) QR, check diagonal of R
        # For QR, can I use directly the tau coefficients? (halves the
        # computation time, going to 1/4 diagonalization)
        if not _patch_jax.isconcrete(cov):
            return
        eigv = linalg.eigvalsh(_patch_jax.concrete(cov))
        mineigv = numpy.min(eigv)
        if mineigv < 0:
            bound = -len(cov) * numpy.finfo(float).eps * numpy.max(eigv) * self._posepsfac
            if mineigv < bound:
                msg = 'covariance matrix is not positive definite: '
                msg += 'mineigv = {:.4g} < {:.4g}'.format(mineigv, bound)
                raise numpy.linalg.LinAlgError(msg)
    
    def _priorpointscov(self, key):
        
        x = self._elements[key]
        classes = (_Points, _Cov)
        assert isinstance(x, classes)
        mean = numpy.zeros(x.size)
        cov = self._covblock(key, key).astype(float)
        assert cov.shape == 2 * mean.shape, cov.shape
        
        cov = numpy.array(cov) # TODO workaround for gvar issue #27

        # get preexisting primary gvars to be correlated with the new ones
        preitems = [
            k
            for k, px in self._elements.items()
            if isinstance(px, classes)
            and k in self._priordict
        ]
        if preitems:
            prex = numpy.concatenate([
                numpy.reshape(self._priordict[k], -1)
                for k in preitems
            ])
            precov = numpy.concatenate([
                self._covblock(k, key).astype(float)
                for k in preitems
            ])
            g = gvar.gvar(mean, cov, prex, precov, fast=True)
        else:
            g = gvar.gvar(mean, cov, fast=True)
        
        return g.reshape(x.shape)
    
    def _priorlintransf(self, key):
        x = self._elements[key]
        assert isinstance(x, _LinTransf)
        
        # Gather all gvars to be transformed.
        elems = [
            self._prior(k).reshape(-1)
            for k in x.keys
        ]
        g = numpy.concatenate(elems)
        
        # Extract jacobian and split it.
        slices = self._slices(x.keys)
        jac, indices = _patch_gvar.jacobian(g)
        jacs = [
            jac[s].reshape(self._elements[k].shape + indices.shape)
            for s, k in zip(slices, x.keys)
        ]
        # TODO the jacobian can be extracted much more efficiently when the
        # elements are _Points or _Cov, since in that case the gvars are primary
        # and contiguous within each block, so each jacobian is the identity + a
        # range. Then write a function _patch_gvar.merge_jacobians to combine
        # them, which also can be optimized knowing the indices are
        # non-overlapping ranges.
        
        # Apply transformation.
        t = jax.vmap(x.transf, -1, -1)
        outjac = t(*jacs)
        assert outjac.shape == x.shape + indices.shape
        
        # Rebuild gvars.
        outg = _patch_gvar.from_jacobian(numpy.zeros(x.shape), outjac, indices)
        return outg
    
    def _prior(self, key):
        prior = self._priordict.get(key, None)
        if prior is None:
            x = self._elements[key]
            if isinstance(x, (_Points, _Cov)):
                prior = self._priorpointscov(key)
            elif isinstance(x, _LinTransf):
                prior = self._priorlintransf(key)
            else:
                raise TypeError(type(x))
            self._priordict[key] = prior
        return prior
    
    def prior(self, key=None, raw=False):
        """
        
        Return an array or a dictionary of arrays of gvars representing the
        prior for the Gaussian process. The returned object is not unique but
        the gvars stored inside are, so all the correlations are kept between
        objects returned by different calls to `prior`.
        
        Calling without arguments returns the complete prior as a dictionary.
        If you specify `key`, only the array for the requested key is returned.
        
        Parameters
        ----------
        key : None, key or list of keys
            Key(s) corresponding to one passed to `addx` or `addtransf`. None
            for all keys.
        raw : bool
            If True, instead of returning a collection of gvars return
            their covariance matrix as would be returned by `gvar.evalcov`.
            Default False.
        
        Returns
        -------
        If raw=False (default):
        
        prior : np.ndarray or gvar.BufferDict
            A collection of gvars representing the prior.
        
        If raw=True:
        
        cov : np.ndarray or dict
            The covariance matrix of the prior.
        """
        raw = bool(raw)
        
        if key is None:
            outkeys = list(self._elements)
        elif isinstance(key, list):
            outkeys = key
        else:
            outkeys = None
        
        if raw and outkeys is not None:
            return {
                (row, col):
                self._covblock(row, col).reshape(self._elements[row].shape + self._elements[col].shape)
                for row in outkeys
                for col in outkeys
            }
        elif raw:
            return self._covblock(key, key).reshape(2 * self._elements[key].shape)
        elif outkeys is not None:
            return gvar.BufferDict({
                key: self._prior(key) for key in outkeys
            })
        else:
            return self._prior(key)
        
    def _flatgiven(self, given, givencov):
        
        if not hasattr(given, 'keys'):
            raise TypeError('`given` must be dict')
        if givencov is not None and not hasattr(givencov, 'keys'):
            raise TypeError('`givenconv` must be None or dict')
        
        ylist = []
        keylist = []
        for key, l in given.items():
            if key not in self._elements:
                raise KeyError(key)
            
            if not isinstance(l, jnp.ndarray):
                # use numpy since there could be gvars
                l = numpy.asarray(l)
            shape = self._elements[key].shape
            if l.shape != shape:
                msg = 'given[{!r}] has shape {!r} different from shape {!r}'
                raise ValueError(msg.format(key, l.shape, shape))
            if l.dtype != object and not jnp.issubdtype(l.dtype, jnp.number):
                msg = 'given[{!r}] has non-numerical dtype {!r}'
                raise TypeError(msg.format(key, l.dtype))
            
            ylist.append(l.reshape(-1))
            keylist.append(key)
        
        # TODO error checking on the unpacking of givencov
        
        if givencov is None:
            covblocks = None
        else:
            covblocks = [
                [
                    givencov[keylist[i], keylist[j]].reshape(ylist[i].shape + ylist[j].shape)
                    for j in range(len(keylist))
                ]
                for i in range(len(keylist))
            ]
            
        return ylist, keylist, covblocks
    
    def _slices(self, keylist):
        """
        Return list of slices for the positions of flattened arrays
        corresponding to keys in `keylist` into their concatenation.
        """
        sizes = [self._elements[key].size for key in keylist]
        stops = numpy.pad(numpy.cumsum(sizes), (1, 0))
        return [slice(stops[i - 1], stops[i]) for i in range(1, len(stops))]
    
    def pred(self, given, key=None, givencov=None, fromdata=None, raw=False, keepcorr=None):
        """
        
        Compute the posterior.
        
        The posterior can be computed either for all points or for a subset,
        and either directly from data or from a posterior obtained with a fit.
        The latter case is for when the Gaussian process was used in a fit with
        other parameters.
        
        The output is a collection of gvars, either an array or a dictionary
        of arrays. They are properly correlated with gvars returned by
        `prior` and with the input data/fit.
        
        The input is a dictionary of arrays, `given`, with keys corresponding
        to the keys in the GP as added by `addx` or `addtransf`.
        
        Parameters
        ----------
        given : dictionary of arrays
            The data or fit result for some/all of the points in the GP.
            The arrays can contain either gvars or normal numbers, the latter
            being equivalent to zero-uncertainty gvars.
        key : None, key or list of keys, optional
            If None, compute the posterior for all points in the GP (also those
            used in `given`). Otherwise only those specified by key.
        givencov : dictionary of arrays, optional
            Covariance matrix of `given`. If not specified, the covariance
            is extracted from `given` with `gvar.evalcov(given)`.
        fromdata : bool
            Mandatory. Specify if the contents of `given` are data or already
            a posterior.
        raw : bool, optional
            If True, instead of returning a collection of gvars, return
            the mean and the covariance. When the mean is a dictionary, the
            covariance is a dictionary whose keys are pairs of keys of the
            mean (the same format used by `gvar.evalcov`). Default False.
        keepcorr : bool, optional
            If True (default), the returned gvars are correlated with the
            prior and the data/fit. If False, they have the correct covariance
            between themselves, but are independent from all other preexisting
            gvars.
        
        Returns
        -------
        If raw=False (default):
        
        posterior : array or dictionary of arrays
            A collections of gvars representing the posterior.
        
        If raw=True:
        
        pmean : array or dictionary of arrays
            The mean of the posterior. Equivalent to `gvar.mean(posterior)`.
        pcov : 2D array or dictionary of 2D arrays
            The covariance matrix of the posterior. If `pmean` is a dictionary,
            the keys of `pcov` are pairs of keys of `pmean`. Equivalent to
            `gvar.evalcov(posterior)`.
        
        """
        
        if fromdata is None:
            raise ValueError('you must specify if `given` is data or fit result')
        fromdata = bool(fromdata)
        raw = bool(raw)
        if keepcorr is None:
            keepcorr = not raw
        if keepcorr and raw:
            raise ValueError('both keepcorr=True and raw=True')
        
        strip = False
        if key is None:
            outkeys = list(self._elements)
        elif isinstance(key, list):
            outkeys = key
        else:
            outkeys = [key]
            strip = True
        outslices = self._slices(outkeys)
        
        ylist, inkeys, ycovblocks = self._flatgiven(given, givencov)
        y = _concatenate(ylist)
        
        Kxxs = self._assemblecovblocks(inkeys, outkeys)
        
        if ycovblocks is not None:
            ycov = jnp.block(ycovblocks)
        elif (fromdata or raw or not keepcorr) and y.dtype == object:
            ycov = gvar.evalcov(gvar.gvar(y))
            # TODO use evalcov_blocks
        else:
            ycov = None
        self._check_ycov(ycov)
        
        if raw or not keepcorr or self._checkfinite:
            if isinstance(y, jnp.ndarray):
                ymean = y
            else:
                ymean = gvar.mean(y)
            self._check_ymean(ymean)
        
        if raw or not keepcorr:
            
            Kxsxs = self._assemblecovblocks(outkeys)
            
            if fromdata:
                solver = self._solver(inkeys, ycov)
            else:
                solver = self._solver(inkeys)

            mean = solver.quad(Kxxs, ymean)
            cov = Kxsxs - solver.quad(Kxxs)
            
            if not fromdata:
                # complete formula:
                # A = Kxx^-1 @ Kxxs
                # cov = Kxsxs - A.T @ (Kxx - ycov) @ A
                if ycov is None:
                    pass
                else:
                    A = solver.solve(Kxxs)
                    cov = cov + A.T @ ycov @ A
            
        else: # (keepcorr and not raw)        
            yplist = [numpy.reshape(self._prior(key), -1) for key in inkeys]
            ysplist = [numpy.reshape(self._prior(key), -1) for key in outkeys]
            yp = _concatenate(yplist)
            ysp = _concatenate(ysplist)
            
            mat = ycov if fromdata else None
            y = numpy.asarray(y) # because y - yp fails if y is a jax array
            flatout = ysp + self._solver(inkeys, mat).quad(Kxxs, y - yp)
        
        if raw and not strip:
            meandict = {
                key: mean[slic].reshape(self._elements[key].shape)
                for key, slic in zip(outkeys, outslices)
            }
            
            covdict = {
                (row, col):
                cov[rowslice, colslice].reshape(self._elements[row].shape + self._elements[col].shape)
                for row, rowslice in zip(outkeys, outslices)
                for col, colslice in zip(outkeys, outslices)
            }
            
            return meandict, covdict
            
        elif raw:
            assert len(outkeys) == 1
            mean = mean.reshape(self._elements[outkeys[0]].shape)
            cov = cov.reshape(2 * self._elements[outkeys[0]].shape)
            return mean, cov
        
        elif not keepcorr:
            
            cov = numpy.array(cov) # TODO workaround for gvar issue #27
            
            flatout = gvar.gvar(mean, cov, fast=True)
        
        if not strip:
            return gvar.BufferDict({
                key: flatout[slic].reshape(self._elements[key].shape)
                for key, slic in zip(outkeys, outslices)
            })
            # TODO A Bufferdict can be initialized directly with the 1D buffer
        else:
            assert len(outkeys) == 1
            return flatout.reshape(self._elements[outkeys[0]].shape)
        
    def predfromfit(self, *args, **kw):
        """
        Like `pred` with `fromdata=False`.
        """
        return self.pred(*args, fromdata=False, **kw)
    
    def predfromdata(self, *args, **kw):
        """
        Like `pred` with `fromdata=True`.
        """
        return self.pred(*args, fromdata=True, **kw)
    
    def _prior_decomp(self, given, givencov=None, **kw):
        ylist, inkeys, ycovblocks = self._flatgiven(given, givencov)
        y = _concatenate(ylist)
        
        # Get mean.
        if y.dtype == object:
            ymean = gvar.mean(y)
        else:
            ymean = y
        self._check_ymean(ymean)
        
        # Get covariance matrix.
        if ycovblocks is not None:
            ycov = jnp.block(ycovblocks)
            if y.dtype == object:
                warnings.warn(f'covariance matrix may have been specified both explicitly and with gvars; the explicit one will be used')
        elif y.dtype == object:
            gvary = gvar.gvar(y)
            ycov = gvar.evalcov(gvary)
        else:
            ycov = None
        self._check_ycov(ycov)
        
        decomp = self._solver(inkeys, ycov, **kw)
        return decomp, ymean
    
    def _check_ymean(self, ymean):
        if self._checkfinite and _patch_jax.isconcrete(ymean):
            if not numpy.all(numpy.isfinite(_patch_jax.concrete(ymean))):
                raise ValueError('mean of `given` is not finite')
    
    def _check_ycov(self, ycov):
        if ycov is not None and _patch_jax.isconcrete(ycov):
            C = _patch_jax.concrete(ycov)
            if self._checkfinite and not numpy.all(numpy.isfinite(C)):
                raise ValueError('covariance matrix of `given` is not finite')
            if self._checksym and not numpy.allclose(C, C.T):
                raise ValueError('covariance matrix of `given` is not symmetric')

    def marginal_likelihood(self, given, givencov=None, separate=False, **kw):
        """
        
        Compute the logarithm of the probability of the data.
        
        The probability is computed under the Gaussian prior and Gaussian error
        model. It is also called marginal likelihood or bayes factor. If `y` is
        the data and `g` is the Gaussian process, this is
        
        .. math::
            \\log \\int p(y|g) p(g) \\mathrm{d} g.
        
        Unlike `pred()`, you can't compute this with a fit result instead of
        data. If you used the Gaussian process as latent variable in a fit,
        use the whole fit to compute the marginal likelihood. E.g. `lsqfit`
        always computes the logGBF (it's the same thing).
        
        The input is an array or dictionary of arrays, `given`. The contents of
        `given` represent the input data.
                
        Parameters
        ----------
        given : dictionary of arrays
            The data for some/all of the points in the GP. The arrays can
            contain either gvars or normal numbers, the latter being
            equivalent to zero-uncertainty gvars.
        givencov : dictionary of arrays, optional
            Covariance matrix of `given`. If not specified, the covariance
            is extracted from `given` with `gvar.evalcov(given)`.
        separate : bool
            If True, return separately the logdet term and the residuals term.
            Default False.
        **kw :
            Additional keyword arguments are passed to the matrix decomposition.
        
        Returns
        -------
        If ``separate == False`` (default):
        
        logp : scalar
            The logarithm of the marginal likelihood.
        
        If ``separate == True``:
            
        logdet : scalar
            The term of the marginal likelihood containing the logarithm of
            the determinant of the prior covariance matrix, without the -1/2
            factor.
        residuals : array or dictionary of arrays
            A vector whose squared 2-norm multiplied by -1/2 gives the other
            term of the marginal likelihood.
        """
        decomp, ymean = self._prior_decomp(given, givencov, **kw)
        logdet = decomp.logdet() + decomp.n * jnp.log(2 * jnp.pi)
        if separate:
            residuals = decomp.decorrelate(ymean)
            return logdet, residuals
        else:
            return -1/2 * (decomp.quad(ymean) + logdet)
