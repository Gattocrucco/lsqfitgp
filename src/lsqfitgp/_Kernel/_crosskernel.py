# lsqfitgp/_Kernel/_crosskernel.py
#
# Copyright (c) 2020, 2022, 2023, Giacomo Petrillo
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

import enum
import functools

import numpy
from jax import numpy as jnp

from .. import _array
from .. import _jaxext

from . import _util

def _greatest_common_superclass(classes):
    # from https://stackoverflow.com/a/25787091/3942284
    classes = [x.mro() for x in classes]
    for x in classes[0]: # pragma: no branch
        if all(x in mro for mro in classes):
            return x
           
class CrossKernel:
    r"""
    
    Base class to represent kernels, i.e., covariance functions.

    A kernel is a two-argument function that computes the covariance between
    two functions at some points according to some probability distribution:

    .. math::
        \mathrm{kernel}(x, y) = \mathrm{Cov}[f(x), g(y)].
    
    `CrossKernel` objects are callable, the signature is ``obj(x, y)``, and
    they can be summed and multiplied between them and with scalars. They
    are immutable; all operations return new objects.

    Parameters
    ----------
    core : callable
        A function with signature ``core(x, y)``, where ``x`` and ``y``
        are two broadcastable numpy arrays, which computes the value of the
        kernel.
    derivable, scale, loc, maxdim, dim :
        If specified, these arguments are passed as arguments to the
        correspondingly named transformations, in the order listed here. See
        `transf.` Briefly: the kernel selects only fields `dim` in the
        input, checks the dimensionality against `maxdim`, transforms as
        ``(x - loc) / scale``, then sets the degree of differentiability. If
        any argument is callable, it is passed `**kw` and must return the
        actual argument.
    forcekron : bool, default False
        If True, apply `Kernel.forcekron` before the other transformations.
    batchbytes : number, optional
        If specified, apply ``.batch(batchbytes)`` to the kernel.
    **kw
        Additional keyword arguments are passed to `core` and saved as
        `initargs` attribute.
    
    Attributes
    ----------
    derivable : pair of int or None
        How many times each function is (mean-square sense) derivable.
        ``sys.maxsize`` if it is smooth. `None` mean unknown.
    initargs : dict
        The `kw` argument. Propagates when the kernel is transformed.
    
    Methods
    -------
    batch :
        Batch the computation of the kernel.
    transf :
        Return a transformed kernel.
    transf_help :
        Return the documentation of a transformation.
    register_transf :
        Register a transformation.
    register_coretransf :
        Register a transformation that acts only on the core.
    register_xtransf :
        Register a transformation that acts only on the input.

    See also
    --------
    Kernel
    
    """
    
    def __new__(cls, core, *,
        dim=None,
        loc=None,
        scale=None,
        forcekron=False,
        derivable=None,
        maxdim=None,
        batchbytes=None,
        **kw,
    ):
        self = super().__new__(cls)
                
        self.initargs = kw        
        self._core = lambda x, y: core(x, y, **kw)
        self._derivable = None, None

        if forcekron:
            self = self.forcekron()

        transf_args = {
            'derivable': derivable,
            'scale': scale,
            'loc': loc,
            'maxdim': maxdim,
            'dim': dim,
        }
        for transfname, arg in transf_args.items():
            if callable(arg):
                arg = arg(**kw)
            if arg is not None:
                self = self.transf(transfname, arg)

        if batchbytes is not None:
            self = self.batch(batchbytes)

        return self

    def __call__(self, x, y):
        x = _array.asarray(x)
        y = _array.asarray(y)
        shape = _array.broadcast(x, y).shape
        result = self._core(x, y)
        assert isinstance(result, (numpy.ndarray, jnp.number, jnp.ndarray))
        assert jnp.issubdtype(result.dtype, jnp.number), result.dtype
        assert result.shape == shape, (result.shape, shape)
        return result

    @property
    def derivable(self):
        return self._derivable

    def _clone(self, core=None, cls=None):
        newself = object.__new__(self.__class__ if cls is None else cls)
        newself.initargs = self.initargs
        newself._core = self._core if core is None else core
        newself._derivable = self._derivable
        return newself

    @staticmethod
    def _newkernel_from(core, kernels):
        """
        Make a new kernel object which is the result of an operation on the
        kernels in `kernels`, with implementation callable `core`.
        """
        assert kernels
        classes = (IsotropicKernel, *map(type, kernels))
        cls = _greatest_common_superclass(classes)
        assert issubclass(cls, __class__)
        self = object.__new__(cls)
        self.initargs = None
        self._core = core
        self._derivable = None, None
        return self
    
    class _side(enum.Enum):
        LEFT = 0
        RIGHT = 1
        BOTH = 2
         
    @classmethod
    def _nary(cls, op, kernels, side):
        
        if side is cls._side.LEFT:
            wrapper = lambda c, _, y: lambda x: c(x, y)
            arg = lambda x, _: x
        elif side is cls._side.RIGHT:
            wrapper = lambda c, x, _: lambda y: c(x, y)
            arg = lambda _, y: y
        elif side is cls._side.BOTH:
            wrapper = lambda c, _, __: lambda xy: c(*xy)
            arg = lambda x, y: (x, y)
        else: # pragma: no cover
            raise KeyError(side)
        
        cores = [k._core for k in kernels]
        def core(x, y):
            wrapped = [wrapper(c, x, y) for c in cores]
            transformed = op(*wrapped)
            return transformed(arg(x, y))
        
        return cls._newkernel_from(core, kernels)

        # TODO propagate the transformations
    
    def _binary(self, value, op):
        if _util.is_numerical_scalar(value):
            unary_op = lambda f: lambda x: op(f, lambda _: value)(x)
            out = self._nary(unary_op, [self], self._side.BOTH)
            out._derivable = self._derivable
            return out
        elif isinstance(value, __class__):
            return self._nary(op, [self, value], self._side.BOTH)
        else:
            return NotImplemented
    
    def __add__(self, value):
        return self._binary(value, lambda f, g: lambda x: f(x) + g(x))
    
    __radd__ = __add__
    
    def __mul__(self, value):
        return self._binary(value, lambda f, g: lambda x: f(x) * g(x))
    
    __rmul__ = __mul__

    def __pow__(self, value):
        if not _util.is_integer_scalar(value):
            return NotImplemented
        with _jaxext.skipifabstract():
            assert value >= 0, value
        return self._binary(value, lambda f, g: lambda x: f(x) ** g(x))
    
    def _swap(self):
        """ permute the arguments (cross kernels are not symmetric) """
        core = self._core
        self = self._clone(core=lambda x, y: core(y, x))
        self._derivable = self._derivable[::-1]
        return self

    def batch(self, maxnbytes):
        """
        Return a batched version of the kernel.

        The batched kernel processes its inputs in chunks to try to limit memory
        usage.

        Parameters
        ----------
        maxnbytes : number
            The maximum number of input bytes per chunk, counted after
            broadcasting the input shapes. Actual broadcasting may not occur if
            not induced by the operations in the kernel.

        Returns
        -------
        batched_kernel : CrossKernel
            The same kernel but with batched computations.
        """
        core = _jaxext.batchufunc(self._core, maxnbytes=maxnbytes)
        return self._clone(core=core)
    
    _transf = {}

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        cls._transf = {}

    @classmethod
    def _crossmro(cls):
        """ MRO iterator excluding subclasses of Kernel """
        return (c for c in cls.mro()[:-1] if not issubclass(c, Kernel))

    @classmethod
    def _gettransf(cls, transfname):
        """
        Find a transformation.

        The transformation is searched following the MRO, skipping subclasses of
        `Kernel`.

        Parameters
        ----------
        transfname : hashable
            The transformation name.

        Returns
        -------
        cls : type
            The class where the transformation was found.
        transf, argparser, doc : tuple
            The objects set by `register_transf`.

        Raises
        ------
        KeyError :
            The transformation was not found.
        """
        for cls in cls._crossmro():
            try:
                return cls, cls._transf[transfname]
            except KeyError:
                pass
        raise KeyError(transfname)

    @classmethod
    def has_transf(cls, transfname):
        """
        Check if a transformation is registered.

        Parameters
        ----------
        transfname : hashable
            The transformation name.

        Returns
        -------
        has_transf : bool
            Whether the transformation is registered.
        """
        try:
            cls._gettransf(transfname)
        except KeyError as exc:
            if exc.args == (transfname,):
                return False
            else: # pragma: no cover
                raise
        else:
            return True

    @classmethod
    def register_transf(cls, transf, transfname=None, argparser=None, doc=None):
        """
        
        Register a transformation for use with `transf`.

        The transformation is registered for the first non-`Kernel`-subclass in
        the MRO and for its subclasses.

        Parameters
        ----------
        transf : callable
            A method ``transf(self, arg1, arg2) -> CrossKernel`` that returns
            the new kernel.
        transfname : hashable, optional.
            The `transfname` parameter to `transf` this transformation will
            be accessible under. If not specified, use the name of `transf`.
        argparser : callable, optional
            A function applied to each of the arguments. It should return
            `None` if the operation is the identity. Not called if the
            argument is `None`.
        doc : str, optional
            The documentation of the transformation returned by `transf_help`.
            If not specified, use the docstring of `transf`.

        Returns
        -------
        transf : callable
            The argument `transf` as is.

        Raises
        ------
        KeyError :
            The transformation is already registered on the target class.

        Notes
        -----
        The function `transf` is called only if `arg1` or `arg2` is not
        `None` before or after conversion with `argparser`.

        See also
        --------
        register_coretransf, register_xtransf, has_transf, transf, transf_help

        """
        # transf(self, argparser(arg1), argparser(arg2)) -> CrossKernel
        if transfname is None:
            transfname = transf.__name__
        if doc is None:
            doc = transf.__doc__
        cls = next(cls._crossmro())
        if transfname in cls._transf:
            raise KeyError(f'transformation {transfname!r} already registered '
                f'for {cls.__name__}')
        cls._transf[transfname] = transf, argparser, doc
        return transf

    # TODO consider renaming `transf` to `linop`.

    def transf(self, transfname, *args):
        r"""

        Transform the kernel to represent a transformation of the functions.

        .. math::
            \mathrm{kernel}(x, y) &= \mathrm{Cov}[f(x), g(y)] \\
            \mathrm{newkernel}(x, y) &= \mathrm{Cov}[T_1(f)(x), T_2(g)(y)]

        Parameters
        ----------
        transfname : hashable
            The name of the transformation. The following transformations are
            defined by `CrossKernel`:

            'xtransf' :
                Arbitrary input transformation.
            'loc' :
                Input translation.
            'scale' :
                Input rescaling.
            'dim' :
                Consider only a subset of dimensions of the input.
            'maxdim' :
                Restrict maximum input dimensionality.
            'diff' :
                Derivative.
            'derivable' :
                Set the degree of differentiability, used by ``'diff'`` for
                error checking.
            'rescale' :
                Multiply the function by another fixed function.
            'normalize' :
                Rescale the process to unit variance.

            Subclasses may define their own.
        *args :
            Two arguments that indicate how to transform each function. If both
            arguments represent the identity, this is a no-op. If there is only
            one argument, it is intended that the two arguments are equal.
            `None` always represents the identity.

        Returns
        -------
        newkernel : CrossKernel
            The transformed kernel.
            
            If the object is not an instance of `Kernel`, or if the two
            arguments differ, the class of `newkernel` is the first superclass
            that defines the transformation.

            If the object is a `Kernel` and the two arguments are equal, the
            class of `newkernel` is the first superclass that defines the
            transformation, or the preceding one in the MRO if that superclass
            is not a subclass of `Kernel` while the preceding one is.

        See also
        --------
        register_transf, register_coretransf, register_xtransf, transf_help
        """

        if len(args) not in (1, 2):
            raise ValueError(f'incorrect number of arguments {len(args)}, '
                'expected 1 or 2')

        cls, (transf, argparser, _) = self._gettransf(transfname)

        if all(a is None for a in args):
            return self
        if argparser:
            conv = lambda x: None if x is None else argparser(x)
            args = tuple(map(conv, args))
        if all(a is None for a in args):
            return self
        
        if len(args) == 1:
            arg1, = arg2, = args
        else:
            arg1, arg2 = args

        if isinstance(self, Kernel) and arg1 == arg2 and not issubclass(cls, Kernel):
            mro = self.__class__.mro()
            pos = mro.index(cls)
            if pos > 0 and issubclass(mro[pos - 1], Kernel):
                cls = mro[pos - 1]
        
        return transf(self, arg1, arg2)._clone(cls=cls)

        # TODO propagate the transformations

    @classmethod
    def transf_help(cls, transfname):
        """
        
        Return the documentation of a transformation.

        Parameters
        ----------
        transfname : hashable
            The name of the transformation.

        Returns
        -------
        doc : str
            The documentation of the transformation.

        """
        _, (_, _, doc) = cls._gettransf(transfname)
        return doc

    @classmethod
    def register_coretransf(cls, coretransf, transfname=None, argparser=None, doc=None):
        """

        Register a transformation that acts only on the core.

        Parameters
        ----------
        coretransf : callable
            A function ``coretransf(core, arg1, arg2) -> newcore``, where
            ``core`` is the function that implements the kernel passed at
            initialization.
        transfname, argparser, doc :
            See `register_transf`.

        Returns
        -------
        transf : callable
            A method in the format of `register_transf` that wraps
            `coretransf`.

        """
        @functools.wraps(coretransf)
        def transf(self, arg1, arg2):
            core = coretransf(self._core, arg1, arg2)
            return self._clone(core=core)
        return cls.register_transf(transf, transfname, argparser, doc)

    @classmethod
    def register_xtransf(cls, xtransf, transfname=None, doc=None):
        """

        Register a transformation that acts only on the input.

        Parameters
        ----------
        xtransf : callable
            A function ``xtransf(arg) -> (transf x: newx)`` that takes in a
            transformation argument and produces a function to transform the
            input. Not called if ``arg`` is `None`.
        transfname, doc :
            See `register_transf`. `argparser` is not provided because its
            functionality can be included in `xtransf`.

        Returns
        -------
        transf : callable
            A method in the format of `register_transf` that wraps `xtransf`.

        """

        @functools.wraps(xtransf)
        def coretransf(core, xfun, yfun):
            if not xfun:
                return lambda x, y: core(x, yfun(y))
            elif not yfun:
                return lambda x, y: core(xfun(x), y)
            else:
                return lambda x, y: core(xfun(x), yfun(y))
        
        return cls.register_coretransf(coretransf, transfname, xtransf, doc)

# TODO add a method similar to transf but for kernel algebra transformations,
# i.e., apply a function with nonnegative power series coefficients. atransf
# for "algebraic transf". => Different plan: the current transf becomes linop.
# transf becomes a more generic thing linop is a particular case of. algop is
# the algebric subcase. Transformation names remain in the same pool, the kind
# is saved with the transf and checked.

# TODO a method inherit_transf to be used in stationary and isotropic to make
# loc and scale preserve the subclass.
