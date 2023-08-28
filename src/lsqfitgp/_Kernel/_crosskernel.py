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
import sys
import collections

import numpy
from jax import numpy as jnp

from .. import _array
from .. import _jaxext

from . import _util

def _least_common_superclass(classes):
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
    scale, loc, derivable, maxdim, dim :
        If specified, these arguments are passed as arguments to the
        correspondingly named operators, in the order listed here. See `linop`.
        Briefly: the kernel selects only fields `dim` in the input, checks the
        dimensionality against `maxdim`, checks there are not too many
        derivatives taken on the arguments, then transforms as ``(x - loc) /
        scale``. If any argument is callable, it is passed `**kw` and must
        return the actual argument. If an argument is a tuple, it is interpreted
        as a pair of arguments.
    forcekron : bool, default False
        If True, apply ``.transf('forcekron')`` to the kernel, before the
        operations above. Available only for `Kernel`.
    batchbytes : number, optional
        If specified, apply ``.batch(batchbytes)`` to the kernel.
    **kw
        Additional keyword arguments are passed to `core`.
    
    Methods
    -------
    batch
    transf
    linop
    algop
    register_transf
    register_linop
    register_corelinop
    register_xtransf
    register_algop
    transf_help
    has_transf

    See also
    --------
    Kernel

    Notes
    -----
    The predefined class hierarchy and the class logic of the transformations
    assume that each kernel class corresponds to a subalgebra, i.e., addition
    and multiplication preserve the class.
    
    """

    __slots__ = '_kw', '_core'
    
    def __new__(cls, core, *,
        scale=None,
        loc=None,
        derivable=None,
        maxdim=None,
        dim=None,
        forcekron=False,
        batchbytes=None,
        **kw,
    ):
        self = super().__new__(cls)
                
        self._kw = kw        
        self._core = lambda x, y: core(x, y, **kw)

        if forcekron:
            self = self.transf('forcekron')

        linop_args = {
            'scale': scale,
            'loc': loc,
            'derivable': derivable,
            'maxdim': maxdim,
            'dim': dim,
        }
        for transfname, arg in linop_args.items():
            if callable(arg):
                arg = arg(**kw)
            if isinstance(arg, tuple):
                self = self.linop(transfname, *arg)
            else:
                self = self.linop(transfname, arg)

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

    def _clone(self, cls=None, **attrs):
        newself = object.__new__(self.__class__ if cls is None else cls)
        newself._kw = self._kw
        newself._core = self._core
        for k, v in attrs.items():
            setattr(newself, k, v)
        return newself

    class _side(enum.Enum):
        LEFT = 0
        RIGHT = 1
         
    @classmethod
    def _nary(cls, op, kernels, side):
        
        if side is cls._side.LEFT:
            wrapper = lambda c, _, y: lambda x: c(x, y)
            arg = lambda x, _: x
        elif side is cls._side.RIGHT:
            wrapper = lambda c, x, _: lambda y: c(x, y)
            arg = lambda _, y: y
        else: # pragma: no cover
            raise KeyError(side)
        
        cores = [k._core for k in kernels]
        def core(x, y):
            wrapped = [wrapper(c, x, y) for c in cores]
            transformed = op(*wrapped)
            return transformed(arg(x, y))
        
        return __class__(core)

    def __add__(self, other):
        return self.algop('add', other)

    __radd__ = __add__

    def __mul__(self, other):
        return self.algop('mul', other)

    __rmul__ = __mul__

    def __pow__(self, other):
        return self.algop('pow', exponent=other)

    def __rpow__(self, other):
        return self.algop('rpow', base=other)
    
    def _swap(self):
        """ permute the arguments """
        return self._clone(
            __class__,
            _core=lambda x, y, core=self._core: core(y, x),
        )

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
        return self._clone(_core=core)
    
    @classmethod
    def _crossmro(cls):
        """ MRO iterator excluding subclasses of Kernel """
        for c in cls.mro(): # pragma: no branch
            if not issubclass(c, Kernel):
                yield c
            if c is __class__:
                break

    _transf = {}

    _Transf = collections.namedtuple('_Transf', ['func', 'doc', 'kind'])

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        cls._transf = {}

    @classmethod
    def _transfmro(cls):
        """ Iterator of superclasses with a _transf attribute """
        for c in cls.mro(): # pragma: no branch
            yield c
            if c is __class__:
                break

    @classmethod
    def _settransf(cls, transfname, transf):
        if transfname in cls._transf:
            raise KeyError(f'transformation {transfname!r} already registered '
                f'for {cls.__name__}')
        cls._transf[transfname] = cls._Transf(*transf)

    @classmethod
    def _gettransf(cls, transfname):
        """
        Find a transformation.

        The transformation is searched following the MRO up to `CrossKernel`.

        Parameters
        ----------
        transfname : hashable
            The transformation name.

        Returns
        -------
        cls : type
            The class where the transformation was found.
        transf, doc, kind : tuple
            The objects set by `register_transf`.

        Raises
        ------
        KeyError :
            The transformation was not found.
        """
        for c in cls._transfmro():
            try:
                return c, c._transf[transfname]
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

        See also
        --------
        transf
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
    def register_transf(cls, func, transfname=None, doc=None, kind=None):
        """
        
        Register a transformation for use with `transf`.

        The transformation will be accessible to subclasses.

        Parameters
        ----------
        func : callable
            A function ``func(tcls, self, *args, **kw) -> object`` implementing
            the transformation, where ``tcls`` is the class that defines the
            transformation.
        transfname : hashable, optional.
            The `transfname` parameter to `transf` this transformation will
            be accessible under. If not specified, use the name of `func`.
        doc : str, optional
            The documentation of the transformation for `transf_help`. If not
            specified, use the docstring of `func`.
        kind : object, optional
            An arbitrary marker.

        Returns
        -------
        func : callable
            The `func` argument as is.

        Raises
        ------
        KeyError :
            The name is already in use for another transformation in the same
            class.

        See also
        --------
        transf

        """
        if transfname is None:
            transfname = func.__name__
        if doc is None:
            doc = func.__doc__
        cls._settransf(transfname, (func, doc, kind))
        return func

    @classmethod
    def inherit_transf(cls, transfname, *, intermediates=False):
        """
        
        Inherit a transformation from a superclass.

        Parameters
        ----------
        transfname : hashable
            The name of the transformation.
        intermediates : bool, default False
            If True, make all superclasses up to the one definining the
            transformation inherit it too.

        Raises
        ------
        KeyError :
            The transformation was not found in any superclass, or the
            transformation is already registered on any of the target classes.

        See also
        --------
        transf

        """
        tcls, transf = cls._gettransf(transfname)
        cls._settransf(transfname, transf)
        if intermediates:
            for c in cls.mro()[1:]: # pragma: no branch
                if c is tcls:
                    break
                c._settransf(transfname, transf)

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

        See also
        --------
        transf

        """
        _, transf = cls._gettransf(transfname)
        return transf.doc

    def transf(self, transfname, *args, **kw):
        """

        Return a transformed kernel.

        Parameters
        ----------
        transfname : hashable
            A name identifying the transformation.
        *args, **kw :
            Arguments to the transformation.

        Returns
        -------
        newkernel : object
            The output of the transformation.

        Raises
        ------
        KeyError
            The transformation is not defined in this class or any superclass.

        See also
        --------
        linop, algop, transf_help, has_transf, register_transf, register_linop, register_corelinop, register_xtransf, register_algop, register_ufuncalgop

        """
        tcls, transf = self._gettransf(transfname)
        return transf.func(tcls, self, *args, **kw)

    @classmethod
    def register_linop(cls, op, transfname=None, doc=None, argparser=None):
        """
        
        Register a transformation for use with `linop`.

        Parameters
        ----------
        op : callable
            A method ``op(self, arg1, arg2) -> CrossKernel`` that returns
            the new kernel.
        transfname, doc : optional
            See `register_transf`.
        argparser : callable, optional
            A function applied to each of the arguments. It should return
            `None` if the operation is the identity. Not called if the
            argument is `None`.

        Returns
        -------
        func : callable
            A transformation in the format required by `register_transf`.

        Notes
        -----
        The function `op` is called only if `arg1` or `arg2` is not `None`
        after potential conversion with `argparser`.

        See also
        --------
        transf

        """
        
        if transfname is None:
            transfname = op.__name__ # for error message
        
        @functools.wraps(op)
        def func(tcls, self, *args):
            
            if len(args) not in (1, 2):
                raise ValueError(f'incorrect number of arguments {len(args)}, '
                    'expected 1 or 2')

            if argparser:
                conv = lambda x: None if x is None else argparser(x)
            else:
                conv = lambda x: x
            
            if len(args) == 1:
                arg = conv(*args)
                different = False
                arg1 = arg2 = arg
            else:
                arg1, arg2 = args
                different = arg1 is not arg2
                arg1 = conv(arg1)
                arg2 = conv(arg2)
                different &= arg1 is not arg2

            if arg1 is None and arg2 is None:
                return self

            result = op(self, arg1, arg2)

            if not isinstance(result, __class__):
                raise TypeError(f'linop {transfname!r} returned '
                    f'object of type {result.__class__.__name__}, expected '
                    f'subclass of {__class__.__name__}')
            
            rcls = result.__class__
            if isinstance(self, Kernel) and different:
                rcls = next(result._crossmro())
            rcls = _least_common_superclass([tcls, rcls])
            if result.__class__ is not rcls:
                result = result._clone(rcls)
            
            return result

        return cls.register_transf(func, transfname, doc, 'linop')

    def linop(self, transfname, *args):
        r"""

        Transform the kernel to represent the application of a linear operator
        to the functions.

        .. math::
            \text{kernel}(x, y) &= \mathrm{Cov}[f(x), g(y)] \\
            \text{newkernel}(x, y) &= \mathrm{Cov}[T_1(f)(x), T_2(g)(y)]

        Parameters
        ----------
        transfname : hashable
            The name of the transformation.
        *args :
            Two arguments that indicate how to transform each function. If both
            arguments represent the identity, this is a no-op. If there is only
            one argument, it is intended that the two arguments are equal.
            `None` always represents the identity.

        Returns
        -------
        newkernel : CrossKernel
            The transformed kernel.

        Raises
        ------
        ValueError :
            The transformation exists but was not defined by `register_linop`.

        See also
        --------
        transf

        Notes
        -----
        If the input object is an instance of `Kernel` and the two arguments are
        not identical, `newkernel` is enforced to have a class which is not a
        subclass of `Kernel`. After that, the least common superclass between
        the result and the class defining the operation is enforced.
        
        `CrossKernel` defines the following operations:

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
        
        """
        tcls, transf = self._gettransf(transfname)
        if transf.kind != 'linop':
            raise ValueError(f'the transformation {transfname!r} was not '
                f'defined with register_linop and so can not be invoked '
                f'by linop')
        return transf.func(tcls, self, *args)

    @classmethod
    def register_corelinop(cls, corefunc, transfname=None, doc=None, argparser=None):
        """

        Register a linear operator with a function that acts only on the core.

        Parameters
        ----------
        corefunc : callable
            A function ``corefunc(core, arg1, arg2) -> newcore``, where
            ``core`` is the function that implements the kernel passed at
            initialization.
        transfname, doc, argparser :
            See `register_linop`.

        Returns
        -------
        func : callable
            A function in the format of `register_transf` that wraps `corefunc`.

        See also
        --------
        transf

        """
        @functools.wraps(corefunc)
        def op(self, arg1, arg2):
            core = corefunc(self._core, arg1, arg2)
            return self._clone(_core=core)
        return cls.register_linop(op, transfname, doc, argparser)

    @classmethod
    def register_xtransf(cls, xfunc, transfname=None, doc=None):
        """

        Register a linear operator that acts only on the input.

        Parameters
        ----------
        xfunc : callable
            A function ``xfunc(arg) -> (transf x: newx)`` that takes in a
            transformation argument and produces a function to transform the
            input. Not called if ``arg`` is `None`.
        transfname, doc :
            See `register_linop`. `argparser` is not provided because its
            functionality can be included in `xfunc`.

        Returns
        -------
        func : callable
            A function in the format of `register_transf` that wraps `xfunc`.

        See also
        --------
        transf

        """

        @functools.wraps(xfunc)
        def corefunc(core, xfun, yfun):
            if not xfun:
                return lambda x, y: core(x, yfun(y))
            elif not yfun:
                return lambda x, y: core(xfun(x), y)
            else:
                return lambda x, y: core(xfun(x), yfun(y))
        
        return cls.register_corelinop(corefunc, transfname, doc, xfunc)

    @classmethod
    def register_algop(cls, op, transfname=None, doc=None):
        """

        Register a transformation for use with `algop`.

        Parameters
        ----------
        func : callable
            A function ``func(*kernels, **kw) -> CrossKernel | NotImplemented``
            that returns the new kernel. ``kernels`` may be scalars but for the
            first argument.
        transfname, doc :
            See `register_transf`.

        Returns
        -------
        func : callable
            A transformation in the format of `register_transf`.

        See also
        --------
        transf

        """

        if transfname is None:
            transfname = op.__name__ # for error message
        
        @functools.wraps(op)
        def func(tcls, *operands, **kw):
            result = op(*operands, **kw)
            
            if result is NotImplemented:
                return result
            elif not isinstance(result, __class__):
                raise TypeError(f'algop {transfname!r} returned '
                    f'object of type {result.__class__.__name__}, expected '
                    f'subclass of {__class__.__name__}')
            
            def classes():
                yield tcls
                for o in operands:
                    if isinstance(o, __class__):
                        yield o.__class__
                    elif _util.is_nonnegative_scalar_trueontracer(o):
                        yield IsotropicKernel
                    elif _util.is_numerical_scalar(o):
                        yield CrossIsotropicKernel
                    else:
                        raise TypeError(f'operands to algop {transfname!r} '
                            f'must be CrossKernel or numbers, found {o!r}')
                        # this type check comes after letting the implementation
                        # return NotImplemented, to support overloading
                yield result.__class__
            
            lcs = _least_common_superclass(classes())
            return result._clone(lcs)
    
        return cls.register_transf(func, transfname, doc, 'algop')

        # TODO delete _kw (also in linop) if there's more than one kernel
        # operand or if the class changed?

    @classmethod
    def register_ufuncalgop(cls, ufunc, transfname=None, doc=None):
        """

        Register an algebraic operation with a function that acts only on the
        kernel value.

        Parameters
        ----------
        corefunc : callable
            A function ``ufunc(*values, **kw) -> value``, where ``values`` are
            the values yielded by the operands.
        transfname, doc :
            See `register_transf`.

        Returns
        -------
        func : callable
            A function in the format of `register_transf` that wraps `corefunc`.

        See also
        --------
        transf

        """
        @functools.wraps(ufunc)
        def op(self, *operands, **kw):
            cores = tuple(
                o._core if isinstance(o, __class__)
                else lambda x, y: o
                for o in (self, *operands)
            )
            def core(x, y):
                values = (core(x, y) for core in cores)
                return ufunc(*values, **kw)
            return self._clone(_core=core)
        return cls.register_algop(op, transfname, doc)

    def algop(self, transfname, *operands, **kw):
        r"""

        Return a positive algebraic transformation of the input kernels.

        .. math::
            \mathrm{newkernel}(x, y) &=
                f(\mathrm{kernel}_1(x, y), \mathrm{kernel}_2(x, y), \ldots), \\
            f(z_1, z_2, \ldots) &= \sum_{k_1,k_2,\ldots=0}^\infty
                a_{k_1 k_2 \ldots} z_1^{k_1} z_2^{k_2} \ldots,
            \quad a_* \ge 0.

        Parameters
        ----------
        transfname : hashable
            A name identifying the transformation.
        *operands : CrossKernel, scalar
            Arguments to the transformation in addition to self.
        **kw :
            Additional arguments to the transformation, not considered as
            operands.

        Returns
        -------
        newkernel : CrossKernel or NotImplemented
            The transformed kernel, or NotImplemented if the operation is
            not supported.

        See also
        --------
        transf

        Notes
        -----
        The class of `newkernel` is the least common superclass of: the
        "natural" output of the operation, the class defining the
        transformation, and the classes of the operands.

        For class determination, scalars in the input count as `IsotropicKernel`
        if nonnegative or traced by jax, else `CrossIsotropicKernel`.

        `CrossKernel` defines the following operations:

        'add' :
            Binary addition.
        'mul' :
            Binary multiplication.
        'pow' :
            Exponentiation to nonnegative integer.

        """
        tcls, transf = self._gettransf(transfname)
        if transf.kind != 'algop':
            raise ValueError(f'the transformation {transfname!r} was not '
                f'defined with register_algop and so can not be invoked '
                f'by algop')
        return transf.func(tcls, self, *operands, **kw)

# TODO method list_transf to list all the available transformations together
# with the class they are defined in and the kind.
