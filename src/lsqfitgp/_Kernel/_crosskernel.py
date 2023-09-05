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
import types
import abc
import warnings

import numpy
from jax import numpy as jnp

from .. import _array
from .. import _jaxext
from .. import _utils

from . import _util

@functools.lru_cache(maxsize=None)
def least_common_superclass(*classes):
    """
    Find a "least" common superclass. The class is searched in all the MROs,
    but the comparison is done with `issubclass` to support virtual inheritance.
    """
    mros = [c.__mro__ for c in classes]
    indices = [0] * len(mros)
    for i, mroi in enumerate(mros):
        for j, mroj in enumerate(mros):
            if i == j:
                continue
            while not issubclass(mroi[0], mroj[indices[j]]):
                indices[j] += 1
    idx = numpy.argmin(indices)
    return mros[idx][indices[idx]]

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
    dynkw : dict, optional
        Additional keyword arguments passed to `core` that can be modified
        by transformations. Deleted by transformations by default.
    **initkw :
        Additional keyword arguments passed to `core` that can be read but not
        changed by transformations.

    Attributes
    ----------
    initkw : dict
        The `initkw` argument.
    dynkw : dict
        The `dynkw` argument, or a modification of it if the object has been
        transformed.
    core : callable
        The `core` argument partially evaluated on `initkw`, or another
        function wrapping it if the object has been transformed.
    
    Methods
    -------
    batch
    linop
    algop
    transf
    register_transf
    register_linop
    register_corelinop
    register_xtransf
    register_algop
    register_ufuncalgop
    transf_help
    has_transf
    list_transf
    super_transf
    make_linop_family

    See also
    --------
    Kernel

    Notes
    -----
    The predefined class hierarchy and the class logic of the transformations
    assume that each kernel class corresponds to a subalgebra, i.e., addition
    and multiplication preserve the class.
    
    """

    __slots__ = '_initkw', '_dynkw', '_core'
        # only __new__ and _clone shall access these attributes

    @property
    def initkw(self):
        return types.MappingProxyType(self._initkw)

    @property
    def dynkw(self):
        return types.MappingProxyType(self._dynkw)

    @property
    def core(self):
        return self._core

    def __new__(cls, core, *,
        scale=None,
        loc=None,
        derivable=None,
        maxdim=None,
        dim=None,
        forcekron=False,
        batchbytes=None,
        dynkw={},
        **initkw,
    ):
        self = super().__new__(cls)
                
        self._initkw = initkw
        self._dynkw = dict(dynkw)
        self._core = lambda x, y, **dynkw: core(x, y, **initkw, **dynkw)

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
                arg = arg(**initkw)
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
        result = self.core(x, y, **self.dynkw)
        assert isinstance(result, (numpy.ndarray, jnp.number, jnp.ndarray))
        assert jnp.issubdtype(result.dtype, jnp.number), result.dtype
        assert result.shape == shape, (result.shape, shape)
        return result

    def _clone(self, cls=None, *, initkw=None, dynkw=None, core=None):
        newself = object.__new__(self.__class__ if cls is None else cls)
        newself._initkw = self._initkw if initkw is None else dict(initkw)
        newself._dynkw = {} if dynkw is None else dict(dynkw)
        newself._core = self._core if core is None else core
        return newself

    class _side(enum.Enum):
        LEFT = 0
        RIGHT = 1
         
    @classmethod
    def _nary(cls, op, kernels, side):
        
        if side is cls._side.LEFT:
            wrapper = lambda c, _, y, **kw: lambda x: c(x, y, **kw)
            arg = lambda x, _: x
        elif side is cls._side.RIGHT:
            wrapper = lambda c, x, _, **kw: lambda y: c(x, y, **kw)
            arg = lambda _, y: y
        else: # pragma: no cover
            raise KeyError(side)
        
        cores = [k.core for k in kernels]
        def core(x, y, **kw):
            wrapped = [wrapper(c, x, y, **kw) for c in cores]
            transformed = op(*wrapped)
            return transformed(arg(x, y))
        
        return __class__(core)

        # TODO instead of wrapping the cores just by closing over an argument,
        # define a `PartialKernel` object that has linop defined, with one arg
        # fixed to None, and __call__ that closes over an argument. This way I
        # could apply ops directly with deflintransf.

    # TODO move in the binary op methods the logic that decides wether to
    # return NotImplemented, while the algops will raise an exception if the
    # argument is not to their liking

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
        core = self.core
        return self._clone(
            __class__,
            core=lambda x, y, **kw: core(y, x, **kw),
        )

        # TODO make _swap a transf inherited by CrossIsotropicKernel => messes
        # up with Kernel subclasses. I want to make Kernel subclasses identity
        # on swap, so it can't be an inherited transf, because Kernel appears
        # as the second base.

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
        core = _jaxext.batchufunc(self.core, maxnbytes=maxnbytes)
        return self._clone(core=core)
    
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
        cls.__slots__ = ()

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
    def _alltransf(cls):
        """ list all accessible transfs as dict name -> (tcls, transf) """
        transfs = {}
        for tcls in cls._transfmro():
            for name, transf in tcls._transf.items():
                transfs.setdefault(name, (tcls, transf))
        return transfs

    @classmethod
    def _gettransf(cls, transfname, transfmro=None):
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
        if transfmro is None:
            transfmro = cls._transfmro()
        for c in transfmro:
            try:
                return c, c._transf[transfname]
            except KeyError:
                pass
        raise KeyError(transfname)

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
    def inherit_all_algops(cls, intermediates=False):
        """

        Inherit all algebraic operations from superclasses.

        This makes sense if the class represents a subalgebra, i.e., it
        should be preserved by addition and multiplication.

        Parameters
        ----------
        intermediates : bool, default False
            If True, make all superclasses up to the one definining the
            transformation inherit it too.

        Raises
        ------
        KeyError :
            An algebraic operation is already registered for one of the target
            classes.

        See also
        --------
        transf

        """
        mro = cls._transfmro()
        next(mro)
        for name, (_, transf) in next(mro)._alltransf().items():
            if transf.kind is cls._algopmarker:
                cls.inherit_transf(name, intermediates=intermediates)

    @classmethod
    def list_transf(cls, superclasses=True):
        """
        List all the available transformations.

        Parameters
        ----------
        superclasses : bool, default True
            Include transformations defined in superclasses.

        Returns
        -------
        transfs: dict of Transf
            The dictionary keys are the transformation names, the values are
            named tuples ``(tcls, kind, impl, doc)`` where ``tcls`` is the class
            defining the transformation, ``kind`` is the kind of transformation,
            ``impl`` is the implementation with signature ``impl(tcls, self,
            *args, **kw)``, ``doc`` is the docstring.
        """
        if superclasses:
            source = cls._alltransf().items
        else:
            def source():
                for name, transf in cls._transf.items():
                    yield name, (cls, transf)
        return {
            name: cls.Transf(tcls, transf.kind, transf.func, transf.doc)
            for name, (tcls, transf) in source()
        }

    Transf = collections.namedtuple('Transf', ['tcls', 'kind', 'func', 'doc'])

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
        linop, algop, transf_help, has_transf, list_transf, super_transf, register_transf, register_linop, register_corelinop, register_xtransf, register_algop, register_ufuncalgop

        """
        tcls, transf = self._gettransf(transfname)
        return transf.func(tcls, self, *args, **kw)

    @classmethod
    def super_transf(cls, transfname, self, *args, **kw):
        """
        
        Transform the kernel using a superclass transformation.

        This is equivalent to `transf` but is invoked on a class and the
        object is passed explicitly. The definition of the transformation is
        searched starting from the next class in the MRO of `self`.

        Parameters
        ----------
        transfname, *args, **kw :
            See `transf`.
        self : CrossKernel
            The object to transform.

        Returns
        -------
        newkernel : object
            The output of the transformation.

        """
        mro = list(self._transfmro())
        idx = mro.index(cls)
        tcls, transf = self._gettransf(transfname, mro[idx + 1:])
        return transf.func(tcls, self, *args, **kw)

    def linop(self, transfname, *args, **kw):
        r"""

        Transform kernels to represent the application of a linear operator.

        .. math::
            \text{kernel}_1(x, y) &= \mathrm{Cov}[f_1(x), g_1(y)], \\
            \text{kernel}_2(x, y) &= \mathrm{Cov}[f_2(x), g_2(y)], \\
            &\ldots \\
            \text{newkernel}(x, y) &=
                \mathrm{Cov}[T_f(f_1, f_2, \ldots)(x), T_g(g_1, g_2, \ldots)(y)]

        Parameters
        ----------
        transfname : hashable
            The name of the transformation.
        *args :
            A sequence of `CrossKernel` instances, representing the operands,
            followed by one or two non-kernel arguments, indicating how to act
            on each side of the kernels. If both arguments represent the
            identity, this is a no-op. If there is only one argument, it is
            intended that the two arguments are equal. `None` always represents
            the identity.

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
        The linear operator is defined on the function the kernel represents the
        distribution of, not in full generality on the kernel itself. When
        multiple kernels are involved, their distributions may be considered
        independent or not, depending on the specific operation.

        If the result is a subclass of the class defining the transformation,
        the result is casted to the latter. Then, if the result and all the
        operands are instances of `Kernel`, but the two operator arguments
        differ, the result is casted to its first non-`Kernel` superclass.
        
        """
        tcls, transf = self._gettransf(transfname)
        if transf.kind is not self._linopmarker:
            raise ValueError(f'the transformation {transfname!r} was not '
                f'defined with register_linop and so can not be invoked '
                f'by linop')
        return transf.func(tcls, self, *args)

    def algop(self, transfname, *operands, **kw):
        r"""

        Return a nonnegative algebraic transformation of the input kernels.

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

        For class determination, scalars in the input count as `Constant`
        if nonnegative or traced by jax, else `CrossConstant`.

        """
        tcls, transf = self._gettransf(transfname)
        if transf.kind is not self._algopmarker:
            raise ValueError(f'the transformation {transfname!r} was not '
                f'defined with register_algop and so can not be invoked '
                f'by algop')
        return transf.func(tcls, self, *operands, **kw)

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

        # TODO forbid to override with a different kind?

    @classmethod
    def register_linop(cls, op, transfname=None, doc=None, argparser=None):
        """
        
        Register a transformation for use with `linop`.

        Parameters
        ----------
        op : callable
            A function ``op(tcls, self, arg1, arg2, *operands) -> CrossKernel``
            that returns the new kernel, where ``arg1`` and ``arg2`` represent
            the operators acting on each side of the kernels, and ``operands``
            are the other kernels beyond ``self``.
        transfname, doc : optional
            See `register_transf`.
        argparser : callable, optional
            A function applied to ``arg1`` and ``arg2``. Not called if the
            argument is `None`. It should map the identity to `None`.

        Returns
        -------
        op : callable
            The `op` argument as is.

        Notes
        -----
        The function `op` is called only if ``arg1`` or ``arg2`` is not `None`
        after potential conversion with `argparser`.

        See also
        --------
        transf

        """
        
        if transfname is None:
            transfname = op.__name__ # for result type error message
        
        @functools.wraps(op)
        def func(tcls, self, *allargs):

            # split the arguments in kernels and non-kernels
            for pos, arg in enumerate(allargs):
                if not isinstance(arg, __class__):
                    break
            else:
                pos = len(allargs)
            operands = allargs[:pos]
            args = allargs[pos:]
            
            # check the arguments from the first non-kernel onwards are 1 or 2
            if len(args) not in (1, 2):
                raise ValueError(f'incorrect number of non-kernel tail '
                    f'arguments {len(args)}, expected 1 or 2')

            # wrap argument parser to enforce preserving None
            if argparser:
                conv = lambda x: None if x is None else argparser(x)
            else:
                conv = lambda x: x
            
            # determine if the two arguments count as "identical" or not
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
                    # they must be not identical both before and after to handle
                    # these cases:
                    #  - if the user passes identical arguments, but argparser
                    #    makes copies, it must still count as identical
                    #  - if the user passes arguments which are not identical,
                    #    but argparser sends them to the same object, then they
                    #    surely represent the same transf

            # handle no-op case
            if arg1 is None and arg2 is None:
                return self

            # invoke implementation
            result = op(tcls, self, arg1, arg2, *operands)

            # check result is a kernel
            if not isinstance(result, __class__):
                raise TypeError(f'linop {transfname!r} returned '
                    f'object of type {result.__class__.__name__}, expected '
                    f'subclass of {__class__.__name__}')
            
            # modify class of the result
            rcls = result.__class__
            if issubclass(rcls, tcls):
                rcls = tcls
            all_operands_kernel = all(isinstance(o, Kernel) for o in operands)
            if isinstance(self, Kernel) and all_operands_kernel and different:
                rcls = next(rcls._crossmro())
            if rcls is not result.__class__:
                result = result._clone(rcls)
            
            return result

        cls.register_transf(func, transfname, doc, cls._linopmarker)
        return op

    class _LinOpMarker(str): pass
    _linopmarker = _LinOpMarker('linop')

    @classmethod
    def register_corelinop(cls, corefunc, transfname=None, doc=None, argparser=None):
        """

        Register a linear operator with a function that acts only on the core.

        Parameters
        ----------
        corefunc : callable
            A function ``corefunc(core, arg1, arg2, *cores) -> newcore``, where
            ``core`` is the function that implements the kernel passed at
            initialization, and ``cores`` for other operands.
        transfname, doc, argparser :
            See `register_linop`.

        Returns
        -------
        corefunc : callable
            The `corefunc` argument as is.

        See also
        --------
        transf

        """
        @functools.wraps(corefunc)
        def op(_, self, arg1, arg2, *operands):
            cores = (o.core for o in operands)
            core = corefunc(self.core, arg1, arg2, *cores)
            return self._clone(core=core)
        cls.register_linop(op, transfname, doc, argparser)
        return corefunc

    @classmethod
    def register_xtransf(cls, xfunc, transfname=None, doc=None):
        """

        Register a linear operator that acts only on the input.

        Parameters
        ----------
        xfunc : callable
            A function ``xfunc(arg) -> (lambda x: newx)`` that takes in a
            `linop` argument and produces a function to transform the input. Not
            called if ``arg`` is `None`. To indicate the identity, return
            `None`.
        transfname, doc :
            See `register_linop`. `argparser` is not provided because its
            functionality can be included in `xfunc`.

        Returns
        -------
        xfunc : callable
            The `xfunc` argument as is.

        See also
        --------
        transf

        """

        @functools.wraps(xfunc)
        def corefunc(core, xfun, yfun):
            if not xfun:
                return lambda x, y, **kw: core(x, yfun(y), **kw)
            elif not yfun:
                return lambda x, y, **kw: core(xfun(x), y, **kw)
            else:
                return lambda x, y, **kw: core(xfun(x), yfun(y), **kw)
        
        cls.register_corelinop(corefunc, transfname, doc, xfunc)
        return xfunc

    @classmethod
    def register_algop(cls, op, transfname=None, doc=None):
        """

        Register a transformation for use with `algop`.

        Parameters
        ----------
        op : callable
            A function ``op(tcls, *kernels, **kw) -> CrossKernel |
            NotImplemented`` that returns the new kernel. ``kernels`` may be
            scalars but for the first argument.
        transfname, doc :
            See `register_transf`.

        Returns
        -------
        op : callable
            The `op` argument as is.

        See also
        --------
        transf

        """

        if transfname is None:
            transfname = op.__name__ # for error message
        
        @functools.wraps(op)
        def func(tcls, *operands, **kw):
            result = op(tcls, *operands, **kw)
            
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
                        yield Constant
                    elif _util.is_numerical_scalar(o):
                        yield CrossConstant
                    else:
                        raise TypeError(f'operands to algop {transfname!r} '
                            f'must be CrossKernel or numbers, found {o!r}')
                        # this type check comes after letting the implementation
                        # return NotImplemented, to support overloading
                yield result.__class__
            
            lcs = least_common_superclass(*classes())
            return result._clone(lcs)
    
        cls.register_transf(func, transfname, doc, cls._algopmarker)
        return op

        # TODO delete initkw (also in linop) if there's more than one kernel
        # operand or if the class changed?

        # TODO consider adding an option domains=callable, returns list of
        # domains from the operands, or list of tuples right away, and the
        # impl checks at runtime (if not traced) that the output values are in
        # the domains, with an informative error message

        # TODO consider escalating to ancestor definitions of the transf
        # until an impl does not return NotImplemented, and then raise an
        # an error instead of letting the NotImplemented escape. This would be
        # useful to partial behavior ovverride in subclasses, e.g, handle
        # multiplication by scalar to rewire transformations.

    class _AlgOpMarker(str): pass
    _algopmarker = _AlgOpMarker('algop')

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
        ufunc : callable
            The `ufunc` argument as is.

        See also
        --------
        transf

        """
        @functools.wraps(ufunc)
        def op(_, self, *operands, **kw):
            cores = tuple(
                o.core if isinstance(o, __class__)
                else lambda x, y: o
                for o in (self, *operands)
            )
            def core(x, y, **kw):
                values = (core(x, y, **kw) for core in cores)
                return ufunc(*values, **kw)
            return self._clone(core=core)
        cls.register_algop(op, transfname, doc)
        return ufunc

    @classmethod
    def make_linop_family(cls, transfname, bothker, leftker, rightker=None, *,
        doc=None, argparser=None, argnames=None, translkw=None):
        """
        
        Form a family of kernels classes related by linear operators.

        The class this method is called on is the seed class. A new
        trasformation is registered to obtain other classes from the seed class
        by applying a newly defined linop transformation.

        Parameters
        ----------
        transfname : str
            The name of the new transformation.
        bothker, leftker, rightker : CrossKernel
            The kernel classes to be obtained by applying the operator to a seed
            class object respectively on both sides, only left, or only right.
            All classes are assumed to require no positional arguments at
            construction, and recognize the same set of keyword arguments. If
            `rightker` is not specified, it is defined by subclassing `leftker`
            and transposing the kernel on object construction.
        doc, argparser : callable, optional
            See `register_linop`.
        argnames : pair of str, optional
            If specified, `leftker` is passed an additional keyword argument
            with name ``argnames[0]`` that specifies the argument of the
            operator, `rightker` is passed ``argnames[1]``, and `bothker`
            similarly is passed both.

            If `leftker` is used to implement `rightker`, it must accept both
            arguments, and may use them to determine if it is being invoked as
            `leftker` or `rightker`.
        translkw : callable, optional
            A function with signature ``translkw(dynkw, <argnames>, **initkw) ->
            dict`` that determines the constructor arguments for a new object
            starting from the ones of the object to be transformed. By default,
            ``initkw`` is passed over, and an error is raised if ``dynkw`` is
            not empty.

        Examples
        --------

        >>> @lgp.kernel
        ... def A(x, y, *, gatto):
        ...     ''' The reknown A kernel of order gatto '''
        ...     return ...
        ...
        >>> @lgp.kernel
        ... def T(n, m, *, gatto, op):
        ...     ''' The kernel of the Topo series of an A process of order
        ...         gatto '''
        ...     op1, op2 = op
        ...     return ...
        ...
        >>> @lgp.crosskernel
        ... def CrossTA(n, x, *, gatto, op):
        ...     ''' The cross covariance between the Topo series of an A
        ...         process of order gatto and the process itself '''
        ...     return ...
        ...
        >>> A.make_linop_family('topo', T, CrossTA)
        >>> a = A(gatto=7)
        >>> ta = A.linop('topo', True, None)
        >>> at = A.linop('topo', None, True)
        >>> t = A.linop('topo', True)

        See also
        --------
        transf

        """
        
        if rightker is None:

            # invent a name for rightker
            rightname = f'Cross{cls.__name__}{bothker.__name__}'

            # define how to set up rightker
            def exec_body(ns):
                
                if leftker.__doc__:
                    header = 'Automatically generated transposed version of:\n\n'
                    ns['__doc__'] = _utils.append_to_docstring(leftker.__doc__, header, front=True)
                
                def __new__(cls, *args, **kw):
                    self = super(rightker, cls).__new__(cls, *args, **kw)
                    
                    if self.__class__ is cls:
                        self = self._swap()
                        if not isinstance(self, leftker):
                            raise TypeError(f'newly created instance of '
                                f'automatically defined {rightker.__name__} is not an '
                                f'instance of {leftker.__name__} after '
                                f'transposition. Either define transposition '
                                f'for {leftker.__name__}, or define '
                                f'{rightker.__name__} manually')
                        return self._clone(cls)

                    else:
                        return self._swap()

                ns['__new__'] = __new__

            # create rightker, evil twin of leftker separated at birth
            rightker = types.new_class(rightname, (leftker,), exec_body=exec_body)

        # check which classes are symmetric
        classes = cls, bothker, leftker, rightker
        sym = tuple(issubclass(c, Kernel) for c in classes)
        exp = True, True, False, False
        if sym != exp:
            desc = lambda t: 'Kernel' if t else 'non-Kernel'
            warnings.warn(f'Expected classes pattern {", ".join(map(desc, exp))}, '
                f'found {", ".join(map(desc, sym))}')

        # set translkw if not specified
        if translkw is None:
            def translkw(*, dynkw, **initkw):
                if dynkw:
                    raise ValueError('found non-empty `dynkw`, the default '
                        'implementation of `translkw` does not support it')
                return initkw

        # function to produce the arguments to the transformed objects
        def makekw(self, arg1, arg2):
            kw = dict(dynkw=self.dynkw, **self.initkw)
            if argnames is not None:
                if arg1 is not None:
                    kw = dict(**kw, **{argnames[0]: arg1})
                if arg2 is not None:
                    kw = dict(**kw, **{argnames[1]: arg2})
            return translkw(**kw)

        # register linop mapping cls to either leftker, rightker or bothker
        regkw = dict(transfname=transfname, doc=doc, argparser=argparser)
        @functools.partial(cls.register_linop, **regkw)
        def op_seed_to_siblings(_, self, arg1, arg2):
            kw = makekw(self, arg1, arg2)
            if arg2 is None:
                return leftker(**kw)
            elif arg1 is None:
                return rightker(**kw)
            else:
                return bothker(**kw)

        # register linop mapping leftker to bothker
        @functools.partial(leftker.register_linop, **regkw)
        def op_left_to_both(_, self, arg1, arg2):
            if arg1 is None:
                return bothker(**makekw(self, arg1, arg2))
            else:
                raise ValueError(f'cannot further transform '
                    f'`{leftker.__name__}` on left side with linop '
                    f'{transfname!r}')

        # register linop mapping rightker to bothker
        @functools.partial(rightker.register_linop, **regkw)
        def op_right_to_both(_, self, arg1, arg2):
            if arg2 is None:
                return bothker(**makekw(self, arg1, arg2))
            else:
                raise ValueError(f'cannot further transform '
                    f'`{rightker.__name__}` on right side with linop '
                    f'{transfname!r}')

class AffineSpan(CrossKernel, abc.ABC):
    """

    Kernel that tracks affine transformations.

    An `AffineSpan` instance accumulates the overall affine transformation
    applied to its inputs and output.

    `AffineSpan` and it subclasses are preserved by the transformations
    'scale', 'loc', 'add' (with scalar) and 'mul' (with scalar).

    `AffineSpan` can not be instantiated directly or used as standalone
    superclass. It must be the first base before concrete superclasses.

    """
    
    _affine_dynkw = dict(lloc=0, rloc=0, lscale=1, rscale=1, offset=0, ampl=1)

    def __new__(cls, *args, dynkw={}, **kw):
        if cls is __class__:
            raise TypeError(f'cannot instantiate {__class__.__name__} directly')
        new_dynkw = dict(cls._affine_dynkw)
        new_dynkw.update(dynkw)
        return super().__new__(cls, *args, dynkw=new_dynkw, **kw)

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        for name in __class__._transf:
            cls.inherit_transf(name)

        # TODO I would like to inherit conditional on there not being another
        # definition. However, since registrations are done after class
        # creation, this method is invoked too early to do that.
        #
        # Is it possible to reimplement the transformation system to have the
        # transformations defined in the class body?
        #
        # Option 1: use a decorator that marks the transf methods and makes
        # them staticmethods. Then an __init_subclass__ above CrossKernel goes
        # through the methods and registers the marked ones.
        # 
        # Option 2: try to use descriptors (classes like property). Is it
        # possible to reimplement from scratch something like classmethod? If
        # so, I could make a transfmethod that bounds two arguments: tcls
        # for the class that defines it, and self for the object that invokes
        # it. Then inheriting would be just copying the thing over. If I wanted
        # to keep the invocation through method interface, I could define the
        # methods with underscores and register them somewhere.
        #
        # However, since I define kernels with decorators, I can't actually
        # count on the transformations being defined into them at class
        # creation. But that would be quite a corner case.
        #
        # To make subclassing convenient without decorators, allow to define a
        # core(x, y) method, which __init_subclass__ converts to staticmethod,
        # and have __new__ use it to set _core (I can't use the same name with
        # slots).
        #
        # to use without post-class registering make_linopfamily, make it a
        # decorator with attributed subdecorators for other members:
        #
        # @linop_family('T')
        # @kernel
        # def A(...)
        # @A.left('arg')
        # @crosskernel
        # def CrossTA(...)
        # @A.right # optional, generated from left if not specified
        # ...
        # @A.both('arg1', 'arg2')
        # @kernel
        # def T(...)
        #
        # Can also be stacked on top of left/right/both to chain families. Works
        # both with decorators and explicit class definitions.

    def _clone(self, *args, **kw):
        newself = super()._clone(*args, **kw)
        if isinstance(newself, __class__):
            for name in self._affine_dynkw:
                newself._dynkw[name] = self._dynkw[name]
        return newself

    @classmethod
    def __subclasshook__(cls, sub):
        if cls is __class__:
            return NotImplemented
                # to avoid algops promoting to unqualified AffineSpan
        if issubclass(cls, Kernel):
            if issubclass(sub, Constant):
                return True
            else:
                return NotImplemented
        elif issubclass(sub, CrossConstant):
            return True
        else:
            return NotImplemented

    # TODO I could do separately AffineLeft, AffineRight, AffineOut, and make
    # this a subclass of those three. AffineOut would also allow keeping the
    # class when adding two objects without keyword arguments in initkw and
    # dynkw beyond those managed by Affine. => Make only AffineSide and have it
    # switch side on _swap.

    # TODO when I reimplement transformations as methods, make AffineSpan not
    # a subclass of CrossKernel. Right now I have to to avoid routing around
    # the assumption that all classes in the MRO implement the transformation
    # management logic.

class PreservedBySwap(CrossKernel):

    def __new__(cls, *args, **kw):
        if cls is __class__:
            raise TypeError(f'cannot instantiate {__class__.__name__} directly')
        return super().__new__(cls, *args, **kw)

    def _swap(self):
        return super()._swap()._clone(self.__class__)

    # TODO when I implement transformations with methods, make this not an
    # instance of CrossKernel.
