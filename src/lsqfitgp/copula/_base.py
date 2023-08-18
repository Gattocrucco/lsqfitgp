# lsqfitgp/copula/_base.py
#
# Copyright (c) 2023, Giacomo Petrillo
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

# TODO document the conditional copula functionality.

import abc
import functools
import collections
import numbers

import gvar
import numpy
import jax
from jax import numpy as jnp

from .. import _patch_gvar
from .. import _array

class Distr(metaclass=abc.ABCMeta):
    r"""
    Abstract base class to represent distributions.

    A `Distr` object represents a probability distribution, and provides a
    transformation function from a (multivariate) Normal variable to the target
    random variable, in particular for use with `gvar.BufferDict`.

    Parameters
    ----------
    *params : tuple of scalar, array or Distr
        The parameters of the distribution. If the parameters have leading axes
        other than those required, the distribution is independently repeated
        over those axes. If a parameter is an instance of `Distr` itself, it
        is a random parameter and its distribution is accounted for.
    name : str, optional
        If specified, the distribution is defined for usage with
        `gvar.BufferDict` using `gvar.BufferDict.add_distribution`, and the
        constructor returns an array of gvars with the appropriate shape for
        convenience. See `add_distribution`.
    shape : int or tuple of int
        The shape of the array of variables to be represented. If the variable
        is multivariate, this shape adds as leading axes in the array. Default
        scalar. This shape broadcasts with those of the parameters.

    Returns
    -------
    If `name` is None (default):

    distr : Distr
        An object representing the distribution.

    Else:

    gvars : array of gvars
        An array of primary gvars that can be set as value in a
        `gvar.BufferDict` under a key that uses the just defined name.

    Attributes
    ----------
    partial_invfcn : callable
        Transformation function from a (multivariate) Normal variable to the
        target random variable. Differs from the class method `invfcn` in that
        the core input shape is flattened or scalar, and the parameters are
        baked in, including transformation of the random parameters.
    params : tuple
        The parameters as passed to the constructor.
    in_shape : tuple of int
        The core shape of the input array to `partial_invfcn`.
    shape : tuple of int
        The core shape of the output array of `partial_invfcn`.
    dtype : dtype
        The dtype of the output array of `partial_invfcn`.
    distrshape : tuple of int
        The sub-core shape of the output array of `partial_invfcn` that
        represents the atomic shape of the distribution.

    Methods
    -------
    add_distribution :
        Define the distribution for usage with `gvar.BufferDict`.
    gvars :
        Return an array of gvars with the appropriate shape for usage with
        `gvar.BufferDict`.
    invfcn :
        Transformation function from a (multivariate) Normal variable to the
        target random variable.
    input_shape :
        Return the input core shape expected by `invfcn`.
    output_shape :
        Return the output core shape of `invfcn`.

    Examples
    --------

    Use directly with `gvar.BufferDict` by setting `name`:

    >>> copula = gvar.BufferDict({
    ...     'A(x)': lgp.copula.beta(1, 1, name='A'),
    ...     'B(y)': lgp.copula.beta(3, 5, name='B'),
    ... })
    >>> copula['x']
    0.50(40)
    >>> copula['y']
    0.36(18)

    Corresponding "unrolled" usage:

    >>> A = lgp.copula.beta(1, 1)
    >>> B = lgp.copula.beta(3, 5)
    >>> A.add_distribution('A')
    >>> B.add_distribution('B')
    >>> copula = gvar.BufferDict({
    ...     'A(x)': A.gvars(),
    ...     'B(y)': B.gvars(),
    ... })

    Notice that, although the name used for `add_distribution` must be globally
    unique, for convenience it is permitted to redefine the same distribution
    family with the same parameters, even from another `Distr` instance.

    Define a distribution with a random parameter:

    >>> X = lgp.copula.halfnorm(lgp.copula.invgamma(1, 1))

    Now `X` represents the model

    .. math::
        \sigma &\sim \mathrm{InvGamma}(1, 1), \\
        X \mid \sigma &\sim \mathrm{HalfNorm}(\sigma).

    Repeated usage of `Distr` instances for random parameters will *not* share
    those parameters in the distributions. The following code:

    >>> sigma = lgp.copula.invgamma(1, 1)
    >>> X = lgp.copula.halfnorm(sigma)
    >>> Y = lgp.copula.halfcauchy(sigma)

    Corresponds to the model

    .. math::
        \sigma_X, \sigma_Y &\sim \mathrm{InvGamma}(1, 1), \\
        X \mid \sigma_X &\sim \mathrm{HalfNorm}(\sigma_X), \\
        Y \mid \sigma_Y &\sim \mathrm{HalfCauchy}(\sigma_Y),

    with independent, separate parameters :math:`\sigma_X` and :math:`\sigma_Y`.
    To compose arbitrary distributions, use manually `invfcn`:

    >>> @functools.partial(lgp.gvar_gufunc, signature='(3)->(3)')
    >>> def model_invfcn(normal_params):
    ...     sigma2 = lgp.copula.invgamma.invfcn(normal_params[0], 1, 1)
    ...     sigma = jnp.sqrt(sigma2)
    ...     X = lgp.copula.halfnorm.invfcn(normal_params[1], sigma)
    ...     Y = lgp.copula.halfcauchy.invfcn(normal_params[2], sigma)
    ...     return jnp.stack([sigma, X, Y], axis=-1)
    
    Now the function `model_invfcn` represents the model

    .. math::
        \sigma^2 &\sim \mathrm{InvGamma}(1, 1), \\
        X \mid \sigma &\sim \mathrm{HalfNorm}(\sigma), \\
        Y \mid \sigma &\sim \mathrm{HalfCauchy}(\sigma).

    The `gvar_gufunc` decorator makes `model_invfcn` accept gvars as input.

    See also
    --------
    gvar.BufferDict.uniform

    Notes
    -----
    Concrete subclasses must define `invfcn`, and possibly override the default
    implementations of `input_shape` and `output_shape`, which return an empty
    tuple. To easily make `invfcn` a generalized ufunc, consider using
    `jax.numpy.vectorize`.

    """
    
    @classmethod
    @abc.abstractmethod
    def invfcn(cls, x, *params):
        r"""

        Normal to desired distribution transformation.

        Maps a (multivariate) Normal variable to a variable with the desired
        marginal distribution. In symbols: :math:`y = F^{-1}(\Phi(x))`. This
        function is a generalized ufunc, jax traceable, and differentiable one
        time.

        Parameters
        ----------
        x : ``(*head_shape, *input_shape)`` array or scalar
            The input Normal variable. If multivariate, the variable shape must
            be in the core shape ``in_shape``. The calculation is broadcasted
            over other leading axes. Use `input_shape` to determine the required
            variable shape.
        params : scalar or arrays
            The parameters of the distribution. They are broadcasted with `x`.

        Returns
        -------
        y : ``(*head_shape, *output_shape)`` array or scalar
            The output variable with the desired marginal distribution. Use
            `output_shape` to determine ``output_shape`` before calling
            `invfcn`.

        """
        pass

    @classmethod
    def input_shape(cls, *params):
        """
        Return the input core shape expected by `invfcn`.

        May depend on the type and shape of the parameters, but not on their
        values.

        Parameters
        ----------
        params :
            The same arguments that would be passed to `invfcn` or to the
            class constructor.

        Returns
        -------
        shape : tuple of int
            The required core shape of the input array `x` to `invfcn`.
        """
        return ()

    @classmethod
    def output_shape(cls, *params):
        """
        Return the output core shape of `invfcn`.

        May depend on the type and shape of the parameters, but not on their
        values.

        Parameters
        ----------
        params :
            The same arguments that would be passed to `invfcn` or to the
            class constructor.

        Returns
        -------
        shape : tuple of int
            The core shape of the output array `y` of `invfcn`.
        """
        return ()

    @classmethod
    def _eval_shapes(cls, params, shape):

        # convert parameters to array dummies
        dummies = []
        for p in params:
            if not hasattr(p, 'shape') or not hasattr(p, 'dtype'):
                p = jnp.asarray(p)
            dummies.append(jax.ShapeDtypeStruct(p.shape, p.dtype))
        
        # determine signature of cls.invfcn
        in_shape_0 = cls.input_shape(*dummies)
        out_shape_0 = cls.output_shape(*dummies)
        
        # make a dummy for the input to cls.invfcn
        x_dtype = jax.dtypes.canonicalize_dtype(jnp.float64)
        x_dummy = jax.ShapeDtypeStruct(shape + in_shape_0, x_dtype)
        
        # use jax to get shape with parameter broadcasting
        out_dummy = jax.eval_shape(cls.invfcn, x_dummy, *dummies)
        broadcast_ndim = len(out_dummy.shape) - len(out_shape_0)
        assert broadcast_ndim >= 0
        assert out_dummy.shape[broadcast_ndim:] == out_shape_0

        # determine shapes to use with cls.invfcn
        in_shape_1 = out_dummy.shape[:broadcast_ndim] + in_shape_0
        out_shape_1 = out_dummy.shape
        out_dtype = out_dummy.dtype

        return in_shape_1, out_shape_1, out_shape_0, out_dtype

    @classmethod
    def _compute_in_shape_2(cls, params, in_shape_1):
        in_size = 0
        for p in params:
            if isinstance(p, __class__):
                in_size += numpy.prod(p.in_shape, dtype=int)
        in_size += numpy.prod(in_shape_1, dtype=int)
        if in_size == 1:
            return ()
        else:
            return in_size,

    @classmethod
    def _make_partial_invfcn(cls, params, in_shape_2, in_shape_1, out_shape_1, out_dtype):
        
        # determine signature
        shapestr = lambda shape: ', '.join(map(str, shape))
        signature = f'({shapestr(in_shape_2)})->({shapestr(out_shape_1)})'

        # wrap to support input array of gvars and unpack input into parameters'
        # distributions and our
        @functools.partial(_patch_gvar.gvar_gufunc, signature=signature)
        @functools.wraps(cls.invfcn)
        def partial_invfcn(x):

            # check input
            x = jnp.asarray(x)
            assert x.ndim >= len(in_shape_2)
            assert x.shape[x.ndim - len(in_shape_2):] == in_shape_2
            if not in_shape_2:
                x = x[..., None]
            in_size = numpy.prod(in_shape_2, dtype=int)

            # loop over parameters
            i = 0
            concrete_params = []
            for p in params:
                
                # parameter is a copula, apply it
                if isinstance(p, __class__):
                    p_in_size = numpy.prod(p.in_shape, dtype=int)
                    assert i + p_in_size <= in_size
                    p_x = x[..., i:i + p_in_size]
                    p_x = p_x.reshape(p_x.shape[:-1] + p.in_shape)
                    p = p.partial_invfcn(p_x)
                    i += p_in_size
                
                concrete_params.append(p)

            # evaluate inverse transformation
            last = x[..., i:]
            last = last.reshape(last.shape[:-1] + in_shape_1)
            out = cls.invfcn(last, *concrete_params)
            assert out.shape == x.shape[:-1] + out_shape_1
            assert out.dtype == out_dtype
            return out

        partial_invfcn.signature = signature
        partial_invfcn.__doc__ = f"""
            
        Map i.i.d. Normal variables to a {cls.__name__} variable.

        This function is a generalized ufunc with signature {signature}.
        It is jax traceable and differentiable one time. It supports arrays
        of gvars as input.

        Parameters
        ----------
        x : array
            An array of values representing draws of i.i.d. Normal
            variates.

        Returns
        -------
        y : array
            An array of values representing draws of the desired
            distribution.

        """

        return partial_invfcn

    def __new__(cls, *params, name=None, shape=()):

        if isinstance(shape, numbers.Integral):
            shape = (shape,)
        else:
            shape = tuple(shape)

        in_shape_1, out_shape_1, out_shape_0, out_dtype = cls._eval_shapes(params, shape)
        in_shape_2 = cls._compute_in_shape_2(params, in_shape_1)
        partial_invfcn = cls._make_partial_invfcn(params, in_shape_2, in_shape_1, out_shape_1, out_dtype)

        self = super().__new__(cls)
        self.partial_invfcn = partial_invfcn
        self.params = params
        self.in_shape = in_shape_2
        self.distrshape = out_shape_0
        self.shape = out_shape_1
        self.dtype = out_dtype

        if name is None:
            return self
        else:
            self.add_distribution(name)
            return self.gvars()
    
    class _Descr(collections.namedtuple('Distr', 'family shape params')):
        """ static representation of a Distr object """

        def _repr(self, conv):
            args = list(map(conv, self.params))
            if len(self.shape) == 1:
                args += [f'shape={self.shape[0]}']
            elif self.shape:
                args += [f'shape={self.shape}']
            arglist = ', '.join(args)
            return f'{self.family.__name__}({arglist})'

        __repr__ = lambda self: self._repr(repr)
        __str__ = lambda self: self._repr(str)

    def _descr(self, param_converter):

        params = []
        for p in self.params:
            if isinstance(p, __class__):
                p = p._descr(param_converter)
            else:
                p = param_converter(p)
            params.append(p)

        return self._Descr(self.__class__, self.shape, tuple(params))

    @functools.cached_property
    def _staticdescr(self):
        """ static description of self, can be compared """
        return self._descr(lambda p: numpy.asarray(p).tolist())

    @functools.cached_property
    def _repr(self):
        """ description of self for representation """
        def param_converter(p):
            if hasattr(p, 'shape'):
                return f'array{p.shape}'
            else:
                return p
        return self._descr(param_converter)

    def __repr__(self):
        return str(self._repr)

    @classmethod
    def _earmark(cls, obj):
        obj._Distr_subclass_ = cls

    @classmethod
    def _is_earmarked(cls, obj):
        return getattr(obj, '_Distr_subclass_', None) is cls

    def __init_subclass__(cls, **_):
        cls._named = {}
    
    def add_distribution(self, name):
        """

        Define the distribution for usage with `gvar.BufferDict`.

        Parameters
        ----------
        name : str
            The name to use for the distribution. It must be globally unique,
            and it should not contain parentheses. To redefine a distribution
            with the same name, use `gvar.BufferDict.del_distribution` first.
            However, it is allowed to reuse the name if the distribution family
            and parameters are identical to those used for the existing
            definition.

        See also
        --------
        gvar.BufferDict.add_distribution, gvar.BufferDict.del_distribution

        """

        if gvar.BufferDict.has_distribution(name):
            invfcn = gvar.BufferDict.invfcn[name]
            if not self._is_earmarked(invfcn):
                raise ValueError(f'distribution {name} already defined')
            existing = self._named[name]
            if existing != self._staticdescr:
                raise ValueError('Attempt to overwrite existing'
                    f' {self.__class__.__name__} distribution with name {name}')
                # cls._named is not updated by
                # gvar.BufferDict.del_distribution, but it is not a problem
        
        else:
            self._earmark(self.partial_invfcn)
            gvar.BufferDict.add_distribution(name, self.partial_invfcn)
            self._named[name] = self._staticdescr
    
    def gvars(self):
        """

        Return an array of gvars intended as value in a `gvar.BufferDict`.

        Returns
        -------
        gvars : array of gvars
            An array of i.i.d. standard Normal primary gvars with shape
            `in_shape`.

        """

        return gvar.gvar(numpy.zeros(self.in_shape), numpy.ones(self.in_shape))

# TODO
# - make partial_invfcn a method, requires an `excluded` arg in gvar_ufunc.
# - make Distr instances shareable by using a cache : dict[Distr, output]
#   in partial_invfcn (Distr should by default compare with `is`, check this)
#   and a cache : set[Distr] in _compute_in_shape_2
# - make Distr instances dispatching array-likes that perform the operations
#   by creating a new instance with a custom invfcn (this requires always
#   getting invfcn from self!) from a generic subclass that just applies the
#   operation to the output of the operand invfcns using jax.numpy.
# - make a function makedict({'a': norm(0, 1)}, prefix='ciao_') ->
#   BufferDict({'ciao_norm{0, 1}(a)': 0(1)}), the default prefix is '__copula_',
#   the name is the _staticdescr, convert parentheses in name to {}.
# - make a class Copula usable by attributes, example usage:
#       c = Copula()
#       c.sigma('invgamma', 1, 1)
#       c.x('norm', 0, c.sigma)
#   implementation: __getattr__ returns the attr if it exists, else a proxy
#   callable object that creates the Distr and stores it. Overwrite is thus
#   forbidden. Make Distr.__call__ emit an explicative error. Use the Copula
#   with c.bufferdict(prefix='__copula_'), c.invfcn(array) -> dict[name, array]
#   c.in_size.
# - Drop custom __new__ in Distr subclasses, generate the class ref string
#   from the signature of invfcn.
