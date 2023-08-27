# lsqfitgp/tests/kernels/test_kernel.py
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

""" Test the generic kernel machinery """

import sys
import operator

import numpy as np
from numpy.lib import recfunctions
import jax
from jax import numpy as jnp
from pytest import mark
import pytest

import lsqfitgp as lgp

from .. import util

@pytest.fixture
def constcore():
    return lambda x, y, **_: jnp.ones(jnp.broadcast_shapes(x.shape, y.shape))

def test_forcekron(constcore):
    # at initialization it's within maxdim
    kernel = lgp.Kernel(constcore, forcekron=True, maxdim=1)
    x = np.empty(1, 'f,f')
    with pytest.raises(ValueError, match='> maxdim='):
        kernel(x, x)

    # after initialization it's not, no error
    kernel = kernel.transf('forcekron')
    kernel(x, x)

    # promotes to Kernel since it changes the core
    class A(lgp.Kernel): pass
    a = A(constcore)
    assert a.transf('forcekron').__class__ is lgp.Kernel

    # not defined for CrossKernel
    with pytest.raises(KeyError, match='forcekron'):
        lgp.CrossKernel(constcore, forcekron=True)

def test_cross_derivable(constcore):
    """ check that derivable of CrossKernel is duplicated """
    kernel = lgp.Kernel(constcore, derivable=4, maxdim=2)
    assert kernel.derivable == 4
    kernel = kernel.linop('normalize', True, None)
    assert kernel.derivable == (4, 4)

@mark.parametrize('op', [operator.add, operator.mul])
@mark.parametrize('cls', [lgp.CrossKernel, lgp.Kernel])
def test_binary_kernel(op, cls, rng):
    f1 = lambda x, y: 1.2 * x + 8.9 * y
    f2 = lambda x, y: 3.4 * x + 5.6 * y
    kernel1 = cls(f1)
    kernel2 = cls(f2)
    kernel = op(kernel1, kernel2)
    x, y = rng.standard_normal((2, 5))
    result = kernel(x, y)
    expected = op(f1(x, y), f2(x, y))
    util.assert_equal(result, expected)

@mark.parametrize('op', [operator.add, operator.mul])
@mark.parametrize('cls', [lgp.CrossKernel, lgp.Kernel])
def test_binary_scalar(op, cls, rng):
    f1 = lambda x, y: 1.2 * x + 8.9 * y
    f2 = 3.4
    x, y = rng.standard_normal((2, 5))
    kernel1 = cls(f1)
    args = kernel1, f2
    for _ in range(2):
        kernel = op(*args)
        result = kernel(x, y)
        expected = op(f1(x, y), f2)
        util.assert_equal(result, expected)
        args = tuple(reversed(args))

@mark.parametrize('op', [operator.add, operator.mul])
@mark.parametrize('cls', [lgp.CrossKernel, lgp.Kernel])
def test_binary_undef(op, cls, constcore):
    kernel = cls(constcore)
    with pytest.raises(TypeError):
        op(kernel, 'gatto')

@mark.parametrize('cls', [lgp.CrossKernel, lgp.Kernel])
def test_pow(cls, rng):
    f = lambda x, y: 1.2 * x + 8.9 * y
    for exp in 3, np.int64(3), np.array(3), jnp.array(3):
        kernel = cls(f) ** exp
        x, y = rng.standard_normal((2, 5))
        result = kernel(x, y)
        expected = f(x, y) ** exp
        util.assert_equal(result, expected)

    for exp in 3., np.float64(3), np.array(3.), jnp.array(3.):
        with pytest.raises(TypeError):
            cls(f) ** exp

    with pytest.raises(TypeError):
        cls(f) ** -1

    @jax.jit
    def traced(exp, x, y):
        return (cls(f) ** exp)(x, y)
    traced(jnp.uint64(3), x, y)
    with pytest.raises(TypeError):
        traced(3., x, y)
    with pytest.raises(TypeError):
        traced(3, x, y)

@mark.parametrize('op', [operator.add, operator.mul])
@mark.parametrize('cls', [lgp.StationaryKernel, lgp.IsotropicKernel])
def test_binary_kernel_class(op, cls, constcore):
    
    assert op(cls(constcore), cls(constcore)).__class__ is cls
    assert op(cls(constcore), lgp.Kernel(constcore)).__class__ is lgp.Kernel
    assert op(lgp.Kernel(constcore), cls(constcore)).__class__ is lgp.Kernel
    
    sup = cls.mro()[1]
    assert sup.__name__.startswith('Cross')
    
    assert op(sup(constcore), sup(constcore)).__class__ is sup
    assert op(cls(constcore), sup(constcore)).__class__ is sup
    assert op(sup(constcore), cls(constcore)).__class__ is sup
    assert op(sup(constcore), lgp.Kernel(constcore)).__class__ is lgp.CrossKernel
    assert op(sup(constcore), lgp.CrossKernel(constcore)).__class__ is lgp.CrossKernel

    class A(cls): pass

    assert op(A(constcore), A(constcore)).__class__ is cls
    assert op(A(constcore), cls(constcore)).__class__ is cls
    assert op(A(constcore), lgp.Kernel(constcore)).__class__ is lgp.Kernel

@mark.parametrize('op', [operator.add, operator.mul])
def test_binary_scalar_class(constcore, op):
    k = lgp.Kernel(constcore)
    convs = [
        lambda x: int(x),
        lambda x: float(x),
        np.float64,
        jnp.float64,
        np.array,
        jnp.array,
    ]
    @jax.jit
    def check(x):
        assert op(k, x).__class__ is lgp.Kernel
    for c in convs:
        assert op(k, c(1)).__class__ is lgp.Kernel
        assert op(k, c(0)).__class__ is lgp.Kernel
        assert op(k, c(-1)).__class__ is lgp.CrossKernel
        check(c(1))
        check(c(0))
        check(c(-1))

@mark.parametrize('cls', [lgp.StationaryKernel, lgp.IsotropicKernel])
def test_pow_class(cls, constcore):
    assert (cls(constcore) ** 1).__class__ is cls
    class A(cls): pass
    assert (A(constcore) ** 1).__class__ is cls

def test_algop_type_error(constcore):
    A = lgp.kernel(constcore)
    @A.register_algop
    def ciao(self, *_):
        return self
    a = A()
    with pytest.raises(TypeError):
        a.algop('ciao', 'duo')

def test_batch(rng):
    class A(lgp.CrossKernel): pass
    core = lambda x, y: 1.2 * x + 4.3 * y
    kernel = A(core)
    batched = [kernel.batch(20), A(core, batchbytes=20)]
    for kernel_batched in batched:
        assert kernel.__class__ is kernel_batched.__class__
        x = rng.standard_normal((2, 3, 5, 1))
        y = rng.standard_normal((   1, 5, 7))
        result = kernel(x, y)
        result_batched = kernel_batched(x, y)
        util.assert_allclose(result, result_batched, rtol=1e-15, atol=1e-15)

@pytest.fixture
def idtransf():
    def idtransf(self, a, b):
        """ porco duo """
        return self
    return idtransf

def test_missing_transf(constcore, idtransf):
    kernel = lgp.Kernel(constcore)
    assert not kernel.has_transf('ciao')
    with pytest.raises(KeyError):
        kernel.linop('ciao', None)
    class A(lgp.CrossKernel): pass

    A.register_linop(idtransf, 'ciao')
    assert A.has_transf('ciao')
    assert not lgp.CrossKernel.has_transf('ciao')

def test_already_registered_transf(idtransf):
    with pytest.raises(KeyError):
        lgp.Kernel.register_linop(idtransf, 'normalize')

    class A(lgp.CrossKernel): pass
    class B(A): pass
    A.register_linop(idtransf, 'ciao')
    B.register_linop(idtransf, 'ciao')

def test_transf_nargs(idtransf):
    with pytest.raises(ValueError):
        lgp.Kernel(lambda x, y: 1).linop('normalize', None, None, None)
    with pytest.raises(ValueError):
        lgp.Kernel(lambda x, y: 1).linop('normalize')

def test_no_unnecessary_clones(constcore, idtransf):
    class A(lgp.CrossKernel): pass
    A.register_linop(idtransf, 'ciao')
    a = A(constcore)
    b = a.linop('ciao', 1, 2)
    assert a is b
    assert a._core is b._core

def test_transfclass_nocrossparent(constcore, idtransf):
    class A(lgp.CrossKernel): pass
    A.register_linop(idtransf, 'ciao')
    class B(A): pass
    class C(B, lgp.Kernel): pass
    k = C(constcore)
    q = k.linop('ciao', True)
    assert q.__class__ is A

def test_transf_help(idtransf):
    class A(lgp.CrossKernel): pass
    A.register_linop(idtransf)
    assert A.transf_help('idtransf') == ' porco duo '
    A.register_linop(idtransf, 'gatto', 'duo gatto')
    assert A.transf_help('gatto') == 'duo gatto'
    A.register_algop(idtransf, 'gesu', '3')
    assert A.transf_help('gesu') == '3'

def test_zero(rng, constcore):
    x, y = rng.standard_normal((2, 10))
    zero = lgp._Kernel.Zero()
    util.assert_allclose(zero(x, y), 0)
    
    assert zero.linop('normalize', True) is zero
    assert zero._swap() is zero
    assert zero.batch(1) is zero
    
    assert zero + 1 == 1
    assert 1 + zero == 1
    assert zero * 1 is zero
    assert 1 * zero is zero
    assert zero ** 1 is zero

    other = lgp.Kernel(constcore)
    assert zero + other is other
    assert other + zero is other
    assert zero * other is zero
    assert other * zero is zero
    # TODO test that this works when other is not a superclass of Zero.

    with pytest.raises(TypeError):
        zero + 'gatto'
    with pytest.raises(TypeError):
        zero * 'gatto'
    with pytest.raises(TypeError):
        zero ** 1.
    with pytest.raises(TypeError):
        zero ** -1

@mark.parametrize('name,arg', [
    ('rescale', jnp.cos),
    ('xtransf', jnp.cos),
    ('diff', 1),
    ('loc', 0),
    ('scale', 1),
    ('dim', 'f0'),
    ('maxdim', 1),
    ('derivable', 1),
    ('normalize', True),
])
def test_transf_swap_and_duplicate(name, arg, rng):
    kernel = lgp.CrossKernel(lambda x, y: x.astype(float) + 2 * y.astype(float))
    xy = rng.standard_normal((2, 10))
    if name == 'dim':
        xy = xy.astype([('', float)])
    x, y = xy
    
    c1 = kernel.linop(name, arg, None)(x, y)
    c2 = kernel._swap().linop(name, None, arg)._swap()(x, y)
    util.assert_equal(c1, c2)

    c1 = kernel.linop(name, arg)(x, y)
    c2 = kernel.linop(name, arg, arg)(x, y)
    util.assert_equal(c1, c2)

@mark.parametrize('name,arg', [
    ('rescale', None),
    ('xtransf', None),
    ('diff', None),
    ('diff', 0),
    ('loc', None),
    ('scale', None),
    ('dim', None),
    ('maxdim', None),
    ('derivable', None),
    ('normalize', None),
    ('normalize', False),
])
def test_transf_noop(name, arg, constcore):
    kernel = lgp.Kernel(constcore)
    assert kernel.linop(name, arg) is kernel
    assert kernel.linop(name, arg, arg) is kernel
    class A(lgp.Kernel): pass
    a = A(constcore)
    assert a.linop(name, arg) is a
    assert a.linop(name, arg, arg) is a

@mark.parametrize('name,arg', [
    ('rescale', 1),
    ('xtransf', 1),
    ('diff', -1),
    ('diff', lambda: None),
    ('loc', lambda x: 0),
    ('scale', lambda x: 1),
    ('dim', 0),
    ('maxdim', -1),
    ('maxdim', 9.0),
    ('maxdim', -jnp.inf),
    ('maxdim', 'f0'),
    ('derivable', -1),
    ('derivable', 'f0'),
    ('normalize', jnp.ones(2)),
])
def test_transf_invalid_arg(name, arg, constcore):
    kernel = lgp.Kernel(constcore)
    with pytest.raises((ValueError, TypeError)):
        kernel.linop(name, arg)
    with pytest.raises((ValueError, TypeError)):
        kernel.linop(name, arg, arg)
    with pytest.raises((ValueError, TypeError)):
        kernel.linop(name, arg, None)
    with pytest.raises((ValueError, TypeError)):
        kernel.linop(name, None, arg)

@mark.parametrize('cls', [lgp.StationaryKernel, lgp.IsotropicKernel])
@mark.parametrize('name,arg', [
    ('rescale', jnp.cos),
    ('loc', 0),
    ('scale', 1),
    ('maxdim', 1),
    ('derivable', 1),
    ('normalize', True),
])
def test_transf_class(cls, name, arg, constcore):
    k = cls(constcore)
    q = k.linop(name, arg)
    assert q.__class__ is cls

def test_derivability(constcore):
    kernel = lgp.Kernel(constcore)
    assert kernel.derivable is None
    kernel.linop('diff', 1) # no check

    kernel = lgp.Kernel(constcore, derivable=0)
    with pytest.raises(ValueError):
        kernel.linop('diff', 1) # hard boundary

    kernel = lgp.Kernel(constcore, derivable=1)
    with pytest.warns(UserWarning):
        kernel.linop('diff', (1, 'a', 1, 'b')) # soft boundary

    # check that the operation system does not try to deduce derivability,
    # it is in general impossible
    kernel = lgp.Kernel(constcore, derivable=1) + lgp.Kernel(constcore, derivable=1)
    assert kernel.derivable is None

@mark.parametrize('cls', [lgp.StationaryKernel, lgp.IsotropicKernel])
def test_invalid_input(cls, constcore):
    with pytest.raises(KeyError):
        cls(constcore, input='ciao')

@mark.parametrize('dtype', [int, float, 'i,2i', 'd,2d'])
def test_isotropic_input(rng, dtype):
    def ssd(x, y):
        if x.dtype.names is not None:
            x = recfunctions.structured_to_unstructured(x)
            y = recfunctions.structured_to_unstructured(y)
            return np.sum((x - y) ** 2, axis=-1)
        else:
            return (x - y) ** 2
    k1 = lgp.IsotropicKernel(lambda x, y: np.exp(-ssd(x, y)), input='raw')
    k2 = lgp.IsotropicKernel(lambda r: np.exp(-r ** 2), input='abs')
    k3 = lgp.IsotropicKernel(lambda r: np.exp(-r ** 2), input='posabs')
    k4 = lgp.IsotropicKernel(lambda r2: np.exp(-r2), input='squared')
    dtype = np.dtype(dtype)
    if dtype.names is None:
        x, y = rng.standard_normal((2, 10))
    else:
        size = sum(np.prod(f[1].shape, dtype=int)
            for f in recfunctions.flatten_descr(dtype))
        data = rng.standard_normal((2, 10, size))
        x, y = recfunctions.unstructured_to_structured(data, dtype)
    c1 = k1(x, y)
    c2 = k2(x, y)
    c3 = k3(x, y)
    c4 = k4(x, y)
    util.assert_allclose(c1, c2, atol=1e-16)
    util.assert_allclose(c1, c3, atol=1e-15)
    util.assert_allclose(c1, c4, atol=1e-16)

def test_where(rng):

    k = lgp.Kernel(lambda x, y: x * y).transf('forcekron')
        
    x = rng.standard_normal((10, 2)).view('d,d').squeeze(-1)
    x0 = x['f0'][0]
    x = x[:, None]
    cond = lambda x: x['f0'] < x0
    q = lgp.where(cond, k, 2 * k)
    c1 = q(x, x.T)
    c2 = np.where(cond(x) & cond(x.T), k(x, x.T),
                  np.where(~cond(x) & ~cond(x.T), 2 * k(x, x.T), 0))
    util.assert_equal(c1, c2)

def test_diff_errors(rng, constcore):
    kernel = lgp.Kernel(constcore)
    
    # named deriv on scalar
    x = rng.standard_normal(10)
    with pytest.raises(ValueError, match='explicit derivatives'):
        kernel.linop('diff', 'f0')(x, x)
    
    # missing field
    x = rng.standard_normal((10, 2)).view('d,d').squeeze(-1)
    with pytest.raises(ValueError, match='along missing field'):
        kernel.linop('diff', 'a')(x, x)
    
    # derivative on non number
    x = ['abc', 'def']
    with pytest.raises(TypeError, match='along non-numeric'):
        kernel.linop('diff', 1)(x, x)

    # derivative on non number with fields
    x = np.array(['abc', 'def'])
    x = x.view([('', x.dtype)])
    with pytest.raises(TypeError, match='non-numeric field'):
        kernel.linop('diff', 'f0')(x, x)

def test_diff(rng):
    derivs = {
        (0, 0): lambda x, y: x * y,
        (0, 1): lambda x, y: x * np.ones_like(y),
        (1, 0): lambda x, y: np.ones_like(x) * y,
        (2, 0): lambda x, y: np.zeros_like(x * y),
        (1, 1): lambda x, y: np.ones_like(x * y),
        (0, 2): lambda x, y: np.zeros_like(x * y),
    }

    kernel = lgp.Kernel(derivs[0, 0], derivable=2)
    for args, core in derivs.items():
        k = kernel.linop('diff', *args)
        x, y = rng.standard_normal((2, 10))
        util.assert_equal(k(x, y), core(x, y))

    wrapper = lambda core: lambda x, y: core(x['f0'], y['f0'])
    kernel = lgp.Kernel(wrapper(derivs[0, 0]))
    for (i, j), core in derivs.items():
        k = kernel.linop('diff', (i, 'f0'), (j, 'f0'))
        x, y = rng.standard_normal((2, 10)).view([('', float)])
        util.assert_equal(k(x, y), wrapper(core)(x, y))

def test_distances(rng):
    x1 = rng.standard_normal(10)
    x2 = x1 - 1 / np.pi
    if np.any(np.abs(x1 - x2) < 1e-6):
        pytest.xfail(reason='generated values too close')
    
    K = lgp.Expon
    with pytest.warns(UserWarning, match='overriding'):
        c1 = K(input='signed')(x1, x2)
        c2 = K(input='abs')(x1, x2)
        c3 = K(input='posabs')(x1, x2)
    util.assert_allclose(c1, c2, atol=1e-14, rtol=1e-14)
    util.assert_allclose(c1, c3, atol=1e-14, rtol=1e-14)

@mark.parametrize('dec,cls', [
    (lgp.stationarykernel, lgp.StationaryKernel),
    (lgp.isotropickernel, lgp.IsotropicKernel),
])
def test_decorator_kw(dec, cls):

    @dec(input='abs')
    def A(delta, ciao=3):
        return jnp.exp(-delta) + ciao

    assert A().__class__ is A
    with pytest.warns(UserWarning, match='overriding'):
        assert A(input='posabs').__class__ is A
    assert A(scale=5).__class__ is cls
    assert A(loc=5).__class__ is cls

    @dec(loc=1)
    def B(delta, ciao=2):
        return ciao
    assert B(ciao=1).__class__ is B

    if cls is lgp.IsotropicKernel:
        @dec(dim='a')
        def C(delta, ciao=2):
            return ciao
        assert C(ciao=1).__class__ is lgp.StationaryKernel

def test_decorator():
    @lgp.kernel
    def A(x, y):
        return x * y

    assert A().__class__ is A
    assert A(scale=5).__class__ is lgp.Kernel
    assert A(loc=5).__class__ is lgp.Kernel

    sup = next(A._crossmro())
    assert sup.__name__ == 'CrossA'
    assert A.__bases__ == (sup, lgp.Kernel)
    assert sup.__bases__ == (lgp.CrossKernel,)

    with pytest.raises(ValueError):
        lgp.kernel(lambda x: 2, 'gatto')

def test_callable_arg(constcore):
    kernel = lgp.Kernel(constcore, derivable=lambda d: d, d=6)
    assert kernel.derivable == 6

def test_init_kw(constcore):
    kernel = lgp.Kernel(constcore, saveargs=True, cippa=4)
    def check(k):
        assert k._kw['cippa'] == 4
    check(kernel._swap())
    check(kernel.linop('loc', 1, 2))
    check(kernel.transf('forcekron'))

def test_nary(rng):
    x, y = rng.standard_normal((2, 10))
    a = lambda x, y: 2 * x + 3 * y
    b = lambda x, y: 5 * x + 7 * y
    ka = lgp.CrossKernel(a)
    kb = lgp.CrossKernel(b)
    op = lambda f, g: lambda x: f(9 * x) + g(11 * x)
    k = ka._nary(op, [ka, kb], ka._side.LEFT)
    util.assert_equal(k(x, y), a(9 * x, y) + b(11 * x, y))
    k = ka._nary(op, [ka, kb], ka._side.RIGHT)
    util.assert_equal(k(x, y), a(x, 9 * y) + b(x, 11 * y))

def test_dim(rng):
    x = rng.standard_normal(10)[:, None]
    xs = lgp.StructuredArray.from_dict({'a': x, 'b': x})
    kernel = lgp.ExpQuad()
    kernels = kernel.linop('dim', 'a')
    c1 = kernel(x, x.T)
    c2 = kernels(xs, xs.T)
    util.assert_equal(c1, c2)
    with pytest.raises(ValueError):
        kernels(x, x.T)
    with pytest.raises(KeyError):
        kernel.linop('dim', 'c')(xs, xs.T)

def test_scale_int_nd(rng):
    """ test that conversion int -> float on division does not break structured
    dtype dispatcher """
    x = rng.integers(0, 10, (10, 2))
    x = x.view(x.shape[-1] * [('', x.dtype)]).squeeze(-1)
    kernel = lgp.ExpQuad(scale=1)
    kernel(x, x)

def test_stationary_broadcast(rng):
    """ test that broadcasting does not break structured dtype dispatcher that
    computes difference """
    x = rng.integers(0, 10, 10)
    kernel = lgp.Expon()
    kernel(x[:, None], x[None, :])

def test_transf_output_type_error():
    @lgp.kernel
    def A(x, y):
        return x * y

    # generic transformations do not check errors
    @A.register_transf
    def ciao(*_):
        return 'ciao'
    a = A()
    a.transf('ciao')

    # linop checks
    @A.register_linop
    def bau(*_):
        return 'bau'
    a = A()
    with pytest.raises(TypeError, match="linop 'bau'"):
        a.linop('bau', 1)

    # algop checks
    @A.register_algop
    def miao(*_):
        return 'miao'
    a = A()
    with pytest.raises(TypeError, match="algop 'miao'"):
        a.algop('miao', 1)

    # algop accepts NotImplemented
    @A.register_algop
    def piu(*_):
        return NotImplemented
    a = A()
    assert a.algop('piu', 1) is NotImplemented

def test_transf_kind_error():
    @lgp.kernel
    def A(x, y):
        return x * y
    @A.register_transf
    def ciao(_, self, *__):
        return self
    a = A()
    with pytest.raises(ValueError):
        a.linop('ciao', None)
    with pytest.raises(ValueError):
        a.algop('ciao')

def test_crossmro():
    class A(lgp.CrossKernel): pass
    class B(lgp.Kernel): pass
    assert tuple(lgp.CrossKernel._crossmro()) == (lgp.CrossKernel,)
    assert tuple(lgp.Kernel._crossmro()) == (lgp.CrossKernel,)
    assert tuple(A._crossmro()) == (A, lgp.CrossKernel)
    assert tuple(B._crossmro()) == (lgp.CrossKernel,)

def test_dim_preserve_structure():
    @lgp.kernel(dim='f0')
    def A(x, y):
        assert x.dtype.names == ('f0',)
        assert y.dtype.names == ('f0',)
        return x['f0'][..., 0] * y['f0'][..., 0]
    x = np.zeros(10, '2d,d')
    a = A()
    a(x, x)
