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
import jax
from jax import numpy as jnp
from pytest import mark
import pytest

import lsqfitgp as lgp

from .. import util

@pytest.fixture
def constcore():
    return lambda x, y, **_: jnp.ones(jnp.broadcast_shapes(x.shape, y.shape))

def test_forcekron_maxdim(constcore):
    """ check that using forcekron and maxdim at initialization does not disable
    maxdim, while using forcekron afterwards does """
    kernel = lgp.Kernel(constcore, forcekron=True, maxdim=1)
    x = np.empty(1, 'f,f')
    with pytest.raises(ValueError, match='> maxdim='):
        kernel(x, x)

    kernel = kernel.forcekron()
    kernel(x, x)

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

def test_no_argparser(constcore, idtransf):
    class A(lgp.CrossKernel): pass
    A.register_linop(idtransf, 'ciao')
    a = A(constcore)
    b = a.linop('ciao', 1, 2)
    assert a is not b
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

def test_zero(rng, constcore):
    x, y = rng.standard_normal((2, 10))
    zero = lgp._Kernel.Zero()
    util.assert_allclose(zero(x, y), 0)
    
    assert zero.linop('normalize', True) is zero
    assert zero._swap() is zero
    assert zero.batch(1) is zero
    assert zero.forcekron() is zero
    
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

@mark.parametrize('cls', [lgp.StationaryKernel, lgp.IsotropicKernel])
def test_invalid_input(cls, constcore):
    with pytest.raises(KeyError):
        cls(constcore, input='ciao')

def test_where(rng):

    kernel = lgp.Kernel(lambda x, y: x * y).forcekron()
        
    x = rng.standard_normal((10, 2)).view('d,d').squeeze(-1)
    x0 = x['f0'][0]
    cond = lambda x: x < x0
    k1 = lgp.where(cond, kernel, 2 * kernel, dim='f0')
    k2 = lgp.where(lambda x: cond(x['f0']), kernel, 2 * kernel)
    c1 = k1(x[:, None], x[None, :])
    c2 = k2(x[:, None], x[None, :])
    
    x = x.view([('', 'd', 2)])
    cond = lambda x: x['f0'][..., 0] < x0
    k1 = lgp.where(cond, kernel, 2 * kernel, dim='f0')
    k2 = lgp.where(cond, kernel, 2 * kernel)
    c3 = k1(x[:, None], x[None, :])
    c4 = k2(x[:, None], x[None, :])

    util.assert_equal(c1, c2)
    util.assert_equal(c3, c4)
    util.assert_equal(c1, c3)
    
    x = rng.standard_normal(10)
    with pytest.raises(ValueError):
        k1(x, x)

def test_diff_errors(rng, constcore):
    kernel = lgp.Kernel(constcore)
    
    x = rng.standard_normal(10)
    with pytest.raises(ValueError):
        kernel.linop('diff', 'f0')(x, x)
    
    x = rng.standard_normal((10, 2)).view('d,d').squeeze(-1)
    with pytest.raises(ValueError):
        kernel.linop('diff', 'a')(x, x)
    
    x = ['abc', 'def']
    with pytest.raises(TypeError):
        kernel.linop('diff', 1)(x, x)

def test_distances(rng):
    x1 = rng.standard_normal(10)
    x2 = x1 - 1 / np.pi
    if np.any(np.abs(x1 - x2) < 1e-6):
        pytest.xfail(reason='generated values too close')
    
    K = lgp.Expon
    c1 = K(input='signed')(x1, x2)
    c2 = K(input='abs')(x1, x2)
    c3 = K(input='posabs')(x1, x2)
    util.assert_allclose(c1, c2, atol=1e-14, rtol=1e-14)
    util.assert_allclose(c1, c3, atol=1e-14, rtol=1e-14)

@mark.parametrize('dec', [lgp.stationarykernel, lgp.isotropickernel])
def test_decorator_kw(dec):

    @dec(input='abs')
    def A(delta, ciao=3):
        return jnp.exp(-delta) + ciao

    assert A().__class__ is A
    with pytest.warns(UserWarning, match='overriding'):
        assert A(input='posabs').__class__ is A
    assert A(scale=5).__class__ is lgp.Kernel
    assert A(loc=5).__class__ is lgp.Kernel
    assert A(ciao=2).__class__ is A

    @dec(loc=1)
    def B(delta, ciao=2):
        return ciao
    assert B(ciao=1).__class__ is B

def test_callable_arg(constcore):
    kernel = lgp.Kernel(constcore, derivable=lambda d: d, d=6)
    assert kernel.derivable == 6

def test_initargs(constcore):
    kernel = lgp.Kernel(constcore, saveargs=True, cippa=4)
    def check(k):
        assert k.initargs['cippa'] == 4
    check(kernel._swap())
    check(kernel.linop('loc', 1, 2))
    check(kernel.forcekron())

def test_nary():
    pass

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
