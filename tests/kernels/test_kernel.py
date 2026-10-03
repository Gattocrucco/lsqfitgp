# lsqfitgp/tests/kernels/test_kernel.py
#
# Copyright (c) 2023, 2026, Giacomo Petrillo
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

"""Test the generic kernel machinery.

This file shall cover at 100% the `_Kernel` submodule.
"""

import contextlib
import functools
import operator
import warnings

import jax
import numpy as np
import pytest
from jax import jit
from jax import numpy as jnp
from numpy.lib import recfunctions

import lsqfitgp as lgp
from tests import util


@pytest.fixture
def constcore():
    """Return a kernel core that is 1 with the broadcast shape of the inputs."""
    return lambda x, y, **_: jnp.ones(jnp.broadcast_shapes(x.shape, y.shape))


def test_batch(rng):
    """Check that batching the computation preserves the class and the result."""

    class A(lgp.CrossKernel):
        pass

    core = lambda x, y: 1.2 * x + 4.3 * y
    kernel = A(core)
    batched = [kernel.batch(20), A(core, batchbytes=20)]
    for kernel_batched in batched:
        assert kernel.__class__ is kernel_batched.__class__
        x = rng.standard_normal((2, 3, 5, 1))
        y = rng.standard_normal((1, 5, 7))
        result = kernel(x, y)
        result_batched = kernel_batched(x, y)
        util.assert_allclose(result, result_batched, rtol=1e-15, atol=1e-15)


class TestAlgOp:
    """Test the algebraic operations on kernels."""

    @pytest.mark.parametrize('op', [operator.add, operator.mul])
    @pytest.mark.parametrize('cls', [lgp.CrossKernel, lgp.Kernel])
    def test_binary_kernel(self, op, cls, rng):
        """Check that adding or multiplying two kernels operates on their values."""
        f1 = lambda x, y: 1.2 * x + 8.9 * y
        f2 = lambda x, y: 3.4 * x + 5.6 * y
        kernel1 = cls(f1)
        kernel2 = cls(f2)
        kernel = op(kernel1, kernel2)
        x, y = rng.standard_normal((2, 5))
        result = kernel(x, y)
        expected = op(f1(x, y), f2(x, y))
        util.assert_equal(result, expected)

    @pytest.mark.parametrize('op', [operator.add, operator.mul])
    @pytest.mark.parametrize('cls', [lgp.CrossKernel, lgp.Kernel])
    def test_binary_scalar(self, op, cls, rng):
        """Check adding or multiplying a kernel and a scalar, in both orders."""
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

    @pytest.mark.parametrize('op', [operator.add, operator.mul, operator.pow])
    @pytest.mark.parametrize('cls', [lgp.CrossKernel, lgp.Kernel])
    def test_binary_undef(self, op, cls, constcore):
        """Check that operations with unsupported types raise or are delegated."""
        # test that adding a string raises through Python's mechanism
        kernel = cls(constcore)
        with pytest.raises(TypeError):
            op(kernel, 'gatto')
        with pytest.raises(TypeError):
            op('gatto', kernel)

        # test that other classes are delegated
        class A:
            __add__ = __radd__ = __mul__ = __rmul__ = __pow__ = __rpow__ = lambda *_: (
                'ciao'
            )

        assert op(A(), kernel) == 'ciao'
        assert op(kernel, A()) == 'ciao'

    @pytest.mark.parametrize('cls', [lgp.CrossKernel, lgp.Kernel])
    def test_pow(self, cls, rng):
        """Check that a kernel can be raised only to non-negative integer powers."""
        f = lambda x, y: 1.2 * x + 8.9 * y
        for exp in 3, np.int64(3), np.array(3), jnp.array(3):
            kernel = cls(f) ** exp
            x, y = rng.standard_normal((2, 5))
            result = kernel(x, y)
            expected = f(x, y) ** exp
            util.assert_equal(result, expected)

        for exp in 3.0, np.float64(3), np.array(3.0), jnp.array(3.0):
            with pytest.raises(TypeError):
                cls(f) ** exp

        with pytest.raises(TypeError):
            cls(f) ** -1

        @jit
        def traced(exp, x, y):
            return (cls(f) ** exp)(x, y)

        traced(jnp.uint64(3), x, y)
        with pytest.raises(TypeError):
            traced(3.0, x, y)
        with pytest.raises(TypeError):
            traced(3, x, y)

    @pytest.mark.parametrize('cls', [lgp.CrossKernel, lgp.Kernel])
    def test_rpow(self, cls, rng):
        """Check exponentiation with a kernel as exponent, with base >= 1."""
        f = lambda x, y: 1.2 * x + 8.9 * y

        def convs(x):
            yield x
            yield np.float64(x)
            yield jnp.float64(x)
            yield np.array(x)
            yield jnp.array(x)

        for base in convs(1.0):
            kernel = base ** cls(f)
            x, y = rng.standard_normal((2, 5))
            result = kernel(x, y)
            expected = base ** f(x, y)
            util.assert_equal(result, expected)

        @jit
        def traced(base, x, y):
            return (base ** cls(f))(x, y)

        for base in convs(0.9999):
            with pytest.raises(TypeError):
                base ** cls(f)
            traced(base, x, y)  # no bound check under tracing

    @pytest.mark.parametrize('op', [operator.add, operator.mul])
    @pytest.mark.parametrize('cls', [lgp.StationaryKernel, lgp.IsotropicKernel])
    def test_binary_kernel_class(self, op, cls, constcore):
        """Check the class of the sum and product of kernels of different classes."""
        assert op(cls(constcore), cls(constcore)).__class__ is cls
        assert op(cls(constcore), lgp.Kernel(constcore)).__class__ is lgp.Kernel
        assert op(lgp.Kernel(constcore), cls(constcore)).__class__ is lgp.Kernel

        sup = cls.mro()[1]
        assert sup.__name__.startswith('Cross')

        assert op(sup(constcore), sup(constcore)).__class__ is sup
        assert op(cls(constcore), sup(constcore)).__class__ is sup
        assert op(sup(constcore), cls(constcore)).__class__ is sup
        assert op(sup(constcore), lgp.Kernel(constcore)).__class__ is lgp.CrossKernel
        assert (
            op(sup(constcore), lgp.CrossKernel(constcore)).__class__ is lgp.CrossKernel
        )

        class A(cls):
            pass

        assert op(A(constcore), A(constcore)).__class__ is cls
        assert op(A(constcore), cls(constcore)).__class__ is cls
        assert op(A(constcore), lgp.Kernel(constcore)).__class__ is lgp.Kernel

    @pytest.mark.parametrize('op', [operator.add, operator.mul])
    @pytest.mark.parametrize(
        'cls,crosscls',
        [
            (lgp.Kernel, lgp.CrossKernel),
            (lgp.StationaryKernel, lgp.CrossStationaryKernel),
            (lgp.IsotropicKernel, lgp.CrossIsotropicKernel),
        ],
    )
    def test_binary_scalar_class(self, constcore, op, cls, crosscls):
        """Check the class of a kernel plus or times a scalar, cross if negative."""
        k = cls(constcore)
        convs = [
            lambda x: int(x),
            lambda x: float(x),
            np.float64,
            jnp.float64,
            np.array,
            jnp.array,
        ]

        @jit
        def check(x):
            assert op(k, x).__class__ is cls

        for c in convs:
            assert op(k, c(1)).__class__ is cls
            assert op(k, c(0)).__class__ is cls
            assert op(k, c(-1)).__class__ is crosscls
            check(c(1))
            check(c(0))
            check(c(-1))

    @pytest.mark.parametrize('cls', [lgp.StationaryKernel, lgp.IsotropicKernel])
    def test_pow_class(self, cls, constcore):
        """Check that the power of a kernel or subclass has the kernel's class."""
        assert (cls(constcore) ** 1).__class__ is cls

        class A(cls):
            pass

        assert (A(constcore) ** 1).__class__ is cls

    def test_algop_type_error(self, constcore):
        """Check that an algop raises `TypeError` on an argument of the wrong type."""
        A = lgp.kernel(constcore)

        @A.register_algop
        def ciao(tcls, self, *_):
            return self

        a = A()
        with pytest.raises(TypeError):
            a.algop('ciao', 'duo')

    def test_ufunc_algop(self, constcore):
        """Check applying a ufunc to a kernel with `algop`."""
        a = lgp.Kernel(constcore).algop('1/cos')
        util.assert_allclose(a(0, 0), 1 / np.cos(1))

    def test_ufunc_algop_scalar_operand(self):
        """Check a ufunc algop with a scalar operand before a kernel operand."""

        class A(lgp.Kernel):
            pass

        A.register_ufuncalgop(lambda a, b, c: a + 10 * b + 100 * c, 'f3')
        k1 = A(lambda x, y: x * y)
        k2 = A(lambda x, y: x + y)
        k = k1.algop('f3', 3.0, k2)
        util.assert_allclose(k(2.0, 5.0), 2 * 5 + 10 * 3 + 100 * (2 + 5))

    def test_inherit_all_algops(self):
        """Check `inherit_all_algops` on a subclass and on `CrossKernel`."""

        # check that inherit_all_algops does not try to inherit from itself
        class A(lgp.CrossKernel):
            pass

        A.register_algop(lambda tcls, self: self, 'ciao')
        A.inherit_all_algops()

        # check that CrossKernel can't inherit
        with pytest.raises(StopIteration):
            lgp.CrossKernel.inherit_all_algops()


@pytest.fixture
def idtransf():
    """Return an identity transformation with a docstring."""

    def idtransf(tcls, self, a, b):
        """Porco duo."""
        return self

    return idtransf


class TestTransf:
    """Test the registration and application of kernel transformations."""

    def test_missing_transf(self, constcore, idtransf):
        """Check `has_transf`, and that a missing transformation raises `KeyError`."""
        kernel = lgp.Kernel(constcore)
        assert not kernel.has_transf('ciao')
        with pytest.raises(KeyError):
            kernel.linop('ciao', None)

        class A(lgp.CrossKernel):
            pass

        A.register_linop(idtransf, 'ciao')
        assert A.has_transf('ciao')
        assert not lgp.CrossKernel.has_transf('ciao')

    def test_already_registered_transf(self, idtransf):
        """Check that a transformation name can be re-registered only in a subclass."""
        with pytest.raises(KeyError):
            lgp.Kernel.register_linop(idtransf, 'normalize')

        class A(lgp.CrossKernel):
            pass

        class B(A):
            pass

        A.register_linop(idtransf, 'ciao')
        B.register_linop(idtransf, 'ciao')

    def test_transf_help(self, idtransf):
        """Check that `transf_help` returns the docstring or the given help text."""

        class A(lgp.CrossKernel):
            pass

        A.register_linop(idtransf)
        assert A.transf_help('idtransf') == idtransf.__doc__
        A.register_linop(idtransf, 'gatto', 'duo gatto')
        assert A.transf_help('gatto') == 'duo gatto'
        A.register_algop(idtransf, 'gesu', '3')
        assert A.transf_help('gesu') == '3'

    def test_output_type_error(self):
        """Check which kinds of transformations check the type of their output."""

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

    def test_kind_error(self):
        """Check that using a transformation of the wrong kind raises `ValueError`."""

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

    def test_forcekron(self, constcore):
        """Check the `forcekron` option and transformation, also with `maxdim`."""
        # at initialization it's within maxdim
        kernel = lgp.Kernel(constcore, forcekron=True, maxdim=1)
        x = np.empty(1, 'f,f')
        with pytest.raises(ValueError, match='> maxdim='):
            kernel(x, x)

        # after initialization it's not, no error
        kernel = kernel.transf('forcekron')
        kernel(x, x)

        # promotes to Kernel since it changes the core
        class A(lgp.Kernel):
            pass

        a = A(constcore)
        assert a.transf('forcekron').__class__ is lgp.Kernel

        # not defined for CrossKernel
        with pytest.raises(KeyError, match='forcekron'):
            lgp.CrossKernel(constcore, forcekron=True)

    def test_list_transf(self):
        """Check that `list_transf` lists the transformations of a class and bases."""

        # register a transf on a new class
        class A(lgp.CrossKernel):
            pass

        @functools.partial(A.register_transf, kind=7)
        def ciao(self, *_):
            """Ciao."""
            pass

        # check the transf is there, and also those of the superclass
        t = A.list_transf()
        assert t['ciao'] == (A, 7, ciao, ciao.__doc__)
        assert 'add' in t

        # check only that transf is in the new class
        t = A.list_transf(superclasses=False)
        assert len(t) == 1 and 'ciao' in t

    def test_super(self, constcore):
        """Check that `super_transf` invokes the transformation of the superclass."""

        class A(lgp.Kernel):
            pass

        class B(A):
            pass

        @A.register_transf
        def ciao(tcls, self, *args):
            return ' '.join(map(str, args))

        @B.register_transf
        def ciao(tcls, self, *args):
            return 'ciao ' + tcls.super_transf('ciao', self, *args)

        a = A(constcore)
        b = B(constcore)
        assert a.transf('ciao', 1, 2) == '1 2'
        assert b.transf('ciao', 1, 2) == 'ciao 1 2'

    def test_super_multiple_inheritance(self, constcore):
        """Check that `super_transf` follows the MRO with multiple inheritance."""

        # class D has mro C, B, A
        class A(lgp.Kernel):
            pass

        class B(A):
            pass

        class C(A):
            pass

        class D(C, B):
            pass

        # set up transformations that return the class name
        @A.register_transf
        def who(tcls, self):
            return tcls

        B.inherit_transf('who')

        # make the transf on D invoke superclasses
        @D.register_transf
        def who(tcls, self):
            return tcls.super_transf('who', self)

        # check the MRO is respected
        d = D(constcore)
        assert d.transf('who') is B


class TestLinOp:
    """Test the linear operators on kernels."""

    def test_args_errors(self, idtransf):
        """Check that `linop` raises `ValueError` on the wrong number of arguments."""
        with pytest.raises(ValueError, match='incorrect number of'):
            lgp.Kernel(lambda x, y: 1).linop('normalize', None, None, None)
        with pytest.raises(ValueError, match='incorrect number of'):
            lgp.Kernel(lambda x, y: 1).linop('normalize')

    def test_no_unnecessary_result_clone(self, constcore, idtransf):
        """Check that a linop returning the kernel itself does not clone it."""

        class A(lgp.CrossKernel):
            pass

        A.register_linop(idtransf, 'ciao')
        a = A(constcore)
        b = a.linop('ciao', 1, 2)
        assert a is b
        assert a.core is b.core

    def test_class_goes_to_cross_parent(self, constcore, idtransf):
        """Check that the result of a linop has the cross class that defines it."""

        class A(lgp.CrossKernel):
            pass

        A.register_linop(idtransf, 'ciao')

        class B(A):
            pass

        class C(B, lgp.Kernel):
            pass

        k = C(constcore)
        q = k.linop('ciao', True)
        assert q.__class__ is A

    def test_result_out_of_transf_tree(self, constcore):
        """Check that the result of a linop is not always enforced to its class.

        If the result is not a descendant of the class defining the
        transformation, it is not enforced to that class.
        """

        class A(lgp.CrossKernel):
            pass

        class B(lgp.CrossKernel):
            pass

        @A.register_linop
        def op(tcls, self, arg1, arg2):
            return B(constcore)

        assert A(constcore).linop('op', 1, 2).__class__ is B

    @pytest.mark.parametrize(
        'name,arg',
        [
            ('rescale', jnp.cos),
            ('xtransf', jnp.cos),
            ('diff', 1),
            ('loc', 0),
            ('scale', 1),
            ('dim', 'f0'),
            ('maxdim', 1),
            ('derivable', 1),
            ('normalize', True),
        ],
    )
    def test_swap_and_duplicate(self, name, arg, rng):
        """Check linops under swapping, and that a single argument applies to both."""
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

    @pytest.mark.parametrize(
        'name,arg',
        [
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
        ],
    )
    def test_identity_noop(self, name, arg, constcore):
        """Check that linops with identity arguments return the kernel unchanged."""
        kernel = lgp.Kernel(constcore)
        assert kernel.linop(name, arg) is kernel
        assert kernel.linop(name, arg, arg) is kernel

        class A(lgp.Kernel):
            pass

        a = A(constcore)
        assert a.linop(name, arg) is a
        assert a.linop(name, arg, arg) is a

    @pytest.mark.parametrize(
        'name,arg',
        [
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
        ],
    )
    def test_invalid_arg(self, name, arg, constcore):
        """Check that linops reject invalid arguments on either side."""
        kernel = lgp.Kernel(constcore)
        with pytest.raises((ValueError, TypeError)):
            kernel.linop(name, arg)
        with pytest.raises((ValueError, TypeError)):
            kernel.linop(name, arg, arg)
        with pytest.raises((ValueError, TypeError)):
            kernel.linop(name, arg, None)
        with pytest.raises((ValueError, TypeError)):
            kernel.linop(name, None, arg)

    @pytest.mark.parametrize(
        'cls',
        [
            lgp.CrossStationaryKernel,
            lgp.StationaryKernel,
            lgp.CrossIsotropicKernel,
            lgp.IsotropicKernel,
        ],
    )
    @pytest.mark.parametrize(
        'name,arg,nops',
        [
            ('rescale', jnp.cos, 0),
            ('loc', 0, 0),
            ('scale', 1, 0),
            ('maxdim', 1, 0),
            ('derivable', 1, 0),
            ('normalize', True, 0),
            ('cond', lambda: None, 1),
        ],
    )
    def test_isotropic_ops(self, cls, name, arg, nops, constcore):
        """Check that these ops preserve `IsotropicKernel` and its ancestors."""
        k = cls(constcore)
        q = k.linop(name, *nops * [k], arg)
        assert q.__class__ is cls

    def test_cond(self, rng):
        """Check that the `'cond'` linop switches between two kernels."""
        k = lgp.Kernel(lambda x, y: x * y).transf('forcekron')

        x = rng.standard_normal((10, 2)).view('d,d').squeeze(-1)
        x0 = x['f0'][0]
        x = x[:, None]
        cond = lambda x: x['f0'] < x0
        q = k.linop('cond', 2 * k, cond)
        c1 = q(x, x.T)
        c2 = np.where(
            cond(x) & cond(x.T),
            k(x, x.T),
            np.where(~cond(x) & ~cond(x.T), 2 * k(x, x.T), 0),
        )
        util.assert_equal(c1, c2)

    def test_diff_errors(self, rng, constcore):
        """Trigger errors not related to `derivable`."""
        kernel = lgp.Kernel(constcore)

        # named deriv on scalar
        x = rng.standard_normal(10)
        with pytest.raises(ValueError, match='derivative on named'):
            kernel.linop('diff', 'f0')(x, x)

        # missing field
        x = rng.standard_normal((10, 2)).view('d,d').squeeze(-1)
        with pytest.raises(ValueError, match='along missing field'):
            kernel.linop('diff', 'a')(x, x)

        # derivative on non-number
        x = ['abc', 'def']
        with pytest.raises(TypeError, match='along non-numeric'):
            kernel.linop('diff', 1)(x, x)

        # derivative on non-number with fields
        x = np.array(['abc', 'def'])
        x = x.view([('', x.dtype)])
        with pytest.raises(TypeError, match='non-numeric field'):
            kernel.linop('diff', 'f0')(x, x)

    def test_diff_value(self, rng):
        """Check the derivatives of a bilinear kernel on plain and structured inputs."""
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

    def test_diff_cross_nd(self, rng):
        """Test `diff` when one argument is scalar and the other structured."""
        x = rng.standard_normal((10, 2)).view('d,d').squeeze(-1)
        y = rng.standard_normal(10)
        k1 = lgp.Kernel(lambda x, y: x['f0'] * y)
        k2 = lgp.Kernel(lambda x, y: x * y)
        c1 = k1.linop('diff', (1, 'f0'), 1)(x, y)
        c2 = k2.linop('diff', 1, 1)(x['f0'], y)
        util.assert_equal(c1, c2)

    def test_derivable(self, constcore, rng):
        """Check the derivability checks configured with the `derivable` argument."""
        # default no derivability check
        x = rng.standard_normal(10)
        kernel = lgp.Kernel(constcore)
        kernel.linop('diff', 1)(x, x)

        # forbid derivatives
        kernel = lgp.Kernel(constcore, derivable=0)
        with pytest.raises(ValueError, match='derivatives'):
            kernel.linop('diff', 1)(x, x)

        # keeping total order within bound does not fool the checker
        kernel = lgp.Kernel(constcore, derivable=1)
        kernel.linop('diff', 1)(x, x)
        with pytest.raises(ValueError, match='derivatives'):
            kernel.linop('diff', 2, 0)(x, x)

        # check is separate by argument
        kernel = lgp.Kernel(constcore, derivable=(1, 0))
        kernel.linop('diff', 1, 0)(x, x)
        with pytest.raises(ValueError, match='derivatives'):
            kernel.linop('diff', 0, 1)(x, x)

        # check distinguishes different baked-in fields
        k = lgp.Kernel(constcore, derivable=1, dim='f0')
        q = lgp.Kernel(constcore, derivable=2, dim='f1')
        n = k + q
        y = rng.standard_normal((10, 2)).view('d,d').squeeze(-1)
        n(y, y)
        n.linop('diff', 'f0')(y, y)
        n.linop('diff', (2, 'f1'))(y, y)
        n.linop('diff', ('f0', 2, 'f1'))(y, y)
        with pytest.raises(ValueError, match='derivatives'):
            n.linop('diff', (2, 'f0'))(y, y)
        with pytest.raises(ValueError, match='derivatives'):
            n.linop('diff', (3, 'f1'))(y, y)

        # check looks at all fields
        k = lgp.Kernel(constcore, derivable=1)
        k.linop('diff', 'f0')(y, y)
        k.linop('diff', 'f1')(y, y)
        with pytest.raises(ValueError, match='derivatives'):
            k.linop('diff', (2, 'f0'))(y, y)
            k.linop('diff', (2, 'f1'))(y, y)

        # check looks at total order with nd inputs
        with pytest.raises(ValueError, match='derivatives'):
            k.linop('diff', ('f0', 'f1'))(y, y)
        with pytest.raises(ValueError, match='derivatives'):
            k.linop('diff', ('f0', 'f1'), None)(y, y)

    @pytest.mark.xfail(
        reason='derivability check does not ignore extraneous derivatives'
    )
    def test_derivable_foreign(self, rng):
        """Test that deriving w.r.t. other stuff does not trigger derivability checks.

        It should be possible to derive w.r.t. other stuff that goes through `x`
        without triggering derivability checks.
        """

        # pass derived quantities through init arguments
        @jax.jacfwd
        def f(val, x, y):
            k = lgp.Kernel(lambda x, y: x * y, derivable=False, loc=val)
            return k(x, y)

        x, y = rng.standard_normal((2, 10))
        f(1.0, x, y)

        # pass derived quantities afterwards
        @jax.jacfwd
        def f(val, x, y):
            k = lgp.Kernel(lambda x, y: x * y, derivable=False).linop('loc', val)
            return k(x, y)

        f(1.0, x, y)

    def test_dim(self, rng):
        """Check that the `'dim'` linop selects a field of the input."""
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

    def test_dim_preserve_structure(self):
        """Check that `dim` passes to the core an array with only the selected field."""

        @lgp.kernel(dim='f0')
        def A(x, y):
            assert x.dtype.names == ('f0',)
            assert y.dtype.names == ('f0',)
            return x['f0'][..., 0] * y['f0'][..., 0]

        x = np.zeros(10, '2d,d')
        a = A()
        a(x, x)

    @pytest.mark.parametrize('rightker', [False, True])
    @pytest.mark.parametrize('doc', [None, 'miao'])
    @pytest.mark.parametrize('argnames', [None, ('xbau', 'ybau')])
    @pytest.mark.parametrize('nonsym', [False, True])
    def test_make_linop_family(self, rng, rightker, doc, argnames, nonsym):
        """Check the classes, arguments and docs of a `make_linop_family` family."""
        decorator = lgp.crosskernel if nonsym else lgp.kernel

        @decorator
        def A(x, y, *, gatto):
            return gatto * x * y

        @lgp.kernel
        def B(a, b, *, gatto, xbau=5, ybau=7):
            return gatto * xbau * ybau * a * b

        @lgp.crosskernel
        def CrossBA(a, y, *, gatto, xbau=2, ybau=3):
            return gatto * xbau * ybau * a * y

        CrossBA._swap = lambda self: super(CrossBA, self)._swap()._clone(CrossBA)
        if doc:
            CrossBA.__doc__ = doc

        if rightker:

            @lgp.crosskernel
            def CrossAB(y, a, *, gatto, xbau=2, ybau=3):
                return gatto * ybau * xbau * y * a
        else:
            CrossAB = None

        if nonsym:

            @contextlib.contextmanager
            def context():
                with pytest.warns(
                    UserWarning, match='non-Kernel, Kernel, non-Kernel, non-Kernel'
                ) as w:
                    yield w
        else:

            @contextlib.contextmanager
            def context():
                with warnings.catch_warnings():
                    warnings.simplefilter('error')
                    yield

        with context():
            A.make_linop_family('ciao', B, CrossBA, CrossAB, argnames=argnames)

        # produce instances
        aa = A(gatto=11)
        bb = aa.linop('ciao', 13, 13)
        ba = aa.linop('ciao', 13, None)
        bb1 = ba.linop('ciao', None, 13)
        ab = aa.linop('ciao', None, 13)
        bb2 = ab.linop('ciao', 13, None)

        # check classes of instances
        assert aa.__class__ is A
        assert ba.__class__ is CrossBA
        assert bb.__class__ is B
        assert bb1.__class__ is B
        assert bb2.__class__ is B

        # check keyword argument passing
        assert aa(1, 1) == 11
        assert ab(1, 1) == (11 * 2 * 13 if argnames else 11 * 2 * 3)
        assert ba(1, 1) == (11 * 13 * 3 if argnames else 11 * 2 * 3)
        assert bb(1, 1) == (11 * 13 * 13 if argnames else 11 * 5 * 7)
        assert bb1(1, 1) == (11 * 13 * 13 if argnames else 11 * 5 * 7)
        assert bb2(1, 1) == (11 * 13 * 13 if argnames else 11 * 5 * 7)

        # check instance with automatically defined class
        if not rightker:
            CrossAB = ab.__class__
            assert CrossAB.__name__ == 'CrossAB'
            assert CrossAB.__bases__ == (CrossBA,)
            if doc:
                assert (
                    CrossAB.__doc__
                    == """\
Automatically generated transposed version of:

miao"""
                )

            # check that automatically generated class is not peremptorily enforced
            assert CrossAB(dim='a').__class__ is lgp.CrossKernel

        # check errors on double transformation
        with pytest.raises(ValueError, match='cannot further transform'):
            ab.linop('ciao', None, 1)
        with pytest.raises(ValueError, match='cannot further transform'):
            ba.linop('ciao', 1, None)

        # check there's not transf on last object
        assert not bb.has_transf('ciao')


class TestStationaryIsotropic:
    """Test stationary and isotropic kernels."""

    @pytest.mark.parametrize('cls', [lgp.StationaryKernel, lgp.IsotropicKernel])
    def test_invalid_input(self, cls, constcore):
        """Check that an invalid `input` argument raises `KeyError`."""
        with pytest.raises(KeyError):
            cls(constcore, input='ciao')

    @pytest.mark.parametrize('dtype', [int, float, 'i,2i', 'd,2d'])
    def test_isotropic_input(self, rng, dtype):
        """Check that the `input` options of `IsotropicKernel` give the same result."""

        def ssd(x, y):
            if x.dtype.names is not None:
                x = recfunctions.structured_to_unstructured(x)
                y = recfunctions.structured_to_unstructured(y)
                return np.sum((x - y) ** 2, axis=-1)
            else:
                return (x - y) ** 2

        k1 = lgp.IsotropicKernel(lambda x, y: np.exp(-ssd(x, y)), input='raw')
        k2 = lgp.IsotropicKernel(lambda r: np.exp(-(r**2)), input='abs')
        k3 = lgp.IsotropicKernel(lambda r: np.exp(-(r**2)), input='posabs')
        k4 = lgp.IsotropicKernel(lambda r2: np.exp(-r2), input='squared')
        dtype = np.dtype(dtype)
        if dtype.names is None:
            x, y = rng.standard_normal((2, 10))
        else:
            size = sum(
                np.prod(f[1].shape, dtype=int)
                for f in recfunctions.flatten_descr(dtype)
            )
            data = rng.standard_normal((2, 10, size))
            x, y = recfunctions.unstructured_to_structured(data, dtype)
        c1 = k1(x, y)
        c2 = k2(x, y)
        c3 = k3(x, y)
        c4 = k4(x, y)
        util.assert_allclose(c1, c2, atol=1e-16)
        util.assert_allclose(c1, c3, atol=1e-15)
        util.assert_allclose(c1, c4, atol=1e-16)

    def test_zero(self, rng, constcore):
        """Check that the `Zero` kernel is zero."""
        x, y = rng.standard_normal((2, 10))
        zero = lgp._Kernel.Zero()
        util.assert_allclose(zero(x, y), 0)

    def test_stationary_distances(self, rng):
        """Check that the `input` options of a stationary kernel agree."""
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

    def test_stationary_broadcast(self, rng):
        """Test that broadcasting does not break the structured dtype dispatcher.

        The dispatcher is the one that computes the difference.
        """
        x = rng.integers(0, 10, 10)
        kernel = lgp.Expon()
        kernel(x[:, None], x[None, :])

    def test_scale_int_nd(self, rng):
        """Test that conversion int -> float on division works on structured dtypes.

        The conversion must not break the structured dtype dispatcher.
        """
        x = rng.integers(0, 10, (10, 2))
        x = x.view(x.shape[-1] * [('', x.dtype)]).squeeze(-1)
        kernel = lgp.ExpQuad(scale=1)
        kernel(x, x)


class TestDecorator:
    """Test the kernel class decorators."""

    def test_class_change(self):
        """Check when instances of a decorated kernel class are demoted to `Kernel`."""

        @lgp.kernel
        def A(x, y):
            return x * y

        assert A().__class__ is A
        assert A(scale=5).__class__ is lgp.Kernel
        assert A(loc=5).__class__ is lgp.Kernel

        with pytest.raises(ValueError):
            lgp.kernel(lambda x: 2, 'gatto')

    @pytest.mark.parametrize(
        'dec,cls,crdec,crcls',
        [
            (
                lgp.stationarykernel,
                lgp.StationaryKernel,
                lgp.crossstationarykernel,
                lgp.CrossStationaryKernel,
            ),
            (
                lgp.isotropickernel,
                lgp.IsotropicKernel,
                lgp.crossisotropickernel,
                lgp.CrossIsotropicKernel,
            ),
        ],
    )
    def test_class_change_user_kw(self, dec, cls, crdec, crcls):
        """Check the class of instances of decorated stationary/isotropic kernels."""

        @dec(input='abs')
        def A(delta, ciao=3):
            return jnp.exp(-delta) + ciao

        assert A().__class__ is A
        with pytest.warns(UserWarning, match='overriding'):
            assert A(input='posabs').__class__ is A
        assert A(scale=5).__class__ is cls
        assert A(loc=5).__class__ is cls
        assert A(loc=(1, 1)).__class__ is cls
        assert A(loc=(1, 2)).__class__ is crcls

        @dec(loc=1)
        def B(delta, ciao=2):
            return ciao

        assert B(ciao=1).__class__ is B

        if cls in (lgp.IsotropicKernel, lgp.CrossIsotropicKernel):

            @dec(dim='a')
            def C(delta, ciao=2):
                return ciao

            assert C(ciao=1).__class__ in (
                lgp.StationaryKernel,
                lgp.CrossStationaryKernel,
            )

        @crdec
        def C(delta):
            return 0

        assert C().__class__ is C
        assert isinstance(C(), crcls)


class TestAffineSpan:
    """Test the `AffineSpan` kernel mixin."""

    def test_preserved_class(self, constcore):
        """Test that affine operations preserve the specific subclass."""

        class A(lgp._Kernel.AffineSpan, lgp.Kernel):
            pass

        a = A(constcore)
        assert a.linop('loc', 0).__class__ is A
        assert a.linop('scale', 1).__class__ is A
        assert (a + 0).__class__ is A
        assert (0 + a).__class__ is A
        assert (a * 1).__class__ is A
        assert (1 * a).__class__ is A

    def test_preserved_class_scalar_only(self, constcore):
        """Test that algebraic operations on pairs do not preserve the class."""

        class A(lgp._Kernel.AffineSpan, lgp.Kernel):
            pass

        a = A(constcore)
        assert (a + a).__class__ is lgp.Kernel
        assert (a * a).__class__ is lgp.Kernel

    def test_class_regression(self, constcore):
        """Check that regressing the underlying class is not prevented."""

        class A(lgp._Kernel.AffineSpan, lgp.IsotropicKernel):
            pass

        a = A(constcore)
        assert a.linop('loc', 0).__class__ is A
        assert a.linop('dim', 'a').__class__ is lgp.StationaryKernel

    @pytest.mark.parametrize('op', [operator.add, operator.mul])
    def test_class_negative_scalar(self, constcore, op):
        """Check that a negative scalar preserves the class only for cross kernels."""

        class A(lgp._Kernel.AffineSpan, lgp.Kernel):
            pass

        a = A(constcore)
        assert op(a, 0).__class__ is A
        assert op(a, -1).__class__ is lgp.CrossKernel

        class B(lgp._Kernel.AffineSpan, lgp.CrossKernel):
            pass

        b = B(constcore)
        assert op(b, -1).__class__ is B

    def test_no_instance(self, constcore):
        """Check that `AffineSpan` can't be instantiated directly."""
        with pytest.raises(TypeError, match='cannot instantiate'):
            lgp._Kernel.AffineSpan(constcore)

    def test_calc(self, constcore, rng):
        """Check the coefficients accumulated by `AffineSpan` and the kernel values."""

        class A(lgp._Kernel.AffineSpan, lgp.CrossKernel):
            pass

        a0 = A(constcore)

        x, y = rng.standard_normal((2, 10))

        # apply affine transformations
        a = a0 + 2
        a = a * 3
        a = a + 5
        a = a * 7
        a = a.linop('scale', 2, 3)
        a = a.linop('loc', 5, 7)
        a = a.linop('scale', 11, 13)
        a = a.linop('loc', 17, 19)

        # compare accumulated coefficients with manual calculation
        assert a.dynkw['offset'] == (2 * 3 + 5) * 7
        assert a.dynkw['ampl'] == 3 * 7
        assert a.dynkw['lloc'] == 2 * (5 + 11 * 17)
        assert a.dynkw['rloc'] == 3 * (7 + 13 * 19)
        assert a.dynkw['lscale'] == 2 * 11
        assert a.dynkw['rscale'] == 3 * 13

        # compare result with specification of coefficients
        c1 = a(x, y)
        c2 = a.dynkw['offset'] + a.dynkw['ampl'] * a0(
            (x - a.dynkw['lloc']) / a.dynkw['lscale'],
            (y - a.dynkw['rloc']) / a.dynkw['rscale'],
        )
        util.assert_allclose(c1, c2)


def test_callable_arg(constcore, rng):
    """Check that a callable init argument is evaluated on the other arguments."""
    x = rng.standard_normal(10)
    kernel = lgp.Kernel(constcore, derivable=lambda d: d, d=1)
    with pytest.raises(ValueError, match='derivatives'):
        kernel.linop('diff', 2)(x, x)


def test_init_kw_preserved(constcore):
    """Check that the init keyword arguments are preserved by transformations."""
    kernel = lgp.Kernel(constcore, cippa=4)

    def check(k):
        assert k.initkw['cippa'] == 4

    check(kernel._swap())
    check(kernel.linop('loc', 1, 2))
    check(kernel.transf('forcekron'))


def test_nary(rng):
    """Check `_nary` applied to the left or right argument."""
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


def test_crossmro():
    """Check `_crossmro` on library and user kernel classes."""

    class A(lgp.CrossKernel):
        pass

    class B(lgp.Kernel):
        pass

    assert tuple(lgp.CrossKernel._crossmro()) == (lgp.CrossKernel,)
    assert tuple(lgp.Kernel._crossmro()) == (lgp.CrossKernel,)
    assert tuple(A._crossmro()) == (A, lgp.CrossKernel)
    assert tuple(B._crossmro()) == (lgp.CrossKernel,)


def test_swap(constcore):
    """Check that `_swap` is a no-op on `Kernel` and demotes cross subclasses."""

    class A(lgp.Kernel):
        pass

    a = A(constcore)
    assert a._swap() is a

    class B(lgp.CrossKernel):
        pass

    b = B(constcore)
    assert b._swap().__class__ is lgp.CrossKernel
