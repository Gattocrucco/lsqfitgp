# lsqfitgp/tests/kernels/test_kernels.py
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

""" Test predefined kernels """

import inspect
import re
import types
import functools

import pytest
from pytest import mark
import numpy as np
from scipy import linalg, optimize
import jax

import lsqfitgp as lgp
from lsqfitgp import _kernels, _Kernel

from .. import util

pytestmark = pytest.mark.filterwarnings(
    r'ignore:overriding init argument\(s\)',
    r'ignore:Output seems independent of input',
)

# list of all the ops in _Kernel/_alg.py
unary_algops = [
    ('rpow', (-np.inf, np.inf), dict(base=1.5)),
    ('tan', (-np.pi / 2, np.pi / 2), {}),
    # ('1/sinc', (-np.pi, np.pi), {}),         # <--- TODO fix this, what's wrong?
    ('1/cos', (-np.pi / 2, np.pi / 2), {}),
    ('arcsin', (-1, 1), {}),
    ('1/arccos', (-1, 1), {}),
    ('1/(1-x)', (-np.inf, 1), {}),      # the convergence circle is in (-1, 1), but
    ('exp', (-np.inf, np.inf), {}),     # if cov <= 1 then also cov >= -1, so it's
    ('-log1p(-x)', (-np.inf, 1), {}),   # automatically satisfied.
    ('expm1', (-np.inf, np.inf), {}),
    ('expm1x', (-np.inf, np.inf), {}),
    ('sinh', (-np.inf, np.inf), {}),
    ('cosh', (-np.inf, np.inf), {}),
    ('arctanh', (-1, 1), {}),
    ('i0', (-np.inf, np.inf), {}),
    ('i1', (-np.inf, np.inf), {}),
    # ('iv', (0, np.inf), dict(order=2.5)), # seems to work but too inaccurate
]

def find_pos_rescale_to_fit(l, r, a, b, L, R):
    """
    (l, r) = target domain
    (a, b) = current range
    find z, f >= 0 s.t.
    z + f * (a, b) is contained in (l, r) and (L, R)
    and f is maximum
    return z, f
    """
    C = [0, -1]
    A = [[-1, -a], [1, b]]
    B = [min(-l, -L), min(r, R)]
    res = optimize.linprog(C, A, B, bounds=2 * [(0, 10)])
    assert res.success, res.message
    return res.x

# TODO test higher dimensions and derivatives?

def skipon(meth, exc, match):
    """ decorator to make a unit test method skip on a certain exception """

    @functools.wraps(meth)
    def newmeth(self, *args, **kw):
        try:
            return meth(self, *args, **kw)
        except exc as e:
            if re.search(match, str(e)):
                pytest.skip(reason=str(e))
            else:
                raise
    return newmeth

skiponmaxdim = functools.partial(skipon, exc=ValueError, match=r'> maxdim=')
skiponderivable = functools.partial(skipon, exc=ValueError, match=r'derivatives')

class Base:
    """
    Base class to test kernels. Each subclass tests one specific kernel.
    """
    
    @property
    def kercls(self):
        """ Kernel subclass to test """
        clsname = self.__class__.__name__
        assert clsname.startswith('Test')
        cls = getattr(_kernels, clsname[4:])
        if issubclass(cls, lgp.StationaryKernel):
            assert issubclass(self.__class__, Stationary)
        return cls
    
    def test_public(self):
        assert self.kercls in vars(lgp).values()

    @pytest.fixture
    def kw(self):
        """ Keyword arguments for the constructor, not used directly by the
        tests if not for inspection """
        return {}

    @pytest.fixture
    def kernel(self, kw):
        """ A kernel instance """
        return self.kercls(**kw)

    @pytest.fixture
    def ranx_scalar(self, rng):
        """ A callable generating a random vector of scalar input values,
        not used directly by tests """
        return lambda size=100: rng.uniform(-5, 5, size=size)

    @pytest.fixture
    def ranx_nd(self, ranx_scalar):
        """ A callable generating a random vector of sized input values,
        not used directly by tests """
        def gen(ndim, size=(100,)):
            x = ranx_scalar(size + (ndim,))
            return x.view([('', x.dtype)] * ndim).squeeze(-1)
        return gen

    @pytest.fixture
    def x_scalar(self, ranx_scalar):
        """ A random vector of scalar input values """
        return ranx_scalar()

    @pytest.fixture
    def x_nd(self, ranx_nd):
        """ A random vector of sized input values """
        return ranx_nd(3)

    @pytest.fixture
    def x(self, x_scalar):
        """ A random vector of input values with unspecified dtype structure,
        default scalar """
        return x_scalar

    @pytest.fixture
    def psdeps(self):
        """ Relative tolerance for the smallest eigenvalue to be negative """
        return np.finfo(float).eps

    def check_sym_and_psd(self, cov, psdeps):
        util.assert_allclose(cov, cov.T, rtol=1e-5, atol=1e-7)
        eigv = linalg.eigvalsh(cov)
        assert np.min(eigv) >= -len(cov) * psdeps * np.max(eigv)

    def impl_positive(self, kernel, deriv, x, psdeps, doops=False):
        d = (deriv, 'f0') if x.dtype.names else deriv
        k = kernel.linop('diff', d, d)
        cov = k(x[None, :], x[:, None])
        self.check_sym_and_psd(cov, psdeps)

        if not doops:
            return
        for op, (l, r), kw in unary_algops:
            a, b = np.min(cov), np.max(cov)
            z, f = find_pos_rescale_to_fit(l + 0.01, r - 0.01, a, b, -5, 5)
            mat = lgp.Kernel(lambda *_: (z + f * cov)).algop(op, **kw)(x[None, :], x[:, None])
            self.check_sym_and_psd(mat, psdeps)

    def test_positive_scalar_0(self, kernel, x_scalar, psdeps):
        self.impl_positive(kernel, 0, x_scalar, psdeps, True)

    @skiponmaxdim
    def test_positive_nd_0(self, kernel, x_nd, psdeps):
        self.impl_positive(kernel, 0, x_nd, psdeps)

    def impl_jit(self, kernel, deriv, x):
        if x.dtype.names:
            x = lgp.StructuredArray(x)
            dtype = np.result_type(*(x.dtype[name].base for name in x.dtype.names))
            deriv = deriv, 'f0'
        else:
            dtype = x.dtype
        if not np.issubdtype(dtype, np.number) and not dtype == bool:
            pytest.skip()
        kernel = kernel.linop('diff', deriv, deriv)
        cov1 = kernel(x[None, :], x[:, None])
        cov2 = jax.jit(kernel)(x[None, :], x[:, None])
        util.assert_allclose(cov2, cov1, rtol=1e-6, atol=1e-5)

    def test_jit_scalar_0(self, kernel, x_scalar):
        self.impl_jit(kernel, 0, x_scalar)

    @skiponmaxdim
    def test_jit_nd_0(self, kernel, x_nd):
        self.impl_jit(kernel, 0, x_nd)

    # TODO jit with kw as arguments. use static_argnames filtering by dtype.
    
    # TODO test vmap
    
    @pytest.fixture(params=[(0, 0)])
    def derivs(self, request):
        """ Pair of numbers of derivatives to take """
        return request.param
    
    @skiponderivable
    def test_symmetric_offdiagonal(self, kernel, derivs, x):
        xderiv, yderiv = derivs
        if xderiv == yderiv:
            a = x[:x.size // 2, None]
            b = x[None, x.size // 2:]
        else:
            a = x[:, None]
            b = a.T
        if x.dtype.names:
            xderiv = xderiv, x.dtype.names[0]
            yderiv = yderiv, x.dtype.names[0]
        b1 = kernel.linop('diff', xderiv, yderiv)(a, b)
        b2 = kernel.linop('diff', yderiv, xderiv)(b, a)
        util.assert_allclose(b1, b2, atol=1e-10)
    
    @staticmethod
    def make_x_nd_implicit(x):
        from numpy.lib import recfunctions
        dtype = recfunctions.repack_fields(x.dtype, align=False, recurse=True)
        return x.astype(dtype).view([('', x.dtype[0], (len(x.dtype),))]).copy()
    
    @skiponmaxdim
    def test_implicit_fields(self, kernel, x_nd):
        x1 = x_nd[:, None]
        x2 = self.make_x_nd_implicit(x1)
        c1 = kernel(x1, x1.T)
        c2 = kernel(x2, x2.T)
        util.assert_allclose(c1, c2, atol=1e-15, rtol=1e-14)
    
    @skiponmaxdim
    def test_loc_scale_nd(self, kernel, x_nd):
        loc = -2  # < 0
        scale = 3 # > abs(loc)
        # TODO maybe put loc and scale in kw and let random_x adapt the
        # domain to loc and scale
        if not np.issubdtype(x_nd.dtype[0].base, np.inexact):
            pytest.skip()
        x1 = x_nd[:, None]
        x2 = self.make_x_nd_implicit(x1)
        x2['f0'] -= loc
        x2['f0'] /= scale
        kernel1 = kernel.linop('scale', scale).linop('loc', loc)
        c1 = kernel1(x1, x1.T)
        c2 = kernel(x2, x2.T)
        util.assert_allclose(c1, c2, rtol=1e-12, atol=1e-13)
    
    def impl_continuous_in_zero(self, kernel, deriv, x_scalar, kw):
        pytest.skip(reason='not a stationary kernel test class')

class Stationary(Base):
    """ Test class for kernels that may be a subclass of StationaryKernel """

    def impl_continuous_in_zero(self, kernel, deriv, x_scalar, kw):
        if not isinstance(kernel, _Kernel.StationaryKernel):
            pytest.skip()
        if not np.issubdtype(x_scalar.dtype, np.inexact):
            pytest.skip()
        
        # discontinuos kernels
        if self.kercls is _kernels.Cauchy and kw.get('alpha', 2) < 1:
            pytest.skip()
        if self.kercls is _kernels.GammaExp and kw.get('gamma', 1) < 1:
            pytest.skip()
        if self.kercls is _kernels.Matern and kw['nu'] - deriv < 0.5:
            pytest.skip()
        if self.kercls is _kernels.StationaryFracBrownian and kw.get('H', 0.5) < 0.5:
            pytest.skip()
        if self.kercls is _kernels.White:
            pytest.skip()
        if self.kercls is _kernels.Zeta and kw['nu'] - deriv < 0.5:
            pytest.skip()
        
        kernel = kernel.linop('diff', deriv)
        c0 = kernel(0, 0)
        c1 = kernel(0, 1e-15)
        util.assert_allclose(c1, c0, rtol=1e-10)

    def test_continuous_in_zero_0(self, kernel, x_scalar, kw):
        self.impl_continuous_in_zero(kernel, 0, x_scalar, kw)

    def test_stationary(self, x_scalar, kernel):
        if isinstance(kernel, _Kernel.StationaryKernel):
            var = kernel(x_scalar, x_scalar)
            util.assert_allclose(var, var[0])
        else:
            pytest.skip()

    def test_normalized(self, kernel, x_scalar):
        if isinstance(kernel, _Kernel.StationaryKernel):
            var = kernel(x_scalar, x_scalar)
            util.assert_allclose(var, 1, rtol=1e-14, atol=1e-15)
        else:
            pytest.skip()
    
class Deriv1(Base):
    """ Test class for kernels that may be derived at least one time """

    @pytest.fixture(params=[(0, 0), (1, 0), (1, 1)])
    def derivs(self, request):
        return request.param

    @skiponmaxdim
    @skiponderivable
    def test_positive_nd_1(self, kernel, x_nd, psdeps):
        self.impl_positive(kernel, 1, x_nd, psdeps)

    @skiponderivable
    def test_positive_scalar_1(self, kernel, x_scalar, psdeps):
        self.impl_positive(kernel, 1, x_scalar, psdeps)

    @skiponderivable
    def test_jit_scalar_1(self, kernel, x_scalar):
        self.impl_jit(kernel, 1, x_scalar)

    @skiponmaxdim
    @skiponderivable
    def test_jit_nd_1(self, kernel, x_nd):
        self.impl_jit(kernel, 1, x_nd)

    @skiponderivable
    def test_continuous_in_zero_1(self, kernel, x_scalar, kw):
        self.impl_continuous_in_zero(kernel, 1, x_scalar, kw)

    @skiponderivable
    def test_double_diff_scalar_first(self, kernel, x_scalar):
        r1 = kernel.linop('diff', 1, 1)(x_scalar[None, :], x_scalar[:, None])
        r2 = kernel.linop('diff', 1, 0).linop('diff', 0, 1)(x_scalar[None, :], x_scalar[:, None])
        util.assert_allclose(r1, r2)
    
    @skiponmaxdim
    @skiponderivable
    def test_double_diff_nd_first(self, kernel, x_nd):
        x = x_nd[:, None]
        f0 = x.dtype.names[0]
        r1 = kernel.linop('diff', f0, f0)(x, x.T)
        r2 = kernel.linop('diff', f0, 0).linop('diff', 0, f0)(x, x.T)
        util.assert_allclose(r1, r2)

class Deriv2(Deriv1):
    """ Test class for kernels that may be derived at least two times """

    @pytest.fixture(params=[(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)])
    def derivs(self, request):
        return request.param

    @skiponmaxdim
    @skiponderivable
    def test_positive_nd_2(self, kernel, x_nd, psdeps):
        self.impl_positive(kernel, 2, x_nd, psdeps)

    @skiponderivable
    def test_positive_scalar_2(self, kernel, x_scalar, psdeps):
        self.impl_positive(kernel, 2, x_scalar, psdeps)

    @skiponderivable
    def test_jit_scalar_2(self, kernel, x_scalar):
        self.impl_jit(kernel, 2, x_scalar)

    @skiponmaxdim
    @skiponderivable
    def test_jit_nd_2(self, kernel, x_nd):
        self.impl_jit(kernel, 2, x_nd)

    @skiponderivable
    def test_continuous_in_zero_2(self, kernel, x_scalar, kw):
        self.impl_continuous_in_zero(kernel, 2, x_scalar, kw)

    @skiponderivable
    def test_double_diff_scalar_second(self, kernel, x_scalar):
        r1 = kernel.linop('diff', 2, 2)(x_scalar[None, :], x_scalar[:, None])
        r2 = kernel.linop('diff', 1, 1).linop('diff', 1, 1)(x_scalar[None, :], x_scalar[:, None])
        util.assert_allclose(r1, r2, atol=1e-15, rtol=1e-9)
    
    @skiponderivable
    def test_double_diff_scalar_second_chopped(self, kernel, x_scalar):
        r1 = kernel.linop('diff', 2, 2)(x_scalar[None, :], x_scalar[:, None])
        r2 = kernel.linop('diff', 2, 0).linop('diff', 0, 2)(x_scalar[None, :], x_scalar[:, None])
        util.assert_allclose(r1, r2)
    
    @skiponmaxdim
    @skiponderivable
    def test_double_diff_nd_second(self, kernel, x_nd):
        x = x_nd[:, None]
        if len(x.dtype) < 2:
            pytest.skip(reason='test needs nd >= 2')
        f0, f1 = x.dtype.names[:2]
        r1 = kernel.linop('diff', (2, f0), (2, f1))(x, x.T)
        r2 = kernel.linop('diff', f0, f0).linop('diff', f1, f1)(x, x.T)
        util.assert_allclose(r1, r2)

    @pytest.fixture
    def ddtol(self):
        return dict(atol=1e-15, rtol=1e-12)

    @skiponmaxdim
    @skiponderivable
    def test_double_diff_nd_second_chopped(self, kernel, x_nd, ddtol):
        x = x_nd[:, None]
        if len(x.dtype) < 2:
            pytest.skip(reason='test needs nd >= 2')
        f0, f1 = x.dtype.names[:2]
        r1 = kernel.linop('diff', (2, f0), (2, f1))(x, x.T)
        r2 = kernel.linop('diff', f0, f1).linop('diff', f0, f1)(x, x.T)
        util.assert_allclose(r1, r2, **ddtol)

class Fourier(Base):
    """ Test class for kernels that may implement the fourier series """

    def test_fourier_swap(self, kernel, x_scalar):
        if not kernel.has_transf('fourier'):
            pytest.skip()
        x = x_scalar[:, None]
        k = np.arange(1, 11)[None, :]
        k1 = kernel.linop('fourier', True, None)
        k2 = kernel.linop('fourier', None, True)
        c1 = k1(k, x)
        c2 = k2(x, k)
        util.assert_equal(c1, c2)

    def test_fourier_chained(self, kernel, kw):
        if not kernel.has_transf('fourier'):
            pytest.skip()
        
        if isinstance(kernel, _kernels.Zeta) and kw['nu'] == 0:
            pytest.skip()
        
        f0 = kernel.linop('fourier', True)
        f1 = kernel.linop('fourier', True, None).linop('fourier', None, True)
        f2 = kernel.linop('fourier', None, True).linop('fourier', True, None)

        k = np.arange(100)[:, None]
        c0 = f0(k, k.T)
        c1 = f1(k, k.T)
        c2 = f2(k, k.T)

        util.assert_allclose(c0, c1)
        util.assert_allclose(c0, c2)

    def test_fourier_inference(self, kernel, kw):
        """ test that removing a mode leaves the other in the posterior mean """
        if not kernel.has_transf('fourier'):
            pytest.skip()
        
        if isinstance(kernel, _kernels.Zeta) and kw['nu'] == 0:
            pytest.skip()
        
        x = np.linspace(0, 1, 100)
        gp = (lgp
            .GP(kernel, posepsfac=200)
            .deflinop('F', 'fourier', True, lgp.GP.DefaultProcess)
            .addx(x, 'x')
            .addx(1, 's1', proc='F')
            .addx(2, 'c1', proc='F')
        )
        ms, cs = gp.predfromdata(dict(s1=1, c1=0), 'x', raw=True)
        mc, cc = gp.predfromdata(dict(c1=1, s1=0), 'x', raw=True)
        util.assert_allclose(ms, np.sin(2 * np.pi * x), atol=1e-15)
        util.assert_allclose(mc, np.cos(2 * np.pi * x), atol=1e-15)
        util.assert_allclose(np.diag(cs), cs[0, 0], atol=1e-15)
        util.assert_allclose(np.diag(cc), cc[0, 0], atol=1e-15)

class All(Stationary, Deriv2, Fourier):
    """ Contains all tests, used by default if no test class is defined """
    pass

class TestMatern(Stationary, Deriv2):
    
    @pytest.fixture(params=
        [dict(nu=v + 0.5 ) for v in range(   5)] +
        [dict(nu=v + 0.49) for v in range(   5)] +
        [dict(nu=v + 0.51) for v in range(   5)] +
        [dict(nu=v       ) for v in range(   5)] +
        [dict(nu=v - 0.01) for v in range(1, 5)] +
        [dict(nu=v + 0.01) for v in range(   5)]
    )
    def kw(self, request):
        return request.param

class TestMaternp(Stationary, Deriv2):

    @pytest.fixture(params=list(range(10)))
    def kw(self, request):
        return dict(p=request.param)

    @mark.parametrize('deriv', [0, 1, 2])
    def test_matern_half_integer(self, rng, deriv):
        """
        Check that the formula for half integer nu gives the same result of the
        formula for real nu.
        """
        for p in range(deriv, 10):
            x = 3 * rng.standard_normal((1, 100))
            y = x.T
            k = _kernels.Matern(nu=p + 1/2)
            r1 = k.linop('diff', deriv)(x, y)
            r2 = _kernels.Maternp(p=p).linop('diff', deriv)(x, y)
            util.assert_allclose(r1, r2, rtol=1e-9, atol=1e-16)

class TestWendland(Stationary, Deriv2):

    @pytest.fixture
    def x_nd(self, kw, ranx_nd):
        nd = int(np.floor(2 * kw['alpha'] - 1))
        return ranx_nd(nd)

    @pytest.fixture(params=
        [dict(k=k, alpha=a) for k in range(4) for a in np.linspace(1, 4, 10)])
    def kw(self, request):
        return request.param

    @pytest.fixture
    def psdeps(self):
        return 1e3 * np.finfo(float).eps

    def test_highk(self):
        kernel = self.kercls(k=4)
        with pytest.raises(NotImplementedError):
            kernel(0, 0)

class TestWiener(Base):

    @pytest.fixture
    def ranx_scalar(self, rng):
        return lambda size=100: rng.uniform(0, 10, size)

class TestWienerIntegral(TestWiener, Deriv1):

    def test_wiener_integral(self, rng):
        """
        Test that the derivative of the Wiener integral is the Wiener.
        """
        x, y = np.abs(rng.standard_normal((2, 100)))
        r1 = _kernels.Wiener()(x, y)
        r2 = _kernels.WienerIntegral().linop('diff', 1, 1)(x, y)
        util.assert_allclose(r1, r2)

class TestOrnsteinUhlenbeck(TestWiener): pass

class TestFracBrownian(Deriv2):

    @pytest.fixture
    def ranx_scalar(self, rng):
        return lambda size=100: rng.uniform(-10, 10, size)

    @pytest.fixture(params=[dict(H=H, K=K)
        for H in [0.1, 0.5, 1]
        for K in [0.1, 0.5, 1]])
    def kw(self, request):
        return request.param

class TestBrownianBridge(Base):

    @pytest.fixture
    def ranx_scalar(self, rng):
        return lambda size=100: rng.uniform(0, 1, size)

class TestCategorical(Base):

    N = 10

    @pytest.fixture
    def ranx_scalar(self, rng):
        return lambda size=100: rng.integers(0, self.N, size)

    @pytest.fixture
    def kw(self, rng):
        a = rng.standard_normal((self.N, self.N))
        return dict(cov=a @ a.T)

class TestNNKernel(Deriv2):

    @pytest.fixture
    def psdeps(self):
        return 4 * np.finfo(float).eps

class TestZeta(All):

    @pytest.fixture(params=[0, 0.1, 1, 1.5, 4.9, 1000])
    def kw(self, request):
        return dict(nu=request.param)

class TestCelerite(Stationary, Deriv1):

    @pytest.fixture(params=
        [dict(), dict(gamma=1, B=1), dict(gamma=0, B=0), dict(gamma=10, B=0)])
    def kw(self, request):
        return request.param

class TestHarmonic(Stationary, Deriv1):

    @pytest.fixture(params=[None, 0.01, 0.25, 0.75, 0.99, 1, 1.01, 2])
    def kw(self, request):
        Q = request.param
        return {} if Q is None else dict(Q=Q)

    @mark.parametrize('deriv', [0, 1])
    @mark.parametrize('Q0', [1/2, 1])
    @mark.parametrize('Qderiv', [False, True])
    def test_harmonic_continuous(self, rng, deriv, Q0, Qderiv):
        eps = 1e-10
        Q0 = float(Q0)
        Qs = [(1 - eps) * Q0, Q0, (1 + eps) * Q0]
        x = rng.standard_normal(100)
        results = []
        for Q in Qs:
            kernelf = lambda Q, x: _kernels.Harmonic(Q=Q).linop('diff', deriv, deriv)(x[None, :], x[:, None])
            if Qderiv:
                kernelf = jax.jacfwd(kernelf)
            results.append(kernelf(Q, x))
        util.assert_allclose(results[0], results[2], atol=1e-5)
        util.assert_allclose(results[0], results[1], atol=1e-5)
        util.assert_allclose(results[1], results[2], atol=1e-5)

    def test_celerite_harmonic(self, rng):
        """
        Check that the Celerite kernel is equivalent to the Harmonic kernel when
        B == gamma.
        """
        x = rng.uniform(-1, 1, size=100)
        Q = rng.uniform(1.1, 3)
        eta = np.sqrt(1 - 1 / Q**2)
        B = 1 / (eta * Q)
        r1 = _kernels.Celerite(gamma=B, B=B)(x[:, None], x[None, :])
        r2 = _kernels.Harmonic(Q=Q, scale=eta)(x[:, None], x[None, :])
        util.assert_allclose(r1, r2, atol=1e-15, rtol=1e-15)

class TestBagOfWords(Base):

    lipsum = """

    Duis mollis, est non commodo luctus, nisi erat porttitor ligula, eget lacinia
    odio sem nec elit. Curabitur blandit tempus porttitor. Sed posuere consectetur
    est at lobortis. Cras mattis consectetur purus sit amet fermentum.

    Morbi leo risus, porta ac consectetur ac, vestibulum at eros. Cras mattis
    consectetur purus sit amet fermentum. Lorem ipsum dolor sit amet, consectetur
    adipiscing elit. Donec ullamcorper nulla non metus auctor fringilla.

    Cras justo odio, dapibus ac facilisis in, egestas eget quam. Donec ullamcorper
    nulla non metus auctor fringilla. Praesent commodo cursus magna, vel
    scelerisque nisl consectetur et. Donec id elit non mi porta gravida at eget
    metus. Nullam id dolor id nibh ultricies vehicula ut id elit. Maecenas sed diam
    eget risus varius blandit sit amet non magna.

    Morbi leo risus, porta ac consectetur ac, vestibulum at eros. Sed posuere
    consectetur est at lobortis. Donec ullamcorper nulla non metus auctor
    fringilla. Fusce dapibus, tellus ac cursus commodo, tortor mauris condimentum
    nibh, ut fermentum massa justo sit amet risus. Lorem ipsum dolor sit amet,
    consectetur adipiscing elit. Vivamus sagittis lacus vel augue laoreet rutrum
    faucibus dolor auctor.

    """

    lipsum_words = np.array(list(set(re.split(r'\s|[,.]', lipsum.lower()))))

    @pytest.fixture
    def ranx_scalar(self, rng):
        return lambda size=30: np.array([
            ' '.join(rng.choice(self.lipsum_words, 10))
            for _ in range(np.prod(size, dtype=int))
        ]).reshape(size)

class TestGibbs(Stationary, Deriv2): pass
class TestRescaling(Stationary, Deriv2): pass

class TestGammaExp(Stationary, Deriv2):

    @pytest.fixture(params=[dict(), dict(gamma=2)])
    def kw(self, request):
        return request.param

    @pytest.fixture
    def psdeps(self):
        return 1e3 * np.finfo(float).eps

class TestBessel(Stationary, Deriv2):

    @pytest.fixture
    def psdeps(self):
        return 4 * np.finfo(float).eps

    @pytest.fixture(params=[dict()] +
        [dict(nu=nu) for nu in range(5)] +
        [dict(nu=nu - 0.01) for nu in range(1, 5)] +
        [dict(nu=nu + 0.01) for nu in range(5)] +
        [dict(nu=nu + 0.5) for nu in range(5)])
    def kw(self, request):
        return request.param

class TestCauchy(Stationary, Deriv2):

    @pytest.fixture(params=[
        dict(alpha=a, beta=b)
        for a in [0.001, 0.5, 0.999, 1, 1.001, 1.5, 1.999, 2]
        for b in [0.001, 0.5, 1, 1.5, 2, 4, 8]
    ])
    def kw(self, request):
        return request.param

class TestCausalExpQuad(Stationary, Deriv2):

    @pytest.fixture(params=[0, 1, 2])
    def kw(self, request):
        return dict(alpha=request.param)

    @pytest.fixture
    def ddtol(self):
        return dict(atol=1e-13, rtol=1e-12)

class TestDecaying(Deriv2):

    @pytest.fixture(params=[0, .5, 1, 2])
    def kw(self, request):
        return dict(alpha=request.param)

    @pytest.fixture
    def ranx_scalar(self, rng):
        return lambda size=100: rng.uniform(0, 5, size)

class TestStationaryFracBrownian(Stationary, Deriv2):

    @pytest.fixture
    def psdeps(self):
        return 8 * np.finfo(float).eps

    @pytest.fixture(params=[0.1, 0.5, 1])
    def kw(self, request):
        return dict(H=request.param)

class TestCircular(Stationary, Deriv2):

    @pytest.fixture(params=[dict(c=c, tau=t)
        for c, t in [(0.1, 4), (0.5, 4), (0.5, 8)]])
    def kw(self, request):
        return request.param

class TestMA(Stationary, Base):

    rng = np.random.default_rng([2023, 8, 25, 0, 25])

    @pytest.fixture(params=[
        [], [0], [1], [1, 1], [1, -1], [2, 1], [1, 2, 3, 4, 5],
        rng.standard_normal(30),
    ])
    def kw(self, request):
        return dict(w=request.param)

    @pytest.fixture
    def ranx_scalar(self, rng):
        return lambda size=100: rng.integers(0, 100, size)

class TestAR(Stationary, Base):

    @pytest.fixture(params=[dict(phi=phi, maxlag=100) for phi in [
            [], [0], [-0.5], [0.5], [0.9], [-0.9], [0.5, 0], [0, 0.5], 3 * [0] + [0.5],
        ]] + [dict(gamma=gamma, maxlag=100) for gamma in [
            [0], [1], [1, 0], [1, 0.5], [1, 0.5, 0.25], [1, -0.9],
        ]] + [dict(slnr=r, lnc=c) for r, c in [
            ([], []),
            ([1/10], []),
            ([1/2], []),
            ([1/10, 1/2], []),
            ([1/10, 1/10], []),
            ([1/10, 1/10, -1/2], []),
            ([], [1/10 + 1j]),
            ([], [1/2 + 1j]),
            ([], [1/10 + 1j, 1/2 + 2j]),
            ([], [1/10 + 1j, 1/10 + 2j]),
            ([], [1/10 + 1j, 1/10 + 1j, 1/2 + 2j]),
            ([1/10, 1/10, -1/2], [1/10 + 1j, 1/10 + 1j, 1/2 + 2j]),
        ]])
    def kw(self, request):
        return request.param

    @pytest.fixture
    def ranx_scalar(self, rng):
        return lambda size=100: rng.integers(0, 100, size)

class TestColor(Stationary, Deriv2):

    @pytest.fixture(params=[2, 3, 4, 5, 6, 20])
    def kw(self, request):
        return dict(n=request.param)

class TestBART(Base):

    @pytest.fixture(params=[1, 5])
    def nd(self, request):
        return request.param

    @pytest.fixture
    def x_scalar(self, nd, ranx_scalar):
        if nd != 1:
            pytest.skip(reason='scalar test, nd parameters')
        return ranx_scalar()

    @pytest.fixture
    def x_nd(self, nd, ranx_nd):
        if nd == 1:
            pytest.skip(reason='nd tests, scalar parameters')
        return ranx_nd(nd)

    @pytest.fixture
    def x(self, nd, ranx_scalar, ranx_nd):
        return ranx_scalar() if nd == 1 else ranx_nd(nd)

    @pytest.fixture(params=[
        dict(alpha=a, beta=b, maxd=d, reset=r)
        for a in [0., 1., 0.95]
        for b in [0, 2, 10]
        for d in [0, 1, 2, 3]
        for r in [None, (d + 1) // 2]
    ])
    def kw(self, request, rng, nd):
        splits = rng.standard_normal((10, nd))
        return dict(**request.param, splits=_kernels.BART.splits_from_coord(splits))

class TestSinc(Stationary, Deriv2):

    @pytest.fixture
    def psdeps(self):
        return 10 * np.finfo(float).eps

# Make list of Kernel concrete subclasses.
kernels = {}
for name, obj in vars(_kernels).items():
    if inspect.isclass(obj) and issubclass(obj, _Kernel.Kernel):
        assert obj not in (_Kernel.Kernel, _Kernel.StationaryKernel, _Kernel.IsotropicKernel), obj
        if name.startswith('_'):
            continue
        kernels[name] = obj

# Create default test classes for all kernels without a test already
for name, kernel in kernels.items():
    testname = 'Test' + name
    if testname not in globals():
        globals()[testname] = types.new_class(testname, (All,))

#####################  XFAILS/SKIPS  #####################

util.skip(TestAR, 'test_normalized')
util.skip(TestMA, 'test_normalized')

# TODO These are isotropic kernels with the input='posabs' option. The problems
# arise where x == y. => use make_jaxpr to debug?
util.xfail(TestWendland, 'test_positive_nd_2')
util.xfail(TestWendland, 'test_double_diff_nd_second_chopped')
util.xfail(TestWendland, 'test_continuous_in_zero_2')
util.xfail(TestWendland, 'test_jit_nd_2', reason="seen xpassing, numerical "
    "accuracy problem?")
util.xfail(TestCausalExpQuad, 'test_positive_nd_2')
util.xfail(TestCausalExpQuad, 'test_double_diff_nd_second_chopped', reason=
    "fails on macos, not on linux, with 1e15 variance")
util.xfail(TestCausalExpQuad, 'test_continuous_in_zero_2')
util.xfail(TestCausalExpQuad, 'test_jit_nd_2', reason="it's a divergence of "
    "the variance; mistake in derivability?")
util.xfail(TestCausalExpQuad, 'test_positive_scalar_2', reason="seems numerical"
    " precision problem, but it's not 1 - erf cancel")

util.xfail(TestWendland, 'test_jit_nd_1', reason="the failures are on kw14, "
    "kw33, and involve integer multiples of the identity, with differences: "
    "24 vs 25, 24 vs. 22.")

# TODO These are not isotropic kernels, what is the problem? Those commented are
# currently skipped, the failures were with forcekron applied.
# util.xfail(TestTaylor, 'test_double_diff_nd_second') # numerical precision
# util.xfail(TestFracBrownian, 'test_double_diff_nd_second') # large difference
util.xfail(TestNNKernel, 'test_double_diff_nd_second', reason="large difference")
