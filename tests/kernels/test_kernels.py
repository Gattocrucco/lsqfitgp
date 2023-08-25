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

import inspect
import re
import types

import pytest
from pytest import mark
import numpy as np
from scipy import linalg
import jax

import lsqfitgp as lgp
from lsqfitgp import _kernels, _Kernel

from .. import util

pytestmark = pytest.mark.filterwarnings(
    r'ignore:overriding init argument\(s\)',
    r'ignore:Output seems independent of input',
)

# TODO a test taking first and second derivatives w.r.t. loc and scale, which
# would apply also to non-derivable kernels. Make sure to have distances both 0
# and very near 0. Add a test option for testing a certain list of parameters,
# which defaults to loc and scale.

# TODO systematically test higher dimensions now that maxdim allows it.
    
class KernelTestBase:
    """
    Base class to test kernels. Each subclass tests one specific kernel.
    """
    
    @property
    def kercls(self):
        """ Kernel subclass to test """
        clsname = self.__class__.__name__
        assert clsname.startswith('Test')
        return getattr(_kernels, clsname[4:])
    
    def test_public(self):
        assert self.kercls in vars(lgp).values()

    @pytest.fixture
    def kw(self):
        """ Keyword arguments for the constructor """
        return {}

    @pytest.fixture
    def kernel(self, kw):
        """ A kernel instance """
        return self.kercls(**kw)

    @pytest.fixture
    def ranx_scalar(self, rng):
        """ A callable generating a random vector of scalar input values """
        return lambda size=100: rng.uniform(-5, 5, size=size)

    @pytest.fixture
    def ranx_nd(self, ranx_scalar):
        """ A callable generating a random vector of sized input values """
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
        return ranx_nd(3)

    @pytest.fixture
    def psdeps(self):
        """ Relative tolerance for the smallest eigenvalue to be negative """
        return np.finfo(float).eps
    
    def impl_positive(self, kernel, deriv, x, psdeps):
        if kernel.derivable < deriv:
            pytest.skip()
        if x.dtype.names and kernel.maxdim < len(x.dtype.names):
            pytest.skip()
        d = (deriv, 'f0') if x.dtype.names else deriv
        cov = kernel.transf('diff', d, d)(x[None, :], x[:, None])
        util.assert_allclose(cov, cov.T, rtol=1e-5, atol=1e-7)
        eigv = linalg.eigvalsh(cov)
        assert np.min(eigv) >= -len(cov) * psdeps * np.max(eigv)

    def test_positive_scalar_0(self, kernel, x_scalar, psdeps):
        self.impl_positive(kernel, 0, x_scalar, psdeps)

    def test_positive_scalar_1(self, kernel, x_scalar, psdeps):
        self.impl_positive(kernel, 1, x_scalar, psdeps)

    def test_positive_scalar_2(self, kernel, x_scalar, psdeps):
        self.impl_positive(kernel, 2, x_scalar, psdeps)

    def test_positive_nd_0(self, kernel, x_nd, psdeps):
        self.impl_positive(kernel, 0, x_nd, psdeps)

    def test_positive_nd_1(self, kernel, x_nd, psdeps):
        self.impl_positive(kernel, 1, x_nd, psdeps)

    def test_positive_nd_2(self, kernel, x_nd, psdeps):
        self.impl_positive(kernel, 2, x_nd, psdeps)

    def impl_jit(self, kernel, deriv, x):
        if kernel.derivable < deriv:
            pytest.skip()
        if x.dtype.names:
            x = lgp.StructuredArray(x)
            dtype = np.result_type(*(x.dtype[name].base for name in x.dtype.names))
            if kernel.maxdim < len(x.dtype):
                pytest.skip()
            deriv = deriv, 'f0'
        else:
            dtype = x.dtype
        if not np.issubdtype(dtype, np.number) and not dtype == bool:
            pytest.skip()
        kernel = kernel.transf('diff', deriv, deriv)
        cov1 = kernel(x[None, :], x[:, None])
        cov2 = jax.jit(kernel)(x[None, :], x[:, None])
        util.assert_allclose(cov2, cov1, rtol=1e-6, atol=1e-5)

    def test_jit_scalar_0(self, kernel, x_scalar):
        self.impl_jit(kernel, 0, x_scalar)

    def test_jit_scalar_1(self, kernel, x_scalar):
        self.impl_jit(kernel, 1, x_scalar)

    def test_jit_scalar_2(self, kernel, x_scalar):
        self.impl_jit(kernel, 2, x_scalar)

    def test_jit_nd_0(self, kernel, x_nd):
        self.impl_jit(kernel, 0, x_nd)

    def test_jit_nd_1(self, kernel, x_nd):
        self.impl_jit(kernel, 1, x_nd)

    def test_jit_nd_2(self, kernel, x_nd):
        self.impl_jit(kernel, 2, x_nd)

    # TODO jit with kw as arguments. use static_argnames filtering by dtype.
    
    # TODO test vmap
    
    @pytest.fixture(params=[(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)])
    def derivs(self, request):
        """ Pair of numbers of derivatives to take """
        return request.param
    
    def test_symmetric_offdiagonal(self, kernel, derivs, x_scalar):
        xderiv, yderiv = derivs
        if xderiv == yderiv:
            x = x_scalar[:x_scalar.size // 2, None]
            y = x_scalar[None, x_scalar.size // 2:]
        else:
            x = x_scalar[:, None]
            y = x.T
        if kernel.derivable < max(xderiv, yderiv):
            pytest.skip()
        b1 = kernel.transf('diff', xderiv, yderiv)(x, y)
        b2 = kernel.transf('diff', yderiv, xderiv)(y, x)
        util.assert_allclose(b1, b2, atol=1e-10)

    # TODO test higher derivatives?
    
    def impl_continuous_in_zero(self, kernel, deriv, x_scalar, kw):
        if not isinstance(kernel, _Kernel.StationaryKernel):
            pytest.skip()
        if not np.issubdtype(x_scalar.dtype, np.inexact):
            pytest.skip()
        if kernel.derivable < deriv:
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
        
        kernel = kernel.transf('diff', deriv)
        c0 = kernel(0, 0)
        c1 = kernel(0, 1e-15)
        util.assert_allclose(c1, c0, rtol=1e-10)

    def test_continuous_in_zero_0(self, kernel, x_scalar, kw):
        self.impl_continuous_in_zero(kernel, 0, x_scalar, kw)

    def test_continuous_in_zero_1(self, kernel, x_scalar, kw):
        self.impl_continuous_in_zero(kernel, 1, x_scalar, kw)

    def test_continuous_in_zero_2(self, kernel, x_scalar, kw):
        self.impl_continuous_in_zero(kernel, 2, x_scalar, kw)

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
    
    def test_double_diff_scalar_first(self, kernel, x_scalar):
        if kernel.derivable < 1:
            pytest.skip()
        r1 = kernel.transf('diff', 1, 1)(x_scalar[None, :], x_scalar[:, None])
        r2 = kernel.transf('diff', 1, 0).transf('diff', 0, 1)(x_scalar[None, :], x_scalar[:, None])
        util.assert_allclose(r1, r2)
    
    def test_double_diff_scalar_second(self, kernel, x_scalar):
        if kernel.derivable < 2:
            pytest.skip()
        r1 = kernel.transf('diff', 2, 2)(x_scalar[None, :], x_scalar[:, None])
        r2 = kernel.transf('diff', 1, 1).transf('diff', 1, 1)(x_scalar[None, :], x_scalar[:, None])
        util.assert_allclose(r1, r2, atol=1e-15, rtol=1e-9)
    
    def test_double_diff_scalar_second_chopped(self, kernel, x_scalar):
        if kernel.derivable < 2:
            pytest.skip()
        r1 = kernel.transf('diff', 2, 2)(x_scalar[None, :], x_scalar[:, None])
        r2 = kernel.transf('diff', 2, 0).transf('diff', 0, 2)(x_scalar[None, :], x_scalar[:, None])
        util.assert_allclose(r1, r2)
    
    def test_double_diff_nd_first(self, kernel, ranx_nd):
        if kernel.derivable < 1:
            pytest.skip()
        if kernel.maxdim < 2:
            pytest.skip()
        x = ranx_nd(2)[:, None]
        r1 = kernel.transf('diff', 'f0', 'f0')(x, x.T)
        r2 = kernel.transf('diff', 'f0', 0).transf('diff', 0, 'f0')(x, x.T)
        util.assert_allclose(r1, r2)

    def test_double_diff_nd_second(self, kernel, ranx_nd):
        if kernel.derivable < 2:
            pytest.skip()
        if kernel.maxdim < 2:
            pytest.skip()
        x = ranx_nd(2)[:, None]
        r1 = kernel.transf('diff', (2, 'f0'), (2, 'f1'))(x, x.T)
        r2 = kernel.transf('diff', 'f0', 'f0').transf('diff', 'f1', 'f1')(x, x.T)
        util.assert_allclose(r1, r2)

    def test_double_diff_nd_second_chopped(self, kernel, ranx_nd):
        if kernel.derivable < 2:
            pytest.skip()
        if kernel.maxdim < 2:
            pytest.skip()
        x = ranx_nd(2)[:, None]
        r1 = kernel.transf('diff', (2, 'f0'), (2, 'f1'))(x, x.T)
        r2 = kernel.transf('diff', 'f0', 'f1').transf('diff', 'f0', 'f1')(x, x.T)
        util.assert_allclose(r1, r2, atol=1e-15, rtol=1e-12)
    
    @staticmethod
    def make_x_nd_implicit(x):
        from numpy.lib import recfunctions
        dtype = recfunctions.repack_fields(x.dtype, align=False, recurse=True)
        return x.astype(dtype).view([('', x.dtype[0], (len(x.dtype),))]).copy()
    
    def test_implicit_fields(self, kernel, ranx_nd):
        if kernel.maxdim < 2:
            pytest.skip()
        x1 = ranx_nd(2)[:, None]
        x2 = self.make_x_nd_implicit(x1)
        c1 = kernel(x1, x1.T)
        c2 = kernel(x2, x2.T)
        util.assert_allclose(c1, c2, atol=1e-15, rtol=1e-14)
    
    def test_loc_scale_nd(self, kernel, kw, ranx_nd, x_scalar):
        if kernel.maxdim < 2:
            pytest.skip()
        loc = -2  # < 0
        scale = 3 # > abs(loc)
        # TODO maybe put loc and scale in kw and let random_x adapt the
        # domain to loc and scale
        if not np.issubdtype(x_scalar.dtype, np.inexact):
            pytest.skip()
        x1 = ranx_nd(2)[:, None]
        x2 = self.make_x_nd_implicit(x1)
        x2['f0'] -= loc
        x2['f0'] /= scale
        kernel1 = self.kercls(loc=loc, scale=scale, **kw)
        c1 = kernel1(x1, x1.T)
        c2 = kernel(x2, x2.T)
        util.assert_allclose(c1, c2, rtol=1e-12, atol=1e-13)
    
    def test_weird_derivable(self, kw):
        kw = dict(kw)
        kw.update(derivable=[1])
        with pytest.raises(TypeError):
            self.kercls(**kw)
        kw.update(derivable=1.5)
        with pytest.raises(ValueError):
            self.kercls(**kw)
    
    def test_dim(self, ranx_nd, kw):
        x1 = ranx_nd(2)[:, None]
        x2 = x1['f0']
        x3 = self.make_x_nd_implicit(x1[['f0']])
        kernel1 = self.kercls(dim='f0', **kw)
        if kernel1.maxdim < 2:
            pytest.skip()
        c1 = kernel1(x1, x1.T)
        c2 = self.kercls(**kw)(x2, x2.T)
        c3 = self.kercls(dim='f0', **kw)(x3, x3.T)
        util.assert_equal(c1, c2)
        util.assert_equal(c1, c3)
        with pytest.raises(ValueError):
            self.kercls(dim='f0', **kw)(x2, x2.T)
    
    def test_binary_type_error(self, kernel):
        with pytest.raises(TypeError):
            kernel = kernel + [0]
        with pytest.raises(TypeError):
            kernel = [0] + kernel
        kernel = kernel.transf('rescale', lambda x: 1, None) # make a CrossKernel
        with pytest.raises(TypeError):
            kernel = kernel + [0]
        with pytest.raises(TypeError):
            kernel = [0] + kernel
    
    @mark.parametrize('name', ['rescale', 'xtransf', 'diff', 'loc', 'scale',
        'dim', 'maxdim', 'derivable'])
    def test_transf_noop(self, kernel, name):
        assert kernel is kernel.transf(name, None)
    
    def test_unknown_derivability(self, kw):
        kw = dict(kw)
        kw.update(derivable=None)
        kernel = self.kercls(**kw)
        assert kernel.derivable is None
    
    def test_invalid_input(self, kw):
        if issubclass(self.kercls, _Kernel.StationaryKernel):
            kw = dict(kw)
            kw.update(input='cippa')
            with pytest.raises(KeyError):
                self.kercls(**kw)
        else:
            pytest.skip()
    
    def test_soft_input(self, x_scalar, kw):
        if issubclass(self.kercls, _Kernel.StationaryKernel) and not issubclass(self.kercls, _Kernel.IsotropicKernel):
            x1 = x_scalar
            if not np.issubdtype(x1.dtype, np.inexact):
                pytest.skip()
            x2 = x1 - 1 / np.pi
            if np.any(np.abs(x1 - x2) < 1e-6):
                pytest.xfail()
            kw = dict(kw)
            kw.update(input='signed')
            c1 = self.kercls(**kw)(x1, x2)
            kw.update(input='soft')
            c2 = self.kercls(**kw)(x1, x2)
            util.assert_allclose(c1, c2, atol=1e-14, rtol=1e-14)
    
    def test_where(self, kernel, ranx_nd, x_scalar):
            
        if kernel.maxdim < 2:
            pytest.skip()
        
        x = ranx_nd(2)
        x0 = x['f0'][0]
        cond = lambda x: x < x0
        k1 = _Kernel.where(cond, kernel, 2 * kernel, dim='f0')
        k2 = _Kernel.where(lambda x: cond(x['f0']), kernel, 2 * kernel)
        c1 = k1(x[:, None], x[None, :])
        c2 = k2(x[:, None], x[None, :])
        
        x = self.make_x_nd_implicit(x)
        cond = lambda x: x['f0'][..., 0] < x0
        k1 = _Kernel.where(cond, kernel, 2 * kernel, dim='f0')
        k2 = _Kernel.where(cond, kernel, 2 * kernel)
        c3 = k1(x[:, None], x[None, :])
        c4 = k2(x[:, None], x[None, :])

        util.assert_equal(c1, c2)
        util.assert_equal(c3, c4)
        util.assert_equal(c1, c3)
        
        with pytest.raises(ValueError):
            k1(x_scalar, x_scalar)
    
    def test_rescale_swap(self, kernel, x_scalar):
        x = x_scalar[:x_scalar.size // 2, None]
        y = x_scalar[None, x_scalar.size // 2:]
        x0 = x[0]
        f = lambda x: np.where(x < x0, -1, 1)
        k1 = kernel.transf('rescale', f, None)
        k2 = kernel.transf('rescale', None, f)
        c1 = k1(x, y)
        c2 = k2(y, x)
        util.assert_equal(c1, c2)
    
    def test_fourier_swap(self, kernel, x_scalar):
        if not kernel.has_transf('fourier'):
            pytest.skip()
        x = x_scalar[:, None]
        k = np.arange(1, 11)[None, :]
        k1 = kernel.transf('fourier', True, None)
        k2 = kernel.transf('fourier', None, True)
        c1 = k1(k, x)
        c2 = k2(x, k)
        util.assert_equal(c1, c2)

    def test_xtransf_swap(self, rng, x_scalar, kernel):
        x = x_scalar[:x_scalar.size // 2]
        y = x_scalar[None, x_scalar.size // 2:]
        xt = np.arange(len(x))
        f = lambda xt: x[xt]
        xt = xt[:, None]
        k1 = kernel.transf('xtransf', f, None)
        k2 = kernel.transf('xtransf', None, f)
        c1 = k1(xt, y)
        c2 = k2(y, xt)
        util.assert_equal(c1, c2)
    
    def test_derivable(self, kernel):
        d = kernel.derivable
        if d is None:
            pytest.skip()
        with pytest.raises(RuntimeError):
            kernel.transf('diff', d + 1, d + 1)
        with pytest.raises(RuntimeError):
            kernel.transf('diff', d + 1, None)
        with pytest.raises(RuntimeError):
            kernel.transf('diff', None, d + 1)
    
    def test_diff_errors(self, kw, x_scalar, ranx_nd):
        kw = dict(kw)
        kw.update(derivable=True)
        kernel = self.kercls(**kw)
        
        with pytest.raises(ValueError):
            kernel.transf('diff', 'f0', 'f0')(x_scalar, x_scalar)
        
        x = ranx_nd(2)
        with pytest.raises(ValueError):
            kernel.transf('diff', 'a', 'a')(x, x)
        
        dtype = x.dtype[0]
        if not np.issubdtype(dtype, np.number):
            with pytest.raises(TypeError):
                kernel.transf('diff', 'f0', 'f0')(x, x)
                
    def test_fourier(self, kernel, kw):
        if not kernel.has_transf('fourier'):
            pytest.skip()
        
        if isinstance(kernel, _kernels.Zeta) and kw['nu'] == 0:
            pytest.skip()
        
        x = np.linspace(0, 1, 100)
        gp = (lgp.GP(kernel, posepsfac=200)
            .defkerneltransf('F', 'fourier', True, lgp.GP.DefaultProcess)
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

class TestMatern(KernelTestBase):
    
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

class TestMaternp(KernelTestBase):

    @pytest.fixture(params=list(range(10)))
    def kw(self, request):
        return dict(p=request.param)

class TestWendland(KernelTestBase):

    @pytest.fixture(params=
        [dict(k=k, alpha=a) for k in range(4) for a in np.linspace(1, 4, 10)])
    def kw(self, request):
        return request.param

    @pytest.fixture
    def psdeps(self):
        return 1e3 * np.finfo(float).eps

class TestWiener(KernelTestBase):

    @pytest.fixture
    def ranx_scalar(self, rng):
        return lambda size=100: rng.uniform(0, 10, size)

class TestWienerIntegral(TestWiener): pass

class TestOrnsteinUhlenbeck(TestWiener): pass

class TestFracBrownian(KernelTestBase):

    @pytest.fixture
    def ranx_scalar(self, rng):
        return lambda size=100: rng.uniform(-10, 10, size)

    @pytest.fixture(params=[dict(H=H, K=K)
        for H in [0.1, 0.5, 1]
        for K in [0.1, 0.5, 1]])
    def kw(self, request):
        return request.param

class TestBrownianBridge(KernelTestBase):

    @pytest.fixture
    def ranx_scalar(self, rng):
        return lambda size=100: rng.uniform(0, 1, size)

class TestCategorical(KernelTestBase):

    N = 10

    @pytest.fixture
    def ranx_scalar(self, rng):
        return lambda size=100: rng.integers(0, self.N, size)

    @pytest.fixture
    def kw(self, rng):
        a = rng.standard_normal((self.N, self.N))
        return dict(cov=a @ a.T)

class TestNNKernel(KernelTestBase):

    @pytest.fixture
    def psdeps(self):
        return 4 * np.finfo(float).eps

class TestZeta(KernelTestBase):

    @pytest.fixture(params=[0, 0.1, 1, 1.5, 4.9, 1000])
    def kw(self, request):
        return dict(nu=request.param)

class TestCelerite(KernelTestBase):

    @pytest.fixture(params=
        [dict(), dict(gamma=1, B=1), dict(gamma=0, B=0), dict(gamma=10, B=0)])
    def kw(self, request):
        return request.param

class TestHarmonic(KernelTestBase):

    @pytest.fixture(params=[None, 0.01, 0.25, 0.75, 0.99, 1, 1.01, 2])
    def kw(self, request):
        Q = request.param
        return {} if Q is None else dict(Q=Q)

class TestBagOfWords(KernelTestBase):

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

class TestGibbs(KernelTestBase):

    @pytest.fixture
    def kw(self):
        return dict(derivable=True)

class TestRescaling(KernelTestBase):

    @pytest.fixture
    def kw(self):
        return dict(derivable=True, maxdim=np.inf)

class TestGammaExp(KernelTestBase):

    @pytest.fixture(params=[dict(), dict(gamma=2)])
    def kw(self, request):
        return request.param

    @pytest.fixture
    def psdeps(self):
        return 1e3 * np.finfo(float).eps

class TestBessel(KernelTestBase):

    @pytest.fixture(params=[dict()] +
        [dict(nu=nu) for nu in range(5)] +
        [dict(nu=nu - 0.01) for nu in range(1, 5)] +
        [dict(nu=nu + 0.01) for nu in range(5)] +
        [dict(nu=nu + 0.5) for nu in range(5)])
    def kw(self, request):
        return request.param

class TestCauchy(KernelTestBase):

    @pytest.fixture(params=[
        dict(alpha=a, beta=b)
        for a in [0.001, 0.5, 0.999, 1, 1.001, 1.5, 1.999, 2]
        for b in [0.001, 0.5, 1, 1.5, 2, 4, 8]
    ])
    def kw(self, request):
        return request.param

class TestCausalExpQuad(KernelTestBase):

    @pytest.fixture(params=[0, 1, 2])
    def kw(self, request):
        return dict(alpha=request.param)

class TestDecaying(KernelTestBase):

    @pytest.fixture(params=[0, .5, 1, 2])
    def kw(self, request):
        return dict(alpha=request.param)

    @pytest.fixture
    def ranx_scalar(self, rng):
        return lambda size=100: rng.uniform(0, 5, size)

class TestStationaryFracBrownian(KernelTestBase):

    @pytest.fixture(params=[0.1, 0.5, 1])
    def kw(self, request):
        return dict(H=request.param)

class TestCircular(KernelTestBase):

    @pytest.fixture(params=[dict(c=c, tau=t)
        for c, t in [(0.1, 4), (0.5, 4), (0.5, 8)]])
    def kw(self, request):
        return request.param

class TestMA(KernelTestBase):

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

class TestAR(KernelTestBase):

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

class TestColor(KernelTestBase):

    @pytest.fixture(params=[2, 3, 4, 5, 6, 20])
    def kw(self, request):
        return dict(n=request.param)

class TestBART(KernelTestBase):

    @pytest.fixture(params=[
        dict(alpha=a, beta=b, maxd=d, reset=r)
        for a in [0., 1., 0.95]
        for b in [0, 1, 2, 10]
        for d in [0, 1, 2, 3]
        for r in [None, (d + 1) // 2]
    ])
    def kw(self, request, rng):
        splits = rng.standard_normal((10, 1))
        return dict(**request.param, splits=_kernels.BART.splits_from_coord(splits))

    # TODO I need a way to use nd splits only with nd x

class TestSinc(KernelTestBase):

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
        globals()[testname] = types.new_class(testname, (KernelTestBase,))

def check_matern_half_integer(rng, deriv):
    """
    Check that the formula for half integer nu gives the same result of the
    formula for real nu.
    """
    for p in range(10):
        x = 3 * rng.standard_normal((1, 100))
        y = x.T
        k = _kernels.Matern(nu=p + 1/2)
        d = min(k.derivable, deriv)
        r1 = k.transf('diff', d, d)(x, y)
        r2 = _kernels.Maternp(p=p).transf('diff', d, d)(x, y)
        util.assert_allclose(r1, r2, rtol=1e-9, atol=1e-16)

def test_matern_half_integer_0(rng):
    check_matern_half_integer(rng, 0)

def test_matern_half_integer_1(rng):
    check_matern_half_integer(rng, 1)

def test_matern_half_integer_2(rng):
    check_matern_half_integer(rng, 2)
    
def test_wiener_integral(rng):
    """
    Test that the derivative of the Wiener integral is the Wiener.
    """
    x, y = np.abs(rng.standard_normal((2, 100)))
    r1 = _kernels.Wiener()(x, y)
    r2 = _kernels.WienerIntegral().transf('diff', 1, 1)(x, y)
    util.assert_allclose(r1, r2)

def test_celerite_harmonic(rng):
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

def check_harmonic_continuous(rng, deriv, Q0, Qderiv=False):
    eps = 1e-10
    Q0 = float(Q0)
    Qs = [(1 - eps) * Q0, Q0, (1 + eps) * Q0]
    x = rng.standard_normal(100)
    results = []
    for Q in Qs:
        kernelf = lambda Q, x: _kernels.Harmonic(Q=Q).transf('diff', deriv, deriv)(x[None, :], x[:, None])
        if Qderiv:
            kernelf = jax.jacfwd(kernelf)
        results.append(kernelf(Q, x))
    util.assert_allclose(results[0], results[2], atol=1e-5)
    util.assert_allclose(results[0], results[1], atol=1e-5)
    util.assert_allclose(results[1], results[2], atol=1e-5)

def test_harmonic_continuous_12(rng):
    check_harmonic_continuous(rng, 0, 1/2)
def test_harmonic_continuous_1(rng):
    check_harmonic_continuous(rng, 0, 1)
def test_harmonic_deriv_continuous_12(rng):
    check_harmonic_continuous(rng, 1, 1/2)
def test_harmonic_deriv_continuous_1(rng):
    check_harmonic_continuous(rng, 1, 1)
def test_harmonic_derivQ_continuous_12(rng):
    check_harmonic_continuous(rng, 0, 1/2, True)
def test_harmonic_derivQ_continuous_1(rng):
    check_harmonic_continuous(rng, 0, 1, True)
def test_harmonic_deriv_derivQ_continuous_12(rng):
    check_harmonic_continuous(rng, 1, 1/2, True)
def test_harmonic_deriv_derivQ_continuous_1(rng):
    check_harmonic_continuous(rng, 1, 1, True)

def test_nonfloat_eps():
    x = np.arange(20)
    c1 = _kernels.Wendland()(x, x)
    eps = np.finfo(float).eps
    c2 = np.exp(-eps)
    util.assert_allclose(c1, c2, rtol=eps, atol=eps)

def test_default_override():
    with pytest.warns(UserWarning):
        _kernels.Maternp(p=0, derivable=17)

def test_kernel_decorator_error():
    with pytest.raises(ValueError):
        _Kernel.kernel(1, 2)

def test_maxdim():
    _Kernel.Kernel(lambda x, y: 1, maxdim=None)
    kernel = _Kernel.Kernel(lambda x, y: 1, maxdim=0)
    with pytest.raises(ValueError):
        kernel(0, 0)
    kernel = _Kernel.Kernel(lambda x, y: 1, maxdim=1)
    with pytest.raises(ValueError):
        z = np.zeros((), 'd,d')
        kernel(z, z)

def test_nary():
    kernel = _Kernel.Kernel(lambda x, y: np.maximum(0, 1 - np.abs(x - y)))
    g = lambda x: 2 * x
    k1 = _Kernel.Kernel._nary(lambda f: lambda x: g(x) * f(x), [kernel], kernel._side.LEFT)
    k2 = _Kernel.Kernel._nary(lambda f: lambda y: g(y) * f(y), [kernel], kernel._side.RIGHT)
    x = np.linspace(-5, 5, 30)
    m1 = k1(x[:, None], x[None, :])
    assert not np.allclose(m1, m1.T)
    m2 = k2(x[:, None], x[None, :])
    assert not np.allclose(m2, m2.T)
    assert not np.allclose(m1, m2)
    util.assert_close_matrices(m1, m2.T)

def test_wendland_highk():
    kernel = _kernels.Wendland(k=4)
    with pytest.raises(NotImplementedError):
        kernel(0, 0)

def test_default_transf():
    k1 = _Kernel.Kernel(lambda x, y: x * y, loc=1, maxdim=None)
    k2 = _Kernel.Kernel(lambda x, y: x * y, loc=1)
    x = np.linspace(-5, 3, 10)[:, None]
    v1 = k1(x, x.T)
    v2 = k2(x, x.T)
    util.assert_equal(v1, v2)

#####################  XFAILS/SKIPS  #####################

util.skip(TestAR, 'test_normalized')
util.skip(TestMA, 'test_normalized')

# TODO These are isotropic kernels with the input='soft' option. The problems
# arise where x == y. => use make_jaxpr to debug?
util.xfail(TestWendland, 'test_positive_nd_2')
util.xfail(TestWendland, 'test_double_diff_nd_second_chopped')
util.xfail(TestWendland, 'test_continuous_in_zero_2')
util.xfail(TestWendland, 'test_jit_nd_2') # seen xpassing, precision?
util.xfail(TestCausalExpQuad, 'test_positive_nd_2')
util.xfail(TestCausalExpQuad, 'test_double_diff_nd_second_chopped')
util.xfail(TestCausalExpQuad, 'test_continuous_in_zero_2')
util.xfail(TestCausalExpQuad, 'test_jit_nd_2') # it's a divergence of the variance; mistake in derivability?

# TODO some xpass, likely numerical precision problems
util.xfail(TestWendland, 'test_positive_scalar_2') # normally xpasses
util.xfail(TestCausalExpQuad, 'test_positive_scalar_2') # NOT 1 - erf cancel

# TODO This one should not fail, it's a first derivative! Probably it's the
# case D = 1 that fails because that's the maximum dimensionality. For some
# reason I don't catch it without taking a derivative. => This explanation is
# likely wrong since the jit test fails too, without checking positivity.
util.xfail(TestWendland, 'test_positive_nd_1') # seen xpassing in the wild
util.xfail(TestWendland, 'test_jit_nd_1') # seen xpassing, precision?

# TODO These are not isotropic kernels, what is the problem?
util.xfail(TestTaylor, 'test_double_diff_nd_second') # numerical precision
util.xfail(TestNNKernel, 'test_double_diff_nd_second') # large difference
util.xfail(TestFracBrownian, 'test_double_diff_nd_second') # large difference
