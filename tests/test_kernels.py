# lsqfitgp/tests/test_kernels.py
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

import sys
import inspect
import abc
import re

import pytest
import numpy as np
from scipy import linalg
from jax import test_util
import jax

sys.path = ['.'] + sys.path
import lsqfitgp as lgp
from lsqfitgp import _kernels, _Kernel

import util

# Make list of Kernel concrete subclasses.
kernels = []
for name, obj in vars(_kernels).items():
    if inspect.isclass(obj) and issubclass(obj, _Kernel.Kernel):
        assert obj not in (_Kernel.Kernel, _Kernel.StationaryKernel, _Kernel.IsotropicKernel), obj
        if name.startswith('_'):
            continue
        kernels.append(obj)

pytestmark = pytest.mark.filterwarnings(
    r'ignore:overriding init argument\(s\)',
    r'ignore:Output seems independent of input',
)

class KernelTestBase(metaclass=abc.ABCMeta):
    """
    Abstract base class to test kernels. Each subclass tests one specific
    kernel.
    """
    
    @property
    @abc.abstractmethod
    def kernel_class(self):
        pass
    
    @property
    def kwargs_list(self):
        return [dict()]
    
    def random_x(self, **kw):
        return np.random.uniform(-5, 5, size=100)
    
    def random_x_nd(self, ndim, **kw):
        xs = [self.random_x(**kw) for _ in range(ndim)]
        x = np.empty(len(xs[0]), dtype=ndim * [('', xs[0].dtype)])
        for i in range(ndim):
            x[x.dtype.names[i]] = xs[i]
        return x
    
    @staticmethod
    def make_x_nd_implicit(x):
        n = len(x.dtype.names)
        dtype = x.dtype.fields[x.dtype.names[0]][0]
        newx = np.empty(x.shape, [('f0', dtype, (n,))])
        for i, name in enumerate(x.dtype.names):
            newx['f0'][..., i] = x[name]
        return newx
    
    @property
    def eps(self):
        return 200 * np.finfo(float).eps
    
    def positive(self, deriv, nd=False):
        donesomething = False
        for kw in self.kwargs_list:
            x = self.random_x_nd(2, **kw) if nd else self.random_x(**kw)
            kernel = self.kernel_class(**kw)
            if kernel.derivable < deriv:
                continue
            d = (deriv, 'f0') if nd else deriv
            cov = kernel.diff(d, d)(x[None, :], x[:, None])
            np.testing.assert_allclose(cov, cov.T, rtol=1e-5, atol=1e-7)
            eigv = linalg.eigvalsh(cov)
            assert np.min(eigv) >= -len(cov) * self.eps * np.max(eigv)
            donesomething = True
        if not donesomething:
            pytest.skip()
    
    def test_positive(self):
        self.positive(0)
    
    def test_positive_deriv(self):
        self.positive(1)
    
    def test_positive_deriv2(self):
        self.positive(2)
    
    def test_positive_nd(self):
        self.positive(0, True)
    
    def test_positive_deriv_nd(self):
        self.positive(1, True)
    
    def test_positive_deriv2_nd(self):
        self.positive(2, True)
    
    def symmetric_offdiagonal(self, xderiv, yderiv):
        donesomething = False
        for kw in self.kwargs_list:
            x = self.random_x(**kw)[None, :]
            if xderiv == yderiv:
                y = self.random_x(**kw)[:, None]
            else:
                y = x.T
            kernel = self.kernel_class(**kw)
            if kernel.derivable < max(xderiv, yderiv):
                continue
            b1 = kernel.diff(xderiv, yderiv)(x, y)
            b2 = kernel.diff(yderiv, xderiv)(y, x)
            assert np.allclose(b1, b2)
            donesomething = True
        if not donesomething:
            pytest.skip()

    def test_symmetric_00(self):
        self.symmetric_offdiagonal(0, 0)
    
    def test_symmetric_10(self):
        self.symmetric_offdiagonal(1, 0)
    
    def test_symmetric_11(self):
        self.symmetric_offdiagonal(1, 1)
    
    def test_symmetric_20(self):
        self.symmetric_offdiagonal(2, 0)
    
    def test_symmetric_21(self):
        self.symmetric_offdiagonal(2, 1)
    
    def test_symmetric_22(self):
        self.symmetric_offdiagonal(2, 2)
    
    # def test_symmetric_33(self):
    #     self.symmetric_offdiagonal(3, 3)
    
    def jit(self, deriv=0, nd=False):
        donesomething = False
        for kw in self.kwargs_list:
            if nd:
                x = self.random_x_nd(2, **kw)
                x = lgp.StructuredArray(x)
                dtype = np.result_type(*(x[name].dtype for name in x.dtype.names))
            else:
                x = self.random_x(**kw)
                dtype = x.dtype
            if not np.issubdtype(dtype, np.number) and not dtype == bool:
                continue
            kernel = self.kernel_class(**kw)
            if kernel.derivable < deriv:
                continue
            d = (deriv, 'f0') if nd else deriv
            kernel = kernel.diff(d, d)
            cov1 = kernel(x[None, :], x[:, None])
            cov2 = jax.jit(kernel)(x[None, :], x[:, None])
            np.testing.assert_allclose(cov2, cov1, rtol=1e-6, atol=1e-5)
            donesomething = True
        if not donesomething:
            pytest.skip()
    
    def test_jit(self):
        self.jit()
    
    def test_jit_deriv(self):
        self.jit(1)
    
    def test_jit_deriv2(self):
        self.jit(2)
    
    def test_jit_nd(self):
        self.jit(0, True)
    
    def test_jit_deriv_nd(self):
        self.jit(1, True)
    
    def test_jit_deriv2_nd(self):
        self.jit(2, True)

    def test_normalized(self):
        kernel = self.kernel_class
        if issubclass(kernel, _Kernel.StationaryKernel):
            for kw in self.kwargs_list:
                x = self.random_x(**kw)
                var = kernel(**kw)(x, x)
                np.testing.assert_allclose(var, 1)
        else:
            pytest.skip()
    
    def test_double_diff_scalar_first(self):
        donesomething = False
        for kw in self.kwargs_list:
            kernel = self.kernel_class(**kw)
            if kernel.derivable < 1:
                continue
            x = self.random_x(**kw)
            r1 = kernel.diff(1, 1)(x[None, :], x[:, None])
            r2 = kernel.diff(1, 0).diff(0, 1)(x[None, :], x[:, None])
            assert np.allclose(r1, r2)
            donesomething = True
        if not donesomething:
            pytest.skip()
    
    def test_double_diff_scalar_second(self):
        donesomething = False
        for kw in self.kwargs_list:
            kernel = self.kernel_class(**kw)
            if kernel.derivable < 2:
                continue
            x = self.random_x(**kw)
            r1 = kernel.diff(2, 2)(x[None, :], x[:, None])
            r2 = kernel.diff(1, 1).diff(1, 1)(x[None, :], x[:, None])
            assert np.allclose(r1, r2)
            donesomething = True
        if not donesomething:
            pytest.skip()
    
    def test_double_diff_scalar_second_chopped(self):
        donesomething = False
        for kw in self.kwargs_list:
            kernel = self.kernel_class(**kw)
            if kernel.derivable < 2:
                continue
            x = self.random_x(**kw)
            r1 = kernel.diff(2, 2)(x[None, :], x[:, None])
            r2 = kernel.diff(2, 0).diff(0, 2)(x[None, :], x[:, None])
            assert np.allclose(r1, r2)
            donesomething = True
        if not donesomething:
            pytest.skip()
    
    def test_double_diff_nd_first(self):
        donesomething = False
        for kw in self.kwargs_list:
            kernel = self.kernel_class(**kw)
            if kernel.derivable < 1:
                continue
            x = self.random_x_nd(2, **kw)[None, :]
            r1 = kernel.diff('f0', 'f0')(x, x.T)
            r2 = kernel.diff('f0', 0).diff(0, 'f0')(x, x.T)
            assert np.allclose(r1, r2)
            donesomething = True
        if not donesomething:
            pytest.skip()

    def test_double_diff_nd_second(self):
        donesomething = False
        for kw in self.kwargs_list:
            kernel = self.kernel_class(**kw)
            if kernel.derivable < 2:
                continue
            x = self.random_x_nd(2, **kw)[None, :]
            r1 = kernel.diff((2, 'f0'), (2, 'f1'))(x, x.T)
            r2 = kernel.diff('f0', 'f0').diff('f1', 'f1')(x, x.T)
            assert np.allclose(r1, r2)
            donesomething = True
        if not donesomething:
            pytest.skip()

    def test_double_diff_nd_second_chopped(self):
        donesomething = False
        for kw in self.kwargs_list:
            kernel = self.kernel_class(**kw)
            if kernel.derivable < 2:
                continue
            x = self.random_x_nd(2, **kw)[None, :]
            r1 = kernel.diff((2, 'f0'), (2, 'f1'))(x, x.T)
            r2 = kernel.diff('f0', 'f1').diff('f0', 'f1')(x, x.T)
            np.testing.assert_allclose(r1, r2)
            donesomething = True
        if not donesomething:
            pytest.skip()
    
    def test_implicit_fields(self):
        for kw in self.kwargs_list:
            x1 = self.random_x_nd(3, **kw)[:, None]
            x2 = self.make_x_nd_implicit(x1)
            covfun = self.kernel_class(**kw)
            c1 = covfun(x1, x1.T)
            c2 = covfun(x2, x2.T)
            np.testing.assert_allclose(c1, c2, atol=1e-15, rtol=1e-15)
    
    def test_loc_scale_nd(self):
        kernel = self.kernel_class
        skip = [_kernels.BagOfWords, _kernels.Categorical]
        if kernel in skip:
            pytest.skip()
        loc = -2  # < 0
        scale = 3 # > abs(loc)
        for kw in self.kwargs_list:
            # TODO maybe put loc and scale in kw and let random_x adapt the
            # domain to loc and scale
            x1 = self.random_x_nd(3, **kw)[:, None]
            x2 = self.make_x_nd_implicit(x1)
            x2['f0'] -= loc
            x2['f0'] /= scale
            c1 = kernel(loc=loc, scale=scale, **kw)(x1, x1.T)
            c2 = kernel(**kw)(x2, x2.T)
            np.testing.assert_allclose(c1, c2, rtol=1e-12, atol=1e-13)
    
    def test_weird_derivable(self):
        for kw in self.kwargs_list:
            kw = dict(kw)
            kw.update(derivable=[1])
            covfun = self.kernel_class(**kw)
            assert covfun.derivable == sys.maxsize
            kw = dict(kw)
            kw.update(derivable=[])
            covfun = self.kernel_class(**kw)
            assert covfun.derivable == 0
    
    def test_dim(self):
        for kw in self.kwargs_list:
            x1 = self.random_x_nd(3, **kw)[:, None]
            x2 = x1['f0']
            # dtype = x1.dtype.fields['f0'][0]
            # x3 = np.empty(x1.shape, [('f0', dtype, (1,))])
            # x3['f0'] = x2[..., None]
            x3 = self.make_x_nd_implicit(x1[['f0']])
            c1 = self.kernel_class(dim='f0', **kw)(x1, x1.T)
            c2 = self.kernel_class(**kw)(x2, x2.T)
            c3 = self.kernel_class(dim='f0', **kw)(x3, x3.T)
            util.assert_equal(c1, c2)
            util.assert_equal(c1, c3)
            with pytest.raises(ValueError):
                self.kernel_class(dim='f0', **kw)(x2, x2.T)
    
    def test_binary_type_error(self):
        for kw in self.kwargs_list:
            kernel = self.kernel_class(**kw)
            with pytest.raises(TypeError):
                kernel = kernel + [0]
            with pytest.raises(TypeError):
                kernel = [0] + kernel
            kernel = kernel.rescale(lambda x: 1, None) # make a _CrossKernel
            with pytest.raises(TypeError):
                kernel = kernel + [0]
            with pytest.raises(TypeError):
                kernel = [0] + kernel
    
    def test_pow(self):
        for kw in self.kwargs_list:
            kernel = self.kernel_class(**kw)
            with pytest.raises(TypeError):
                kernel = kernel ** kernel
            k1 = kernel ** 2
            k2 = kernel * kernel
            x = self.random_x(**kw)[:, None]
            c1 = k1(x, x.T)
            c2 = k2(x, x.T)
            util.assert_equal(c1, c2)
        
    def test_transf_noop(self):
        meths = ['rescale', 'xtransf', 'diff', 'fourier', 'taylor']
        for kw in self.kwargs_list:
            kernel = self.kernel_class(**kw)
            for meth in meths:
                assert kernel is getattr(kernel, meth)(None, None)
    
    def test_unknown_derivability(self):
        for kw in self.kwargs_list:
            kw = dict(kw)
            kw.update(derivable=None)
            kernel = self.kernel_class(**kw)
            assert kernel.derivable is None
    
    def test_invalid_input(self):
        kernel = self.kernel_class
        if issubclass(kernel, _Kernel.StationaryKernel):
            for kw in self.kwargs_list:
                kw = dict(kw)
                kw.update(input='cippa')
                with pytest.raises(KeyError):
                    kernel(**kw)
        else:
            pytest.skip()
    
    def test_soft_input(self):
        kernel = self.kernel_class
        if issubclass(kernel, _Kernel.StationaryKernel) and not issubclass(kernel, _Kernel.IsotropicKernel):
            for kw in self.kwargs_list:
                x1 = self.random_x(**kw)
                x2 = x1 - 1 / np.pi
                if np.any(np.abs(x1 - x2) < 1e-6):
                    pytest.xfail()
                kw = dict(kw)
                kw.update(input='signed')
                c1 = kernel(**kw)(x1, x2)
                kw = dict(kw)
                kw.update(input='soft')
                c2 = kernel(**kw)(x1, x2)
                np.testing.assert_allclose(c1, c2, atol=1e-14, rtol=1e-14)
        else:
            pytest.skip()
    
    def test_where(self):
        for kw in self.kwargs_list:
            kernel = self.kernel_class(**kw)
            
            x = self.random_x_nd(2, **kw)
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
            
            x = self.random_x(**kw)
            with pytest.raises(ValueError):
                k1(x, x)
    
    def test_rescale_swap(self):
        for kw in self.kwargs_list:
            kernel = self.kernel_class(**kw)
            x = self.random_x(**kw)[:, None]
            y = self.random_x(**kw)[None, :]
            x0 = x[0]
            f = lambda x: np.where(x < x0, -1, 1)
            k1 = kernel.rescale(f, None)
            k2 = kernel.rescale(None, f)
            c1 = k1(x, y)
            c2 = k2(y, x)
            util.assert_equal(c1, c2)
    
    def test_fourier_swap(self):
        for kw in self.kwargs_list:
            kernel = self.kernel_class(**kw)
            try:
                kernel.fourier(True, None)
            except NotImplementedError:
                pytest.skip()
            x = self.random_x(**kw)[:, None]
            k = np.arange(1, 11)[None, :]
            k1 = kernel.fourier(True, None)
            k2 = kernel.fourier(None, True)
            c1 = k1(k, x)
            c2 = k2(x, k)
            util.assert_equal(c1, c2)

    def test_xtransf_swap(self):
        for kw in self.kwargs_list:
            kernel = self.kernel_class(**kw)
            x = self.random_x(**kw)
            y = self.random_x(**kw)[None, :]
            xt = np.arange(len(x))
            f = lambda xt: x[xt]
            xt = xt[:, None]
            k1 = kernel.xtransf(f, None)
            k2 = kernel.xtransf(None, f)
            c1 = k1(xt, y)
            c2 = k2(y, xt)
            util.assert_equal(c1, c2)
    
    def test_derivable(self):
        for kw in self.kwargs_list:
            kernel = self.kernel_class(**kw)
            d = kernel.derivable
            with pytest.raises(RuntimeError):
                kernel.diff(d + 1, d + 1)
            with pytest.raises(RuntimeError):
                kernel.diff(d + 1, None)
            with pytest.raises(RuntimeError):
                kernel.diff(None, d + 1)
            if d == 0:
                continue
            d = min(d, 2)
            kernel *= _kernels.White()
            with pytest.warns(UserWarning):
              kernel.diff(1, 1)
            with pytest.warns(UserWarning):
                kernel.diff(None, d)
            with pytest.warns(UserWarning):
                kernel.diff(d, None)
    
    def test_diff_errors(self):
        donesomething = False
        for kw in self.kwargs_list:
            kw = dict(kw)
            kw.update(derivable=True)
            kernel = self.kernel_class(**kw)
            
            x = self.random_x(**kw)
            with pytest.raises(ValueError):
                kernel.diff('f0', 'f0')(x, x)
            
            x = self.random_x_nd(2, **kw)
            with pytest.raises(ValueError):
                kernel.diff('a', 'a')(x, x)
            
            dtype = x.dtype.fields['f0'][0]
            if not np.issubdtype(dtype, np.number):
                with pytest.raises(TypeError):
                    kernel.diff('f0', 'f0')(x, x)
            
            donesomething = True
        if not donesomething:
            pytest.skip()
    
    def test_fourier(self):
        for kw in self.kwargs_list:
            kernel = self.kernel_class(**kw)
            try:
                kernel.fourier(True, True)
            except NotImplementedError:
                pytest.skip()
            x = np.linspace(0, 1, 100)
            gp = lgp.GP(kernel, posepsfac=200)
            gp.addkernelop('fourier', True, 'F')
            gp.addx(x, 'x')
            gp.addx(1, 's1', proc='F')
            gp.addx(2, 'c1', proc='F')
            ms, cs = gp.predfromdata(dict(s1=1, c1=0), 'x', raw=True)
            mc, cc = gp.predfromdata(dict(c1=1, s1=0), 'x', raw=True)
            np.testing.assert_allclose(ms, np.sin(2 * np.pi * x))
            np.testing.assert_allclose(mc, np.cos(2 * np.pi * x))
            np.testing.assert_allclose(np.diag(cs), cs[0, 0])
            np.testing.assert_allclose(np.diag(cc), cc[0, 0])

    @classmethod
    def make_subclass(cls, kernel_class, kwargs_list=None, random_x_fun=None, eps=None):
        name = 'Test' + kernel_class.__name__
        subclass = type(cls)(name, (cls,), {
            'kernel_class': property(lambda self: kernel_class)
        })
        if kwargs_list is not None:
            subclass.kwargs_list = property(lambda self: kwargs_list)
        if random_x_fun is not None:
            subclass.random_x = lambda self, **kw: random_x_fun(**kw)
        if eps is not None:
            subclass.eps = property(lambda self: eps)
        return subclass

def matrix_square(A):
    return A.T @ A

def random_nd(size, ndim):
    out = np.empty(size, dtype=[('xyz', float, (ndim,))])
    out['xyz'] = np.random.uniform(-5, 5, size=(size, ndim))
    return out

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

def bow_rand(**kw):
    return np.array([' '.join(np.random.choice(lipsum_words, 10)) for _ in range(30)])
        
# Define a concrete subclass of KernelTestBase for each kernel.
test_kwargs = {
    _kernels.Matern: dict(kwargs_list=[
        dict(nu=0.5), dict(nu=0.6), dict(nu=1.5), dict(nu=2.5)
    ]),
    _kernels.PPKernel: dict(kwargs_list=[
        dict(q=q, D=D) for q in range(4) for D in range(1, 6)
    ]),
    _kernels.Wiener: dict(random_x_fun=lambda **kw: np.random.uniform(0, 10, size=100)),
    _kernels.WienerIntegral: dict(random_x_fun=lambda **kw: np.random.uniform(0, 10, size=100)),
    _kernels.FracBrownian: dict(random_x_fun=lambda **kw: np.random.uniform(0, 10, size=100)),
    _kernels.BrownianBridge: dict(random_x_fun=lambda **kw: np.random.uniform(0, 1, size=100)),
    _kernels.OrnsteinUhlenbeck: dict(random_x_fun=lambda **kw: np.random.uniform(0, 10, size=100)),
    _kernels.Categorical: dict(kwargs_list=[
        dict(cov=matrix_square(np.random.randn(10, 10)))
    ], random_x_fun=lambda **kw: np.random.randint(10, size=100)),
    _kernels.NNKernel: dict(eps=4 * np.finfo(float).eps),
    _kernels.Fourier: dict(kwargs_list=[
        dict(n=n) for n in range(1, 5)
    ], eps=2048 * np.finfo(float).eps),
    _kernels.Celerite: dict(kwargs_list=[
        dict(), dict(gamma=1, B=1), dict(gamma=0, B=0), dict(gamma=10, B=0)
    ]),
    _kernels.Harmonic: dict(kwargs_list=[
        dict(), dict(Q=0.01), dict(Q=0.25), dict(Q=0.75), dict(Q=0.99), dict(Q=1), dict(Q=1.01), dict(Q=2)
    ]),
    _kernels.BagOfWords: dict(random_x_fun=bow_rand),
    _kernels.Gibbs: dict(kwargs_list=[dict(derivable=True)]),
    _kernels.Rescaling: dict(kwargs_list=[dict(derivable=True)]),
    _kernels.GammaExp: dict(kwargs_list=[dict(), dict(gamma=2)]),
}

for kernel in kernels:
    factory_kw = test_kwargs.get(kernel, {})
    newclass = KernelTestBase.make_subclass(kernel, **factory_kw)
    exec('{} = newclass'.format(newclass.__name__))

TestGammaExp.eps = property(lambda self: 1e3 * np.finfo(float).eps)
TestPPKernel.eps = property(lambda self: 1e3 * np.finfo(float).eps)

def test_matern_half_integer():
    """
    Check that the formula for half integer nu gives the same result of the
    formula for real nu.
    """
    for p in range(10):
        nu = p + 1/2
        assert nu - 1/2 == p
        nualt = nu * (1 + 4 * np.finfo(float).eps)
        assert nualt > nu
        x, y = 3 * np.random.randn(2, 100)
        r1 = _kernels.Matern(nu=nu)(x, y)
        r2 = _kernels.Matern(nu=nualt)(x, y)
        assert np.allclose(r1, r2)

def check_matern_spec_p(p, xderiv, yderiv):
    """
    Test implementations of specific cases of nu.
    """
    nu = p + 1/2
    spec = eval('_kernels.Matern{}2'.format(1 + 2 * p))
    if spec().derivable >= max(xderiv, yderiv):
        x = np.random.randn(10)[:, None]
        r1 = _kernels.Matern(nu=nu).diff(xderiv, yderiv)(x, x.T)
        r2 = spec().diff(xderiv, yderiv)(x, x.T)
        assert np.allclose(r1, r2)

def check_matern_spec(xderiv, yderiv):
    for p in range(3):
        check_matern_spec_p(p, xderiv, yderiv)

def test_matern_spec_00():
    check_matern_spec(0, 0)
def test_matern_spec_10():
    check_matern_spec(1, 0)
def test_matern_spec_11():
    check_matern_spec(1, 1)
def test_matern_spec_20():
    check_matern_spec(2, 0)
def test_matern_spec_21():
    check_matern_spec(2, 1)
def test_matern_spec_22():
    check_matern_spec(2, 2)

def test_wiener_integral():
    """
    Test that the derivative of the Wiener integral is the Wiener.
    """
    x, y = np.abs(np.random.randn(2, 100))
    r1 = _kernels.Wiener()(x, y)
    r2 = _kernels.WienerIntegral().diff(1, 1)(x, y)
    assert np.allclose(r1, r2)

def bernoulli_poly_handwritten(n, x):
    return [
        lambda x: 1,
        lambda x: x - 1/2,
        lambda x: x**2 - x + 1/6,
        lambda x: x**3 - 3/2 * x**2 + 1/2 * x,
        lambda x: x**4 - 2 * x**3 + x**2 - 1/30,
        lambda x: x**5 - 5/2 * x**4 + 5/3 * x**3 - 1/6 * x,
        lambda x: x**6 - 3 * x**5 + 5/2 * x**4 - 1/2 * x**2 + 1/42
    ][n](x)

def check_bernoulli(n, x):
    r1 = bernoulli_poly_handwritten(n, x)
    r2 = _kernels._bernoulli_poly(n, x)
    assert np.allclose(r1, r2)

def test_bernoulli():
    for n in range(7):
        x = np.random.uniform(0, 1, size=100)
        check_bernoulli(n, x)

def test_celerite_harmonic():
    """
    Check that the Celerite kernel is equivalent to the Harmonic kernel when
    B == gamma.
    """
    x = np.random.uniform(-1, 1, size=100)
    Q = np.random.uniform(1.1, 3)
    eta = np.sqrt(1 - 1 / Q**2)
    B = 1 / (eta * Q)
    r1 = _kernels.Celerite(gamma=B, B=B)(x[:, None], x[None, :])
    r2 = _kernels.Harmonic(Q=Q, scale=eta)(x[:, None], x[None, :])
    assert np.allclose(r1, r2)

def check_harmonic_continuous(deriv, Q0, Qderiv=False):
    eps = 1e-10
    Q0 = float(Q0)
    Qs = [(1 - eps) * Q0, Q0, (1 + eps) * Q0]
    x = np.random.randn(100)
    results = []
    for Q in Qs:
        kernelf = lambda Q, x: _kernels.Harmonic(Q=Q).diff(deriv, deriv)(x[None, :], x[:, None])
        if Qderiv:
            kernelf = jax.jacfwd(kernelf)
        results.append(kernelf(Q, x))
    np.testing.assert_allclose(results[0], results[2], atol=1e-5)
    np.testing.assert_allclose(results[0], results[1], atol=1e-5)
    np.testing.assert_allclose(results[1], results[2], atol=1e-5)

def test_harmonic_continuous_12():
    check_harmonic_continuous(0, 1/2)
def test_harmonic_continuous_1():
    check_harmonic_continuous(0, 1)
def test_harmonic_deriv_continuous_12():
    check_harmonic_continuous(1, 1/2)
def test_harmonic_deriv_continuous_1():
    check_harmonic_continuous(1, 1)
def test_harmonic_derivQ_continuous_12():
    check_harmonic_continuous(0, 1/2, True)
def test_harmonic_derivQ_continuous_1():
    check_harmonic_continuous(0, 1, True)
def test_harmonic_deriv_derivQ_continuous_12():
    check_harmonic_continuous(1, 1/2, True)
def test_harmonic_deriv_derivQ_continuous_1():
    check_harmonic_continuous(1, 1, True)

def test_nonfloat_eps():
    x = np.arange(20)
    c1 = _kernels.Matern12()(x, x)
    c2 = np.exp(-np.finfo(float).eps)
    util.assert_equal(c1, c2)

def test_default_override():
    with pytest.warns(UserWarning):
        _kernels.Matern12(derivable=17)

def test_kernel_decorator_error():
    with pytest.raises(ValueError):
        _Kernel.kernel(1, 2)

def test_transf_not_implemented():
    meths = ['fourier', 'taylor']
    kernel = _kernels.Matern12()
    for meth in meths:
        with pytest.raises(NotImplementedError):
            getattr(kernel, meth)(True, None)
        with pytest.raises(NotImplementedError):
            getattr(kernel, meth)(None, True)
        with pytest.raises(NotImplementedError):
            getattr(kernel, meth)(True, True)

def test_matern_derivatives():
    for p in range(10):
        x = np.linspace(0, 10, 100)
        test_util.check_grads(lambda x: _kernels._maternp(x, p), (x,), 2)

#####################  XFAILS  #####################

import functools

def xfail(cls, meth):
    impl = getattr(cls, meth)
    @pytest.mark.xfail
    @functools.wraps(impl) # `wraps` needed because pytest uses the method name
    def newimpl(self):
        # wrap because otherwise the superclass method would be marked too
        impl(self)
    setattr(cls, meth, newimpl)

# TODO These are isotropic kernels with the input='soft' option. The problems
# arise where x == y. => After porting JAX, I can use make_jaxpr to debug.
xfail(TestMatern, 'test_positive_deriv2_nd')
xfail(TestMatern, 'test_double_diff_nd_second_chopped')
xfail(TestMatern52, 'test_positive_deriv2_nd')
xfail(TestMatern52, 'test_double_diff_nd_second_chopped')
xfail(TestMatern52, 'test_jit_deriv2_nd')
xfail(TestPPKernel, 'test_positive_deriv2_nd')
xfail(TestPPKernel, 'test_double_diff_nd_second_chopped')
xfail(TestPPKernel, 'test_jit_deriv2_nd')
xfail(TestGammaExp, 'test_positive_deriv2_nd')
xfail(TestGammaExp, 'test_double_diff_nd_second_chopped')

# TODO often xpass, numerical precision problems
xfail(TestGammaExp, 'test_positive_deriv2')
xfail(TestPPKernel, 'test_positive_deriv2')

# TODO This one should not fail, it's a first derivative! Probably it's the
# case D = 1 that fails because that's the maximum dimensionality. For some
# reason I don't catch it without taking a derivative. => This explanation is
# likely wrong since the jit test fails too.
xfail(TestPPKernel, 'test_positive_deriv_nd') # seen xpassing in the wild
xfail(TestPPKernel, 'test_jit_deriv_nd')

# TODO These are not isotropic kernels, what is the problem?
xfail(TestTaylor, 'test_double_diff_nd_second')
xfail(TestNNKernel, 'test_double_diff_nd_second')

# TODO functions not supported by XLA. Wait for jax to add them?
xfail(TestMatern, 'test_jit')
xfail(TestMatern, 'test_jit_nd')
xfail(TestTaylor, 'test_jit')
xfail(TestTaylor, 'test_jit_deriv')
xfail(TestTaylor, 'test_jit_deriv2')
xfail(TestTaylor, 'test_jit_nd')
xfail(TestTaylor, 'test_jit_deriv_nd')
xfail(TestTaylor, 'test_jit_deriv2_nd')
