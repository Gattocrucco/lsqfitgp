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
import autograd
from scipy import linalg

sys.path = ['.'] + sys.path
from lsqfitgp import _kernels, _Kernel

# Make list of Kernel concrete subclasses.
kernels = []
for obj in _kernels.__dict__.values():
    if inspect.isclass(obj) and issubclass(obj, _Kernel.Kernel):
        if obj is _Kernel.Kernel or obj is _Kernel.IsotropicKernel:
            continue
        kernels.append(obj)

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
    
    @property
    def eps(self):
        return 100 * np.finfo(float).eps
    
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
    
    def test_normalized(self):
        stationary = [
            _kernels.Cos,
            _kernels.Fourier,
            _kernels.Periodic,
            _kernels.Celerite,
            _kernels.Harmonic
        ]
        kernel = self.kernel_class
        if kernel in stationary or issubclass(kernel, _Kernel.IsotropicKernel):
            for kw in self.kwargs_list:
                x = self.random_x(**kw)
                var = kernel(**kw)(x, x)
                assert np.allclose(var, 1)
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
            assert np.allclose(r1, r2)
            donesomething = True
        if not donesomething:
            pytest.skip()

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
    _kernels.Rescaling: dict(kwargs_list=[dict(derivable=True)])
}
for kernel in kernels:
    factory_kw = test_kwargs.get(kernel, {})
    newclass = KernelTestBase.make_subclass(kernel, **factory_kw)
    exec('{} = newclass'.format(newclass.__name__))

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

def test_matern32_jvp():
    x = np.random.randn(100)
    r1 = autograd.elementwise_grad(_kernels._matern32)(x)
    r2 = autograd.deriv(_kernels._matern32)(x)
    assert np.allclose(r1, r2)

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
            kernelf = autograd.deriv(kernelf, 0)
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
# arise where x == y.
xfail(TestMatern, 'test_symmetric_21')
xfail(TestMatern, 'test_double_diff_nd_second_chopped')
xfail(TestMatern, 'test_positive_deriv2_nd')
xfail(TestMatern52, 'test_symmetric_21')
xfail(TestMatern52, 'test_double_diff_nd_second_chopped')
xfail(TestMatern52, 'test_positive_deriv2_nd')
xfail(TestPPKernel, 'test_positive_deriv2')
xfail(TestPPKernel, 'test_positive_deriv2_nd')
pytest.mark.xfail(test_matern_spec_21)
pytest.mark.xfail(test_matern_spec_22)

# TODO This one should not fail, it's a first derivative! Probably it's the
# case D = 1 that fails because that's the maximum dimensionality. For some
# reason I don't catch it without taking a derivative.
xfail(TestPPKernel, 'test_positive_deriv_nd')

# TODO These are not isotropic kernels, what is the problem?
xfail(TestTaylor, 'test_double_diff_nd_second')
xfail(TestNNKernel, 'test_double_diff_nd_second')
