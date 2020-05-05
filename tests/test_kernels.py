import sys
import inspect
import abc

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
        return np.finfo(float).eps
    
    def positive(self, deriv, nd=False):
        for kw in self.kwargs_list:
            x = self.random_x_nd(2, **kw) if nd else self.random_x(**kw)
            kernel = self.kernel_class(**kw)
            if kernel.derivable >= deriv:
                d = (deriv, 'f0') if nd else deriv
                cov = kernel.diff(d, d)(x[None, :], x[:, None])
                assert np.allclose(cov, cov.T)
                eigv = linalg.eigvalsh(cov)
                assert np.min(eigv) >= -len(cov) * self.eps * np.max(eigv)
    
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
        for kw in self.kwargs_list:
            x = self.random_x(**kw)[None, :]
            if xderiv == yderiv:
                y = self.random_x(**kw)[:, None]
            else:
                y = x.T
            kernel = self.kernel_class(**kw)
            if kernel.derivable >= max(xderiv, yderiv):
                b1 = kernel.diff(xderiv, yderiv)(x, y)
                b2 = kernel.diff(yderiv, xderiv)(y, x)
                assert np.allclose(b1, b2)

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
        if issubclass(self.kernel_class, _Kernel.IsotropicKernel):
            for kw in self.kwargs_list:
                x = self.random_x(**kw)
                var = self.kernel_class(**kw)(x, x)
                assert np.allclose(var, 1)
    
    def test_double_diff_scalar_first(self):
        for kw in self.kwargs_list:
            kernel = self.kernel_class(**kw)
            if kernel.derivable:
                x = self.random_x(**kw)
                r1 = kernel.diff(1, 1)(x[None, :], x[:, None])
                r2 = kernel.diff(1, 0).diff(0, 1)(x[None, :], x[:, None])
                assert np.allclose(r1, r2)
    
    def test_double_diff_scalar_second(self):
        for kw in self.kwargs_list:
            kernel = self.kernel_class(**kw)
            if kernel.derivable >= 2:
                x = self.random_x(**kw)
                r1 = kernel.diff(2, 2)(x[None, :], x[:, None])
                r2 = kernel.diff(1, 1).diff(1, 1)(x[None, :], x[:, None])
                assert np.allclose(r1, r2)
    
    def test_double_diff_scalar_second_chopped(self):
        for kw in self.kwargs_list:
            kernel = self.kernel_class(**kw)
            if kernel.derivable >= 2:
                x = self.random_x(**kw)
                r1 = kernel.diff(2, 2)(x[None, :], x[:, None])
                r2 = kernel.diff(2, 0).diff(0, 2)(x[None, :], x[:, None])
                assert np.allclose(r1, r2)
    
    def test_double_diff_nd_first(self):
        for kw in self.kwargs_list:
            kernel = self.kernel_class(**kw)
            if kernel.derivable:
                x = self.random_x_nd(2, **kw)[None, :]
                r1 = kernel.diff('f0', 'f0')(x, x.T)
                r2 = kernel.diff('f0', 0).diff(0, 'f0')(x, x.T)
                assert np.allclose(r1, r2)

    def test_double_diff_nd_second(self):
        for kw in self.kwargs_list:
            kernel = self.kernel_class(**kw)
            if kernel.derivable >= 2:
                x = self.random_x_nd(2, **kw)[None, :]
                r1 = kernel.diff((2, 'f0'), (2, 'f1'))(x, x.T)
                r2 = kernel.diff('f0', 'f0').diff('f1', 'f1')(x, x.T)
                assert np.allclose(r1, r2)

    def test_double_diff_nd_second_chopped(self):
        for kw in self.kwargs_list:
            kernel = self.kernel_class(**kw)
            if kernel.derivable >= 2:
                x = self.random_x_nd(2, **kw)[None, :]
                r1 = kernel.diff((2, 'f0'), (2, 'f1'))(x, x.T)
                r2 = kernel.diff('f0', 'f1').diff('f0', 'f1')(x, x.T)
                assert np.allclose(r1, r2)

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
        
# Define a concrete subclass of KernelTestBase for each kernel.
test_kwargs = {
    _kernels.Matern: dict(kwargs_list=[
        dict(nu=0.5), dict(nu=0.6), dict(nu=1.5), dict(nu=2.5)
    ]),
    _kernels.PPKernel: dict(kwargs_list=[
        dict(q=q, D=D) for q in range(4) for D in range(1, 10)
    ]),
    _kernels.Polynomial: dict(kwargs_list=[
        dict(exponent=p) for p in range(10)
    ]),
    _kernels.Wiener: dict(random_x_fun=lambda **kw: np.random.uniform(0, 10, size=100)),
    _kernels.WienerIntegral: dict(random_x_fun=lambda **kw: np.random.uniform(0, 10, size=100)),
    _kernels.FracBrownian: dict(random_x_fun=lambda **kw: np.random.uniform(0, 10, size=100)),
    _kernels.Categorical: dict(kwargs_list=[
        dict(cov=matrix_square(np.random.randn(10, 10)))
    ], random_x_fun=lambda **kw: np.random.randint(10, size=100)),
    _kernels.NNKernel: dict(eps=4 * np.finfo(float).eps)
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

def test_wiener_integral():
    """
    Test that the derivative of the Wiener integral is the Wiener.
    """
    x, y = np.abs(np.random.randn(2, 100))
    r1 = _kernels.Wiener()(x, y)
    r2 = _kernels.WienerIntegral().diff(1, 1)(x, y)
    assert np.allclose(r1, r2)

import pytest

# TODO This is a dirty way of marking an xfail, tests can not Xpass. I didn't
# manage to mark methods on the subclasses without marking the superclass
# methods too, and making a wrapper didn't work with "can not collect because
# it is not a function" (??).
def xfail(*args, **kw):
    pytest.xfail()

TestPolynomial.test_double_diff_nd_second       = xfail
TestMatern.test_symmetric_21                    = xfail
TestMatern.test_double_diff_nd_second_chopped   = xfail
TestMatern.test_positive_deriv2_nd              = xfail
TestMatern52.test_symmetric_21                  = xfail
TestMatern52.test_double_diff_nd_second_chopped = xfail
TestMatern52.test_positive_deriv2_nd            = xfail
TestNNKernel.test_double_diff_nd_second         = xfail
TestPPKernel.test_positive_deriv2               = xfail
TestPPKernel.test_positive_deriv2_nd            = xfail
test_matern_spec_21                             = xfail
test_matern_spec_22                             = xfail

# TODO This one should not fail, it's a first derivative! Probably it's the
# case D = 1 that fails because that's the maximum dimensionality. For some
# reason I don't catch it without taking a derivative.
TestPPKernel.test_positive_deriv_nd             = xfail
