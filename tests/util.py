import functools
import warnings

from jax import tree_util
import numpy as np
from jax import numpy as jnp
import gvar
import pytest
from scipy import linalg

def jaxtonumpy(x):
    """
    Recursively convert jax arrays in x to numpy arrays.
    """
    children, meta = tree_util.tree_flatten(x)
    children = (np.array(x) if isinstance(x, jnp.ndarray) else x for x in children)
    return tree_util.tree_unflatten(meta, children)

def assert_equal(*args):
    """
    Version of assert_equal that works with jax arrays, and which fixes
    numpy issue #21739
    """
    assert_array_equal(*map(jaxtonumpy, args))

def assert_array_equal(*args):
    a = args[0]
    if isinstance(a, np.ndarray) and a.size == 0 and a.dtype.names:
        assert all(b.dtype == a.dtype for b in args)
        assert all(b.shape == a.shape for b in args)
    elif isinstance(a, np.ndarray) and a.dtype.names:
        # old versions of numpy force structured dtypes to be equal when
        # comparing, instead of casting
        assert all(b.dtype.names == a.dtype.names for b in args)
        for n in a.dtype.names:
            assert_array_equal(*(x[n] for x in args))
    else:
        np.testing.assert_equal(*args)

def mark(cls, meth, mark):
    """
    Function to mark a test method.
    """
    impl = getattr(cls, meth)
    if not meth.startswith('test_'):
        warnings.warn(f'method {cls.__name__}.{meth} not prefixed with test_')
    marker = getattr(pytest.mark, mark)
    @marker
    @functools.wraps(impl) # `wraps` needed because pytest uses the method name
    def newimpl(self):
        # wrap because otherwise the superclass method would be marked too,
        # but set the mark temporarily to allow introspection
        marker(impl)
        try:
            impl(self)
        finally:
            impl.pytestmark.pop()
    setattr(cls, meth, newimpl)

def xfail(cls, meth):
    mark(cls, meth, 'xfail')

def skip(cls, meth):
    mark(cls, meth, 'skip')

def tryagain(fun, rep=2, method=False):
    """
    Decorates test `fun` to make it try again in case of failure. Inhibited
    by xfails.
    """
    meta = {}
    @functools.wraps(fun)
    def newfun(*args, _meta=meta, **kw):

        job = lambda: fun(*args, **kw)

        marks = getattr(_meta['newfun'], 'pytestmark', [])
        if any(m.name == 'xfail' for m in marks):
            return job()

        if method:
            self = args[0]
            name = f'{self.__class__.__name__}.{fun.__name__}'
        else:
            name = fun.__name__

        for i in range(rep):
            try:
                x = job()
                if i > 0:
                    warnings.warn(f'Test {name} failed {i} times with exception {exc.__class__.__name__}: ' + ", ".join(exc.args))
                return x
            except Exception as e:
                exc = e

        # if rep > 1:
        #     warnings.warn(f'Test {name} failed {rep} times')
        raise exc
    
    if hasattr(fun, 'pytestmark'):
        newfun.pytestmark = fun.pytestmark
    meta['newfun'] = newfun
    return newfun

def assert_close_matrices(actual, desired, *, rtol=0, atol=0):
    if actual.shape == desired.shape and actual.size == 0:
        return
    dnorm = linalg.norm(desired, 2)
    adnorm = linalg.norm(actual - desired, 2)
    msg = f"""\
matrices actual and desired are not close in 2-norm
norm(desired) = {dnorm:.2g}
norm(actual - desired) = {adnorm:.2g}  (atol = {atol:.2g})
ratio = {adnorm / dnorm:.2g}  (rtol = {rtol:.2g})"""
    assert adnorm <= atol + rtol * dnorm, msg

def _assert_similar_gvars(g, h, rtol, atol):
    assert_allclose(gvar.mean(g), gvar.mean(h), rtol=rtol, atol=atol)
    g = np.reshape(g, -1)
    h = np.reshape(h, -1)
    assert_close_matrices(gvar.evalcov(g), gvar.evalcov(h), rtol=rtol, atol=atol)

def assert_similar_gvars(*gs, rtol=0, atol=0):
    if gs:
        for g in gs[1:]:
            _assert_similar_gvars(g, gs[0], rtol, atol)

def assert_same_gvars(g, h, *, atol=0):
    z = g - h
    z = np.reshape(z, -1)
    assert_allclose(gvar.mean(z), np.zeros(z.shape), rtol=0, atol=atol)
    assert_close_matrices(gvar.evalcov(z), np.zeros(2 * z.shape), rtol=0, atol=atol)

def assert_close_decomps(actual, desired, *, rtol=0, atol=0):
    assert actual.n == desired.n
    assert_close_matrices(actual.inv(), desired.inv(), rtol=rtol, atol=atol)

def assert_allclose(actual, desired, *, rtol=0, atol=0, equal_nan=False, **kw):
    """ change the default arguments of np.testing.assert_allclose """
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, equal_nan=equal_nan, **kw)
