import functools
import warnings

from jax import tree_util
import numpy as np
from jax import numpy as jnp
import pytest

def jaxtonumpy(x):
    """
    Recursively convert jax arrays in x to numpy arrays.
    """
    children, meta = tree_util.tree_flatten(x)
    children = (np.array(x) if isinstance(x, jnp.ndarray) else x for x in children)
    return tree_util.tree_unflatten(meta, children)

def assert_equal(*args):
    """
    Version of assert_equal that works with jax arrays.
    """
    np.testing.assert_equal(*map(jaxtonumpy, args))

def mark(cls, meth, mark):
    """
    Function to mark a test method.
    """
    impl = getattr(cls, meth)
    if not meth.startswith('test_'):
        warnings.warn(f'method {cls.__name__}.{meth} not prefixed with test_')
    @getattr(pytest.mark, mark)
    @functools.wraps(impl) # `wraps` needed because pytest uses the method name
    def newimpl(self):
        # wrap because otherwise the superclass method would be marked too,
        # but set the mark temporarily to allow introspection
        pytest.mark.xfail(impl)
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

    meta['newfun'] = newfun
    return newfun
