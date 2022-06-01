import functools

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

def xfail(cls, meth):
    """
    Function to mark a test method as xfail.
    """
    impl = getattr(cls, meth)
    @pytest.mark.xfail
    @functools.wraps(impl) # `wraps` needed because pytest uses the method name
    def newimpl(self):
        # wrap because otherwise the superclass method would be marked too
        impl(self)
    setattr(cls, meth, newimpl)
