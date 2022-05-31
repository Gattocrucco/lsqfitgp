from jax import tree_util
import numpy as np
from jax import numpy as jnp

def jaxtonumpy(x):
    children, meta = tree_util.tree_flatten(x)
    children = (np.array(x) if isinstance(x, jnp.ndarray) else x for x in children)
    return tree_util.tree_unflatten(meta, children)

def assert_equal(*args):
    np.testing.assert_equal(*map(jaxtonumpy, args))
