import functools

import jax
import numpy
from jax import numpy as jnp
from jax import tree_util

class AutoPyTree:
    """
    Class adding automatic recursive support for jax pytree flattening
    """
    
    def __init_subclass__(cls, **kw):
        tree_util.register_pytree_node_class(cls)
        super().__init_subclass__(**kw)
    
    # Since I decide dinamically which members are children based on their type,
    # I have to cache the jax pytree structure aux_data such that the
    # structure is preserved when constructing an object with tree_unflatten
    # with dummies as children. This happens in jax.jacfwd for some reason.
    @functools.cached_property
    def _aux_data(self):
        jax_vars = []
        other_vars = []
        for n, v in vars(self).items():
            if isinstance(v, (jnp.ndarray, numpy.ndarray, AutoPyTree)):
                jax_vars.append(n)
            else:
                other_vars.append((n, v))
        # assert jax_vars
        return jax_vars, other_vars
    
    def tree_flatten(self):
        """JAX PyTree encoder. See `jax.tree_util.tree_flatten`."""
        jax_vars, _ = self._aux_data
        # print(f'unpacking {jax_vars} from {self.__class__.__name__}')
        children = tuple(getattr(self, n) for n in jax_vars)
        return children, self._aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """JAX PyTree decoder. See `jax.tree_util.tree_unflatten`."""
        self = cls.__new__(cls)
        self._aux_data = aux_data
        jax_vars, other_vars = aux_data
        for n, v in zip(jax_vars, children):
            setattr(self, n, v)
        for n, v in other_vars:
            setattr(self, n, v)
        return self
