# lsqfitgp/_linalg/_pytree.py
#
# Copyright (c) 2022, 2023, Giacomo Petrillo
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

import functools

import numpy
from jax import numpy as jnp
from jax import tree_util

class AutoPyTree:
    """
    Class adding automatic recursive support for jax pytree flattening
    """
    
    def _jax_vars(self):
        """ Returns list of object attribute names which are to be considered
        children of the PyTree node """
        return [
            n for n, v in vars(self).items()
            if isinstance(v, (jnp.ndarray, numpy.ndarray, AutoPyTree))
        ]
    
    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        tree_util.register_pytree_node_class(cls)
    
    # Since I decide dinamically which members are children based on their type,
    # I have to cache the jax pytree structure aux_data such that the
    # structure is preserved when constructing an object with tree_unflatten
    # with dummies as children. This happens in jax.jacfwd for some reason.
    @functools.cached_property
    def _aux_data(self):
        jax_vars = self._jax_vars()
        other_vars = [
            (n, v) for n, v in vars(self).items()
            if n not in jax_vars
        ]
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
