# lsqfitgp/copula/_copula.py
#
# Copyright (c) 2023, Giacomo Petrillo
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

""" defines Copula """

import functools
import pprint

import jax
from jax import tree_util
from jax import numpy as jnp
import gvar

from .. import _array
from .. import _patch_gvar
from . import _base

class Copula(_base.DistrBase):

    @staticmethod
    def _tree_path_str(path):
        """ format a jax pytree key path as a compact, readable string """
        def parsekey(key):
            if hasattr(key, 'key'):
                return key.key
            elif hasattr(key, 'idx'):
                return key.idx
            else:
                return key
        def keystr(key):
            key = parsekey(key)
            return str(key).replace('.', r'\.')
        return '.'.join(map(keystr, path))

    @classmethod
    def _patch_jax_dict_sorting(cls, pytree):
        """ replace dicts in pytree with a custom dict subclass such their
        insertion order is maintained, see
        https://github.com/google/jax/issues/4085 """
        
        def is_dict(obj):
            return obj.__class__ is dict
        
        def patch_dict(obj):
            if is_dict(obj):
                return tree_util.tree_map(patch_dict, cls._Dict(obj))
            else:
                return obj
        
        return tree_util.tree_map(patch_dict, pytree, is_leaf=is_dict)

    @tree_util.register_pytree_with_keys_class
    class _Dict(dict):

        def tree_flatten_with_keys(self):
            treedef = {k: None for k in self}
            keys_values = [(tree_util.DictKey(k), v) for k, v in self.items()]
            return keys_values, treedef

        @classmethod
        def tree_unflatten(cls, treedef, values):
            return cls(zip(treedef, values))

    def __init__(self, variables):
        variables = self._patch_jax_dict_sorting(variables)
        def check_type(path, obj):
            if not isinstance(obj, _base.DistrBase):
                raise TypeError(f'only Distr or Copula objects can be '
                    f'contained in a Copula, found {obj!r} at '
                    f'<{self._tree_path_str(path)}>')
            return obj
        self._variables = tree_util.tree_map_with_path(check_type, variables)
        cache = set()
        self.in_shape = self._compute_in_size(cache),
        self._ancestor_count = len(cache) - 1
        self.shape = self._compute_shape()

    def _compute_in_size(self, cache):
        if (out := super()._compute_in_size(cache)) is not None:
            return out
        def accumulate(in_size, obj):
            return in_size + obj._compute_in_size(cache)
        return tree_util.tree_reduce(accumulate, self._variables, 0)

    def _compute_shape(self):
        def shape(obj):
            if isinstance(obj, __class__):
                return obj._compute_shape()
            else:
                return obj.shape
        return tree_util.tree_map(shape, self._variables)

    def _partial_invfcn_internal(self, x, i, cache):
        if (out := super()._partial_invfcn_internal(x, i, cache)) is not None:
            return out
        
        distributions, treedef = tree_util.tree_flatten(self._variables)
        outputs = []
        for distr in distributions:
            out, i = distr._partial_invfcn_internal(x, i, cache)
            outputs.append(out)
        out = tree_util.tree_unflatten(treedef, outputs)

        cache[self] = out
        return out, i

    @functools.cached_property
    def _partial_invfcn(self):

        # non vectorized version, check core shapes and call recursive impl
        # @jax.jit
        def partial_invfcn_0(x):
            assert x.shape == self.in_shape
            cache = {}
            y, i = self._partial_invfcn_internal(x, 0, cache)
            assert i == x.size
            assert len(cache) == 1 + self._ancestor_count
            return y
        partial_invfcn_0_deriv = jax.jacfwd(partial_invfcn_0)

        # add 1-axis vectorization
        partial_invfcn_1 = jax.vmap(partial_invfcn_0)
        partial_invfcn_1_deriv = jax.vmap(partial_invfcn_0_deriv)

        # add gvar support
        def partial_invfcn_2(x):
            
            if x.dtype == object:
                
                # unpack the gvars
                in_mean = gvar.mean(x)
                in_jac, indices = _patch_gvar.jacobian(x)

                # apply function
                out_mean = partial_invfcn_1(in_mean)
                jac = partial_invfcn_1_deriv(in_mean)

                # concatenate derivatives and repack as gvars
                def contract_and_pack(out_mean, jac):
                    # indices:
                    # b = broadcast
                    # i = input
                    # ... = output
                    # g = gvar indices
                    out_jac = jnp.einsum('b...i,big->b...g', jac, in_jac)
                    return _patch_gvar.from_jacobian(out_mean, out_jac, indices)

                return tree_util.tree_map(contract_and_pack, out_mean, jac)

            else:
                return partial_invfcn_1(x)

        # add full vectorization
        def partial_invfcn_3(x):
            x = _array.asarray(x)
            assert x.shape[-1:] == self.in_shape
            head = x.shape[:-1]
            x = x.reshape((-1,) + self.in_shape)
            y = partial_invfcn_2(x)
            def reshape_y(y, shape):
                assert y.shape[1:] == shape
                return y.reshape(head + shape)
            return tree_util.tree_map(reshape_y, y, self.shape)

        return partial_invfcn_3

    def __repr__(self, path='', cache=None):
        
        if isinstance(cache := super().__repr__(path, cache), str):
            return cache
        
        def subrepr(k, obj):
            if isinstance(obj, _base.DistrBase):
                k = self._tree_path_str(k)
                return obj.__repr__('.'.join((path, k)).lstrip('.'), cache)
            else:
                return repr(obj)

        class NoQuotesRepr:
            def __init__(self, s):
                self.s = s
            def __repr__(self):
                return self.s
        
        out = tree_util.tree_map_with_path(subrepr, self._variables)
        out = tree_util.tree_map(NoQuotesRepr, out)
        out = pprint.pformat(out, sort_dicts=False)
        return f'{self.__class__.__name__}({out})'

    @functools.cached_property
    def _staticdescr(self):
        return tree_util.tree_map(lambda x: x._staticdescr, self._variables)

# TODO method to export to BufferDict, raises an error if there are dependencies
# between the keys, works only if _variables is has attr keys(), the non-distr
# leaves have invfcns with non-numerical output
