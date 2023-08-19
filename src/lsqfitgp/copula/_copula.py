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

import textwrap
import functools

import jax
from jax import tree_util

from .. import _array
from .. import _patch_gvar
from . import _distr

class Copula:

    def __init__(self, variables={}):
        self._variables = {}
        for k, v in variables.items():
            self._setitem(k, v)
                # avoid __setitem__ for immutable subclass; typically the
                # superclass would be immutable, but here the immutable version
                # has substantial additional functionality, so it is naturally a
                # subclass

    def __getitem__(self, name):
        return self._variables[name]

    def __setitem__(self, name, value):
        self._setitem(name, value)

    def _setitem(self, name, value):
        if name in self._variables:
            raise KeyError(f'cannot overwrite variable {name!r}')
        elif isinstance(value, (__class__, _distr.Distr)):
            if isinstance(value, __class__):
                found = list(value._recursive_object_search(self, '<copula>'))
                if found:
                    paths = ', '.join(found)
                    raise ValueError(f'cannot set variable {name!r} to a '
                        f'copula that contains self as {paths}')
            self._variables[name] = value
        else:
            raise TypeError(f'cannot set variable {name!r} to {value!r}, ',
                'type must be Copula or Distr')

    def _recursive_object_search(self, obj, path='self'):
        if obj is self:
            yield path
        for k, v in self._variables.items():
            subpath = path + '.' + k
            if isinstance(v, __class__):
                yield from v._recursive_object_search(obj, subpath)
            elif v is obj:
                yield subpath

    def __repr__(self, path='', cache=None):
        if cache is None:
            cache = {}
        if self in cache:
            return cache[self]
        cache[self] = f'<{path}>'
        
        indent = '    '
        out = ''
        
        for k, v in self._variables.items():
            if isinstance(v, (__class__, _distr.Distr)):
                sub = v.__repr__('.'.join((path, k)).lstrip('.'), cache)
            else:
                sub = repr(v)
            
            sub = textwrap.indent(sub, indent).lstrip()
            out += f"{indent}'{k}': {sub},\n"
        
        if out:
            out = f'{self.__class__.__name__}({{\n{out}}})'
        else:
            out = f'{self.__class__.__name__}()'
        
        return out

    def freeze(self):
        variables = {
            k: v.freeze() if isinstance(v, __class__) else v
            for k, v in self._variables.items()
        }
        return ImmutableCopula(variables)

class ImmutableCopula(Copula, _distr._DistrBase):

    def __setitem__(self, name, value):
        raise TypeError('ImmutableCopula object is immutable')

    def freeze(self):
        return self

    def __init__(self, variables):
        super().__init__(variables)
        cache = set()
        self.in_shape = self._compute_in_size(cache),
        self._ancestor_count = len(cache) - 1
        self.shape = self._compute_shape()

    def _compute_in_size(self, cache):
        if self in cache:
            return 0
        cache.add(self)
        in_size = 0
        for k, v in self._variables.items():
            in_size += v._compute_in_size(cache)
        return in_size

    def _compute_shape(self):
        shape = {}
        for k, v in self._variables.items():
            if isinstance(v, __class__):
                shape[k] = v._compute_shape()
            else:
                shape[k] = v.shape
        return shape

    def _partial_invfcn_internal(self, x, i, cache):
        assert x.ndim == 1

        if self in cache:
            return cache[self], i

        out = {}
        for k, v in self._variables.items():
            out[k], i = v._partial_invfcn_internal(x, i, cache)

        cache[self] = out
        return out, i

    @functools.cached_property
    def _partial_invfcn(self):

        # non vectorized version, check core shapes and call recursive impl
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

    def partial_invfcn(self, x):
        return self._partial_invfcn(x)

# TODO methods to make the thing behave more like a dictionary => or maybe not.
# that would enable converting it to a dict, which would remove the ability to
# track dependencies.

# TODO method to export to BufferDict, raises an error if there are dependencies
# between the keys.
