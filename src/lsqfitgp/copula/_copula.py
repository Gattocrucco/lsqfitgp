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
import inspect

from . import _distr
from . import _copulas

class Copula:

    _distrs = {
        k: v for k, v in vars(_copulas).items()
        if inspect.isclass(v) and issubclass(v, _distr.Distr)
    }

    def __init__(self, variables={}):
        self._variables = {}
        for k, v in variables.items():
            self[k] = v

    def __getitem__(self, name):
        return self._variables[name]

    def __setitem__(self, name, value):
        if name in self._variables:
            raise AttributeError(f'cannot overwrite variable {name!r}')
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

    def __getattr__(self, name):
        if name in self._distrs:
            return self._distrs[name]
        elif name == 'Copula':
            return __class__
        else:
            raise AttributeError(name)

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
            out = f'Copula({{\n{out}}})'
        else:
            out = 'Copula()'
        
        return out

# TODO since the copula is mutable, it can't have an invfcn an in_size, but it
# should instead generate them. So Copula.make_invfcn() -> invfcn, in_size.
