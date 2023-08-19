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

from . import _distr
from . import _copulas

#   Make Distr.__call__ emit an explicative error. Use the Copula
#   with c.invfcn(array) -> T = dict[name, T | array], c.in_size.

class Copula:

    def __init__(self, **variables):
        super().__setattr__('variables', {})
        super().__setattr__('names', {})
        for k, v in variables.items():
            setattr(self, k, v)

    def __setattr__(self, name, value):
        assert name != '__wrapped__'
        if name in self.variables:
            raise AttributeError(f'cannot overwrite attribute {name!r}')
        elif isinstance(value, (__class__, _distr.Distr)):
            if isinstance(value, __class__):
                found = list(value._recursive_object_search(self, '<copula>'))
                if found:
                    paths = ', '.join(found)
                    raise ValueError(f'cannot set attribute {name!r} to a '
                        f'copula that contains self as {paths}')
            self.variables[name] = value
            self.names[value] = name
        else:
            raise TypeError(f'cannot set attribute {name!r} to {value!r}')

    def _recursive_object_search(self, obj, path='self'):
        if obj is self:
            yield path
        for k, v in self.variables.items():
            subpath = path + '.' + k
            if isinstance(v, __class__):
                yield from v._recursive_object_search(obj, subpath)
            elif v is obj:
                yield subpath

    def __getattr__(self, name):
        if name in self.variables:
            return self.variables[name]
        else:
            return self._CopulaAttrProxy(self, name)

    def __delattr__(self, name):
        raise AttributeError(f'cannot delete attribute {name!r}')

    class _CopulaAttrProxy:

        def __init__(self, copula, name):
            super.__setattr__(self, 'copula', copula)
            super.__setattr__(self, 'name', name)

        def __call__(self, distr, *args, **kw):
            if isinstance(distr, str):
                distr = getattr(_copulas, distr)
            if not issubclass(distr, _distr.Distr):
                raise TypeError(f'expected a Distr, got {distr!r}')
            distr = distr(*args, **kw)
            setattr(self.copula, self.name, distr)

        def __getattr__(self, name):
            copula = Copula()
            setattr(self.copula, self.name, copula)
            return getattr(copula, name)

        def __setattr__(self, name, value):
            copula = Copula()
            setattr(self.copula, self.name, copula)
            setattr(copula, name, value)

        def __repr__(self):
            return f'attribute proxy .{self.name} of {self.copula!r}'

    def __repr__(self, path='', cache=None):
        if cache is None:
            cache = {}
        if self in cache:
            return cache[self]
        cache[self] = f'<{path}>'
                
        indent = '    '
        out = 'Copula(\n'
        
        for k, v in self.variables.items():
            if isinstance(v, (__class__, _distr.Distr)):
                sub = v.__repr__('.'.join((path, k)).lstrip('.'), cache)
            else:
                sub = repr(v)
            
            sub = textwrap.indent(sub, indent).lstrip()
            out += f'{indent}{k}={sub},\n'
        
        if not self.variables:
            out = out.rstrip()
        out += ')'
        
        return out
