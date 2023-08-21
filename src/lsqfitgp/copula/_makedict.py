# lsqfitgp/copula/_makedict.py
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

""" defines makedict """

import gvar

from . import _base

def makedict(variables, prefix='__copula_'):
    """

    Expand distributions in a dictionary.

    Parameters
    ----------
    variables : dict
        A dictionary representing a collection of probability distribution. If a
        value is an instance of `DistrBase`, the key is converted to mark a
        transformation and the value is replaced with new primary gvars.
    prefix : str
        A prefix to make the transformation names unique.

    Returns
    -------
    out : BufferDict
        The transformed dictionary. Recognizes the same keys as `variables`,
        but squashes the values through the transformation that sends a Normal
        to the desired distribution.

    Raises
    ------
    ValueError :
        If any `DistrBase` object appears under different keys.

    Examples
    --------

    Put a `Distr` into a `gvar.BufferDict`:

    >>> bd = lgp.copula.makedict({'x': lgp.copula.beta(1, 1)})
    >>> bd
    BufferDict({'__copula_beta{1, 1}(x)': 0.0(1.0)})
    >>> bd['x']
    0.50(40)
    >>> bd['__copula_beta{1, 1}(x)']
    0.0(1.0)

    You can also put an entire `Copula`:

    >>> bd = lgp.copula.makedict({
    ...     'x': lgp.copula.Copula({
    ...         'y': lgp.copula.gamma(2, 1 / 2),
    ...         'z': lgp.copula.invgamma(2, 2),
    ...     }),
    ... })
    >>> bd
    BufferDict({"__copula_{'y': gamma{2, 0.5}, 'z': invgamma{2, 2}}(x)": array([0.0(1.0), 0.0(1.0)], dtype=object)})
    >>> bd['x']
    {'y': 3.4(2.5), 'z': 1.19(90)}

    Other entries and transformations are left alone:

    >>> bd = lgp.copula.makedict({
    ...     'x': gvar.gvar(3, 0.2),
    ...     'log(y)': gvar.gvar(0, 1),
    ...     'z': lgp.copula.dirichlet(1.5, [1, 2, 3]),
    ... })
    >>> bd
    BufferDict({'x': 3.00(20), 'log(y)': 0.0(1.0), '__copula_dirichlet{1.5, [1, 2, 3], shape=3}(z)': array([0.0(1.0), 0.0(1.0), 0.0(1.0)], dtype=object)})
    >>> bd['z']
    array([0.06(20), 0.31(49), 0.63(51)], dtype=object)
    >>> bd['y']
    1.0(1.0)

    Since shared `DistrBase` objects represent statistical dependency, it is
    forbidden to have the same object appear under different keys, as that
    would make it impossible to take the dependencies into account:

    >>> x = lgp.copula.beta(1, 1)
    >>> y = lgp.copula.beta(1, x)
    >>> lgp.copula.makedict({'x': x, 'y': y})
    ValueError: cross-key occurrences of object(s):
    beta with id 10952248976: <x>, <y.1>

    """

    # collect all objects and their representations in DistrBase instances
    caches = {}
    for k, v in variables.items():
        if isinstance(v, _base.DistrBase):
            cache = {}
            v.__repr__(k, cache)
            caches[k] = cache

    # put everything into a single multiple valued dict
    allobjects = {}
    for cache in caches.values():
        for obj, descr in cache.items():
            allobjects.setdefault(obj, []).append(descr)

    # find objects that appear multiple times
    multiple = ''
    for obj, descrs in allobjects.items():
        if len(descrs) > 1:
            multiple += f'{obj.__class__.__name__} with id {id(obj)}: {", ".join(descrs)}\n'

    # raise an error if there are
    if multiple:
        raise ValueError(f'cross-key occurrences of object(s):\n{multiple}')

    out = {}
    for k, v in variables.items():
        if isinstance(v, _base.DistrBase):
            name = str(v._staticdescr).replace('(', '{').replace(')', '}')
            assert '(' not in prefix and ')' not in prefix
                # gvar does not currently check presence of parentheses, see
                # https://github.com/gplepage/gvar/issues/39
            name = prefix + name
            v.add_distribution(name)
            v = v.gvars()
            k = f'{name}({k})'
        assert k not in out
        out[k] = v
    return gvar.BufferDict(out)


