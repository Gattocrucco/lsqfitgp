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

from . import _distr

def makedict(variables, prefix='__copula_'):
    """

    Expand distributions in a dictionary.

    Parameters
    ----------
    variables : dict
        A dictionary representing a collection of probability distribution. If a
        value is an instance of `Distr`, the key is converted to mark a
        transformation and the value is replaced with new primary gvars.
    prefix : str
        A prefix to make the transformation names unique.

    Returns
    -------
    out : BufferDict
        The transformed dictionary. Recognizes the same keys as `variables`,
        but squashes the values through the transformation that sends a Normal
        to the desired distribution.

    Examples
    --------

    >>> bd = lgp.copula.makedict({
    ...    'x': lgp.copula.beta(1, 1),
    ...    'y': lgp.copula.gamma(3, 5),
    ...    'z': gvar.gvar(0, 1),
    ... })
    >>> bd
    BufferDict({'__copula_beta{1, 1}(x)': 0.0(1.0), '__copula_gamma{3, 5}(y)': 0.0(1.0), 'z': 0.0(1.0)})
    >>> bd['x']
    0.50(40)
    >>> bd['__copula_beta{1, 1}(x)']
    0.0(1.0)

    """
    out = {}
    for k, v in variables.items():
        if isinstance(v, _distr.Distr):
            name = str(v._staticdescr).replace('(', '{').replace(')', '}')
            assert '(' not in prefix and ')' not in prefix
            name = prefix + name
            v.add_distribution(name)
            v = v.gvars()
            k = f'{name}({k})'
        assert k not in out
        out[k] = v
    return gvar.BufferDict(out)
