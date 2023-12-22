# lsqfitgp/_gvarext/_tabulate.py
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

import textwrap

import gvar
import numpy

def tabulate_together(*gs, headers=True, offset='', ndecimal=None, keys=None):
    """
    
    Format a table comparing side by side various collections of gvars.
    
    Parameters
    ----------
    *gs : sequence of arrays or dictionaries of gvars
        The variables to be tabulated. The structures of arrays and dictionaries
        must match.
    headers : bool or sequence of strings
        If True (default), add automatically an header. If False, don't add an
        header. If a sequence with length len(gs) + 1, it contains the column
        names for the keys/indices and for each set of variables.
    offset : str
        Prefix to each line, default empty.
    ndecimal : int, optional
        Number of decimal places. If not specified (default), keep two error
        digits.
    keys : sequence, optional
        If ``gs`` are dictionaries, a subset of keys to be extracted from each
        dictionary. Ignored if they are arrays.
    
    Examples
    --------
    >>> print(tabulate_together(gvar.gvar(dict(a=1)), gvar.gvar(dict(a=2))))
    key/index   value1   value2
    ---------------------------
            a    1 (0)    2 (0)
    
    See also
    --------
    gvar.tabulate
    
    """
    if not gs:
        return ''
    gs = [g if hasattr(g, 'keys') else numpy.asarray(g) for g in gs]
    assert all(hasattr(g, 'keys') for g in gs) or all(not hasattr(g, 'keys') for g in gs)
    if keys is not None and hasattr(gs[0], 'keys'):
        gs = [{k: g[k] for k in keys} for g in gs]
    g0 = gs[0]
    if hasattr(g0, 'keys'):
        assert all(set(g.keys()) == set(g0.keys()) for g in gs[1:])
        gs = [{k: g[k] for k in g0} for g in gs]
    else:
        assert all(g.shape == g0.shape for g in gs[1:])
        if g0.shape == ():
            gs = [{'--': g} for g in gs]
    tables = [
        _splittable(gvar.tabulate(g, headers=['@', ''], ndecimal=ndecimal))
        for g in gs
    ]
    columns = list(tables[0]) + [t[1] for t in tables[1:]]
    if not hasattr(headers, '__len__'):
        if headers:
            headers = ['key/index'] + [f'value{i+1}' for i in range(len(gs))]
        else:
            headers = None
    else:
        assert len(headers) == len(columns)
    if headers is not None:
        columns = (_head(col, head) for col, head in zip(columns, headers))
    return textwrap.indent(_join(columns), offset)

def _splittable(table):
    lines = table.split('\n')
    header = lines[0]
    col = header.find('@') + 1
    contentlines = lines[2:]
    col1 = '\n'.join(line[:col] for line in contentlines)
    col2 = '\n'.join(line[col:] for line in contentlines)
    return col1, col2

def _head(col, head):
    head = str(head)
    width = col.find('\n')
    if width < 0:
        width = len(col)
    hwidth = len(head)
    if hwidth > width:
        col = textwrap.indent(col, (hwidth - width) * ' ')
    else:
        head = (width - hwidth) * ' ' + head
    return head + '\n' + len(head) * '-' + '\n' + col

def _join(cols):
    split = (col.split('\n') for col in cols)
    return '\n'.join(''.join(lines) for lines in zip(*split))
