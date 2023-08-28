# lsqfitgp/docs/reference/kernelop.py
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

""" Generate documentation of kernel transformations """

import inspect
import pathlib
import collections
import textwrap

import numpy as np
import lsqfitgp as lgp
import tabulate
import polars as pl

out = """\
.. file generated automatically by lsqfitgp/docs/reference/kernelop.py

.. currentmodule:: lsqfitgp

Kernel transformations
======================

Kernel transformations are applied through the `CrossKernel` methods
`~CrossKernel.transf`, `~CrossKernel.linop`, `~CrossKernel.algop`. A
transformation returns a new kernel object derived from the input ones and
additional arguments. Example::

    import lsqfitgp as lgp
    K = lgp.ExpQuad()
    Q = (K
        .linop('scale', 2)   # rescale the input
        .linop('diff', 1, 0) # derive w.r.t. the first argument
        .algop('expm1')      # amplify positive correlations
        .transf('forcekron') # force the kernel to be a product over dimensions
    )

A kernel can access all transformations defined in its superclasses. However,
most transformations will regress the class of the output at least to the
superclass which actually defines the transformation. Example::

    K = lgp.ExpQuad()
    assert isinstance(K, lgp.IsotropicKernel)
    Q = K.linop('dim', 'a') # consider only dimension 'a' of the input
    assert not isinstance(Q, lgp.IsotropicKernel))

Index
-----
"""

# collect all transformations from all kernels
transfs = {}
for name, obj in vars(lgp).items():
    if inspect.isclass(obj) and issubclass(obj, lgp.CrossKernel):
        for name, transf in obj.list_transf(superclasses=False).items():
            transfs.setdefault(name, []).append(transf)

# check that there are no namesakes with different documentation or kind
for name, tlist in transfs.items():
    t0 = tlist[0]
    for t in tlist[1:]:
        assert t.doc == t0.doc
        assert t.kind == t0.kind

# define how to print method in table
kind = {
    'algop': 'algop',
    'linop': 'linop',
    None: 'transf',
}
def get_method(tlist):
    meth = kind[tlist[0].kind]
    return f'`~CrossKernel.{meth}`'

# define how to print name in table
def get_name(name):
    return f'`{name}`_'

# define how to print class list in table
groups = {
    frozenset({
        lgp.IsotropicKernel, lgp.StationaryKernel, lgp.Kernel,
        lgp.CrossKernel, lgp.CrossStationaryKernel, lgp.CrossIsotropicKernel,
    }): ':ref:`All generic classes <generickernel>`',
}
def get_classes(tlist):
    classes = [t.tcls for t in tlist]
    default = ', '.join(f'`{c.__name__}`' for c in classes)
    return groups.get(frozenset(classes), default)

# make table
columns = collections.defaultdict(list)
for name, tlist in transfs.items():
    columns['name'].append(name)
    columns['tlist'].append(tlist)
    columns['Method'].append(get_method(tlist))
    columns['Name'].append(get_name(name))
    columns['Classes'].append(get_classes(tlist))
table = pl.DataFrame(columns).sort('Method', 'Name')

# write table to text
index_table = table.select('Method', 'Name', 'Classes').to_dict()
out += tabulate.tabulate(index_table, headers='keys', tablefmt='rst')

out += """

Transformations
---------------
"""

# format each transformation
for name, tlist in table.select('name', 'tlist').iter_rows():
    tlist = transfs[name]
    t = tlist[0]
    meth = kind[t.kind]
    out += f"""
.. _{name}:
.. method:: CrossKernel.{meth}('{name}')
    :no-index:
"""
    if t.doc:
        doc = textwrap.dedent(t.doc).strip()
        doc = textwrap.indent(doc, '        ')
        out += f"""
    .. code-block:: text

{doc}

"""

# TODO maybe I can use https://www.sphinx-doc.org/en/master/extdev/appapi.html#sphinx.application.Sphinx.add_object_type

# write file
outfile = pathlib.Path(__file__).with_suffix('.rst').relative_to(pathlib.Path().absolute())
print(f'writing to {outfile}...')
outfile.write_text(out)
