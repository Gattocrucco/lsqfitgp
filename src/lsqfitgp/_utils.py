# lsqfitgp/_utils.py
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

def append_to_docstring(docs, doctail, front=False):
    doctail = textwrap.dedent(doctail)
    dedocs = textwrap.dedent(docs)
    lineend = docs.find('\n')
    indented_lineend = dedocs.find('\n')
    indent = docs[:indented_lineend - lineend]
    if front:
        newdocs = doctail + dedocs
    else:
        newdocs = dedocs + doctail
    return textwrap.indent(newdocs, indent)
