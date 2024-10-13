# lsqfitgp/_utils.py
#
# Copyright (c) 2023, 2024, Giacomo Petrillo
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

def top_bottom_rule(title, body):
    body = textwrap.dedent(body)
    body_lines = list(map(str.rstrip, body.split('\n')))
    body = '\n'.join(body_lines)
    body_width = max(map(len, body_lines))
    title_width = 2 + len(title)
    width = max(body_width, title_width + 4)
    pre_length = (width - title_width) // 2
    post_length = width - title_width - pre_length
    toprule = '=' * pre_length + ' ' + title + ' ' + '=' * post_length
    bottomrule = '=' * width
    return '\n'.join([toprule, body, bottomrule])
