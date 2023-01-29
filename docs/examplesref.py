# lsqfitgp/docs/examplesref.py
#
# Copyright (c) 2022, 2023, Giacomo Petrillo
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

"""Generate a file with the list of example scripts."""

outputfile = 'examplesref.rst'

import pathlib
import re
import textwrap

examples = pathlib.Path('../examples').glob('*.py')
examples = list(sorted(examples))

pattern = re.compile(r'(?s)"""(.+)"""')

shortindex = ''
longindex = ''

for example in examples:
    
    # get script name and url
    name = example.name
    if name == 'runexamples.py':
        continue
    url = f'https://github.com/Gattocrucco/lsqfitgp/blob/master/examples/{name}'
    
    # get description from docstring
    with open(example, 'r') as stream:
        text = stream.read()
    if match := pattern.search(text):
        descr = match.group(1)
        descr = ': ' + textwrap.indent(textwrap.dedent(descr), 4 * ' ').strip()
    else:
        descr = ''
        
    # get eventual figure
    base = example.stem
    imgfile = example.parent / 'plot' / f'{base}.png'
    if imgfile.exists():
        image = f"""
    
    .. image:: {imgfile}"""
    else:
        image = ''
    
    # append list item
    shortindex += f'{base}_ '
    longindex += f"""
.. _{base}:

  * `{name} <{url}>`_{descr}{image}
"""

out = f"""\
.. file generated automatically by lsqfitgp/docs/examplesref.py

.. currentmodule:: lsqfitgp

.. _examplesref:

Example scripts
===============

This is an index of the example scripts in the ``examples`` directory in the
repository. The links point to the file preview on github.

Short index
-----------
{shortindex}

Long index
----------
{longindex}
"""

print(f'writing to {outputfile}...')
with open(outputfile, 'w') as file:
    file.write(out)
