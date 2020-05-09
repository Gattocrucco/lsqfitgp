# lsqfitgp/docs/runcode.py
#
# Copyright (c) 2020, Giacomo Petrillo
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

"""Search for rst files in the current directory and run the python code in
them. Or, run the files specified on the command line."""

import re
import sys
import numpy as np

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter

def pyprint(text):
    print(highlight(text, PythonLexer(), TerminalFormatter()))

sys.path.insert(0, '..')

pattern = re.compile(r'::\n\s*?\n(( {4,}.*\n)+)\s*?\n')

def runcode(file):
    with open(file, 'r') as stream:
        text = stream.read()
    globals_dict = {}
    np.random.seed(0)
    for match in pattern.finditer(text):
        codeblock = match.group(1)
        print('---------------------------------------------------------\n')
        pyprint(codeblock)
        code = '\n'.join(line[4:] for line in codeblock.split('\n'))
        exec(code, globals_dict)

for file in sys.argv[1:]:
    print('running {}...'.format(file))
    runcode(file)
