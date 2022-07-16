# lsqfitgp/docs/runcode.py
#
# Copyright (c) 2020, 2022, Giacomo Petrillo
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

"""Run the python code in the rst files specified on the command line, but
only if the leading indentation is at least 4 spaces and there is a blank line
after the code block"""

import re
import sys
import textwrap
import warnings

import numpy as np
from matplotlib import pyplot as plt
import gvar

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
    
    # reset working environment
    plt.close('all')
    gvar.switch_gvar()
    np.random.seed(0)
    globals_dict = {}
    
    for match in pattern.finditer(text):
        codeblock = match.group(1)
        print(58 * '-' + '\n')
        code = textwrap.dedent(codeblock).strip()
        printcode = '\n'.join(f' {i + 1:2d}  ' + l for i, l in enumerate(code.split('\n')))
        pyprint(printcode)
        exec(code, globals_dict)

for file in sys.argv[1:]:
    s = f'*  running {file}  *'
    line = '*' * len(s)
    print('\n' + line + '\n' + s + '\n' + line)
    runcode(file)
