# lsqfitgp/docs/runcode.py
#
# Copyright (c) 2020, 2022, 2023, Giacomo Petrillo
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
import contextlib
import os
import pathlib

import numpy as np
from matplotlib import pyplot as plt
import gvar
import pygments
from pygments import lexers, formatters

def pyprint(text):
    print(pygments.highlight(text, lexers.PythonLexer(), formatters.TerminalFormatter()))

@contextlib.contextmanager
def switchgvar():
    try:
        yield gvar.switch_gvar()
    finally:
        gvar.restore_gvar()

@contextlib.contextmanager
def chdir(path):
    try:
        cwd = pathlib.Path.cwd()
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)

pattern = re.compile(r'(?m)(?!\.\..+?)^.*?::\n\s*?\n(( {4,}.*\n)+)\s*?\n')

def runcode(file):

    file = pathlib.Path(file)

    # read source
    text = pathlib.Path(file).read_text()
    
    # reset working environment
    plt.close('all')
    np.random.seed(0)
    gvar.ranseed(0)
    globals_dict = {}
    with switchgvar():

        # run code
        for match in pattern.finditer(text):
            codeblock = match.group(1)
            print(58 * '-' + '\n')
            code = textwrap.dedent(codeblock).strip()
            printcode = '\n'.join(f' {i + 1:2d}  ' + l for i, l in enumerate(code.split('\n')))
            pyprint(printcode)

            with chdir(file.parent):
                exec(code, globals_dict)

for file in sys.argv[1:]:
    s = f'*  running {file}  *'
    line = '*' * len(s)
    print('\n' + line + '\n' + s + '\n' + line)
    runcode(file)
