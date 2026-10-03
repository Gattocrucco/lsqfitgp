# lsqfitgp/docs/runcode.py
#
# Copyright (c) 2020, 2022, 2023, 2024, 2025, 2026, Giacomo Petrillo
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
after the code block
"""

import contextlib
import gc
import os
import pathlib
import re
import sys
import textwrap
import warnings

import gvar
import jax
import numpy as np
import pygments
from matplotlib import pyplot as plt
from pygments import formatters, lexers

import lsqfitgp as lgp

warnings.filterwarnings('ignore', r'Negative eigenvalue with ')


def pyprint(text):
    print(
        pygments.highlight(text, lexers.PythonLexer(), formatters.TerminalFormatter())
    )


# a literal block (`::`) of lines indented by at least 4 spaces, possibly
# separated by blank lines, ended by a blank line
pattern = re.compile(
    r'(?m)(?!\.\..+?)^.*?::\n\s*?\n((?: {4,}.*\n)(?:(?:[ \t]*\n)*(?: {4,}.*\n))*)\s*?\n'
)


@contextlib.contextmanager
def chdir(path):
    """Change current working directory, and restore it when done."""
    old_dir = pathlib.Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_dir)


def runcode(file):

    file = pathlib.Path(file)

    # read source
    text = pathlib.Path(file).read_text()

    # reset working environment
    plt.close('all')
    np.random.seed(0)
    gvar.ranseed(0)
    globals_dict = {}
    with plt.style.context('tableau-colorblind10', after_reset=True), lgp.switchgvar():
        # run code
        for match in pattern.finditer(text):
            codeblock = match.group(1)
            print(58 * '-' + '\n')
            code = textwrap.dedent(codeblock).strip()
            printcode = '\n'.join(
                f' {i + 1:2d}  ' + l for i, l in enumerate(code.split('\n'))
            )
            pyprint(printcode)

            with chdir(file.parent):
                exec(code, globals_dict)  # noqa: S102, running the docs code is the point

    # cleanup
    gc.collect()
    jax.clear_caches()


for file in sys.argv[1:]:
    s = f'*  running {file}  *'
    line = '*' * len(s)
    print('\n' + line + '\n' + s + '\n' + line)
    runcode(file)
