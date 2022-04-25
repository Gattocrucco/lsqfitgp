# lsqfitgp/examples/runexamples.py
#
# Copyright (c) 2022, Giacomo Petrillo
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

"""Run the scripts given on the command line and saves the figures produced
in the same directory of each corresponding script."""

import sys
import os

import numpy as np
from matplotlib import pyplot as plt
import gvar

sys.path.insert(0, '..') # to import lsqfitgp if run from examples/

for file in sys.argv[1:]:
    print('running {}...'.format(file))
    with open(file, 'r') as stream:
        code = stream.read()
    
    # reset working environment
    plt.close('all')
    np.random.seed(0)
    gvar.switch_gvar()

    globals_dict = {}
    exec(code, globals_dict)
    
    # save figures
    nums = plt.get_fignums()
    path, name = os.path.split(file)
    base, _ = os.path.splitext(name)
    prefix = os.path.join(path, 'plot', base)
    for num in nums:
        fig = plt.figure(num)
        suffix = f'-{num}' if num > 1 else ''
        fig.savefig(f'{prefix}{suffix}.png')
