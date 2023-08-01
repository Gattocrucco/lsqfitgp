# lsqfitgp/examples/runexamples.py
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

"""Run the scripts given on the command line and saves the figures produced
in the same directory of each corresponding script."""

import sys
import os
import warnings
import gc

import numpy as np
from matplotlib import pyplot as plt
import gvar

warnings.filterwarnings('ignore', r'Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure\.')

class SwitchGVar:
    __enter__ = lambda _: gvar.switch_gvar()
    __exit__ = lambda *_: gvar.restore_gvar()

for file in sys.argv[1:]:

    # load source file
    print('\nrunexamples.py: running {}...'.format(file))
    with open(file, 'r') as stream:
        code = stream.read()
    
    # reset working environment and run
    with SwitchGVar():
        plt.close('all')
        np.random.seed(0)
        gvar.ranseed(0)
        globals_dict = {}
        gc.collect()
        exec(code, globals_dict)
    
    # save figures
    nums = plt.get_fignums()
    path, name = os.path.split(file)
    base, _ = os.path.splitext(name)
    prefix = os.path.join(path, 'plot', base)
    directory, _ = os.path.split(prefix)
    os.makedirs(directory, exist_ok=True)
    for num in nums:
        fig = plt.figure(num)
        suffix = f'-{num}' if num > 1 else ''
        out = f'{prefix}{suffix}.png'
        print(f'runexamples.py: write {out}')
        fig.savefig(out)
