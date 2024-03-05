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
import warnings
import gc
import pathlib
import runpy

import numpy as np
from matplotlib import pyplot as plt
import gvar
import lsqfitgp as lgp
import jax

warnings.filterwarnings('ignore', r'Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure\.')
warnings.filterwarnings('ignore', r'FigureCanvasAgg is non-interactive, and thus cannot be shown')
warnings.filterwarnings('ignore', r'Negative eigenvalue with ')

for file in sys.argv[1:]:

    file = pathlib.Path(file)
    print(f'\nrunexamples.py: running {file}...')
    
    # reset working environment and run
    with lgp.switchgvar():
        plt.close('all')
        np.random.seed(0)
        gvar.ranseed(0)
        runpy.run_path(str(file))
        gc.collect()
        jax.clear_caches()
    
    # save figures
    nums = plt.get_fignums()
    directory = file.parent / 'plot'
    directory.mkdir(exist_ok=True)
    for num in nums:
        fig = plt.figure(num)
        suffix = f'-{num}' if num > 1 else ''
        out = directory / f'{file.stem}{suffix}.png'
        print(f'runexamples.py: write {out}')
        fig.savefig(out)
