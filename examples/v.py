# lsqfitgp/examples/v.py
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

"""

                            EXAMPLE V.

    Where we go on an expedition to survey the many and wondrous
    kernels that inhabit our software.

"""

import lsqfitgp as lgp
from matplotlib import pyplot as plt
import numpy as np

fig, ax = plt.subplots(num='v', clear=True)

kernels = [
    ['expquad', lgp.ExpQuad()],
    ['cos', lgp.ExpQuad(scale=3) * lgp.Cos()],
    ['wiener', lgp.Wiener()],
    ['fb1/2', lgp.FracBrownian()],
    ['fb1/10', lgp.FracBrownian(H=1/10)],
    ['fb9/10', lgp.FracBrownian(H=9/10)],
    ['fb0.99', lgp.FracBrownian(H=99/100)],
    ['NN', lgp.NNKernel(loc=10)],
    ['Zeta(nu=0.5)', lgp.Zeta(nu=0.5, scale=10)],
    ['Zeta(nu=1.5)', lgp.Zeta(nu=1.5, scale=10)],
    ['Celerite', lgp.Celerite(gamma=0.1)],
    ['Harmonic', lgp.Harmonic(Q=10)],
    # ['BrownianBridge', lgp.BrownianBridge()],
    # ['Taylor', lgp.Taylor()]
]

for label, kernel in kernels:
    gp = lgp.GP(kernel)
    x = np.linspace(0, 20, 500)
    gp.addx(x, 'x')
    cov = gp.prior(raw=True)['x', 'x']
    samples = np.random.multivariate_normal(np.zeros_like(x), cov, size=1)
    ax.plot(x, samples.T, label=label)

ax.legend(loc='best')
fig.show()
