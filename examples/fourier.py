import lsqfitgp as lgp
from matplotlib import pyplot as plt
import numpy as np
import gvar

"""Constraint the values of Fourier series coefficients"""

gp = lgp.GP(lgp.Fourier(n=3), checkpos=False)
gp.addkernelop('fourier', True, 'F')
x = np.linspace(0, 1, 100)
gp.addx(x, 'x')
gp.addx(1, 's1', proc='F')
gp.addx(2, 'c1', proc='F')

comb = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
]

fig, ax = plt.subplots(num='fourier', clear=True)

for s, c in comb:
    y = gp.predfromdata(dict(s1=s, c1=c), 'x')
    m = gvar.mean(y)
    u = gvar.sdev(y)
    pc = ax.fill_between(x, m - u, m + u, alpha=0.5, label=f's{s}c{c}')
    color = pc.get_facecolor()
    for sample in gvar.raniter(y, 3):
        ax.plot(x, sample, color=color)

ax.legend()

fig.show()
