import lsqfitgp as lgp
from matplotlib import pyplot as plt
import numpy as np

"""Check the symmetry of the Bernoulli polynomials"""

fig, axs = plt.subplots(1, 2, num='bernoulli', clear=True, figsize=[8, 5])

axs[0].set_title('$B_n(x)$')
axs[1].set_title('$B_n(x) - (-1)^n B_n(1-x)$')

eps = np.finfo(float).eps
x = np.linspace(eps, 1 - eps, 1000)

for n in range(2, 11):
    y = lgp._kernels._bernoulli_poly(n, x)
    yrev = (-1) ** n * lgp._kernels._bernoulli_poly(n, 1 - x)
    axs[0].plot(x, y, label=str(n))
    axs[1].plot(x, y - yrev, label=str(n))

for ax in axs:
    ax.legend()

fig.tight_layout()
fig.show()
