import lsqfitgp as lgp
from matplotlib import pyplot as plt
import numpy as np

fig = plt.figure('v')
fig.clf()
ax = fig.subplots(1, 1)

kernels = [
    ['expquad', lgp.ExpQuad()],
    ['cos', lgp.ExpQuad(scale=3) * lgp.Cos()],
    ['wiener', lgp.Wiener()],
    ['fb1/2', lgp.FracBrownian()],
    ['fb1/10', lgp.FracBrownian(H=1/10)],
    ['fb9/10', lgp.FracBrownian(H=9/10)],
    ['fb0.99', lgp.FracBrownian(H=99/100)],
    ['NN', lgp.NNKernel(loc=10)],
    ['Fourier(n=1)', lgp.Fourier(n=1, scale=10)],
    ['Fourier(n=2)', lgp.Fourier(n=2, scale=10)],
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
