# lsqfitgp/docs/kernelsref.py
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

"""Generate a file with the list of kernels. They are documented using
autofunction instead of autoclass because they are built using decorators and
autoclass is not good at supporting that."""

outputfile = 'kernelsref.rst'

import inspect
import sys

import numpy as np
from matplotlib import pyplot as plt

sys.path = ['.', '..'] + sys.path
import lsqfitgp as lgp

classes = (lgp.IsotropicKernel, lgp.StationaryKernel, lgp.Kernel     )
titles  = ('Isotropic kernels', 'Stationary kernels', 'Other kernels')

kernels = []
for name, obj in vars(lgp).items():
    if inspect.isclass(obj) and issubclass(obj, lgp.Kernel):
        if obj not in classes:
            kernels.append(name)
kernels.sort()

out = """\
.. file generated automatically by lsqfitgp/docs/kernelsref.py

.. currentmodule:: lsqfitgp

.. _kernels:

Kernels reference
=================

This is a list of all the specific kernels implemented in :mod:`lsqfitgp`.

Kernels are reported with a simplified signature where the positional arguments
are `r` or `r2` if the kernel is isotropic, `delta` if it is stationary, or
`x`, `y` for generic kernels, and with only the keyword arguments specific to
the kernel. All kernels also understand the general keyword arguments of
:class:`Kernel` (or their specific superclass), while there are no positional
arguments when instantiating the kernel and the call signature of instances is
always `x`, `y`.

Example: the kernel :class:`GammaExp` is listed as ``GammaExp(r, gamma=1)``.
This means you could use it this way::

    import lsqfitgp as lgp
    import numpy as np
    kernel = lgp.GammaExp(loc=0.3, scale=2, gamma=1.4)
    x = np.random.randn(100)
    covmat = kernel(x[:, None], x[None, :])

On multidimensional input, isotropic kernels will compute the euclidean
distance. In general non-isotropic kernels will act separately on each
dimension, i.e., :math:`k(x_1,y_1,x_2,y_2) = k(x_1,y_1) k(x_2,y_2)`, apart from
kernels defined in terms of the dot product.

For all isotropic and stationary (i.e., depending only on :math:`x - y`)
kernels :math:`k(x, x) = 1`, and the typical lengthscale is approximately 1 for
default values of the keyword parameters, apart from some specific cases like
:class:`Constant`.

.. warning::

   Some kernels have problems with derivatives. Second derivatives do not work
   for :class:`Circular`, :class:`Maternp`, and :class:`Wendland`. You may
   encounter problems with second derivatives for :class:`CausalExpQuad`,
   :class:`FracBrownian`, :class:`NNKernel`, and :class:`Taylor`.
   :class:`Matern` is not derivable for integer :math:`\\nu`.

.. warning::

    The following kernels are not compatible with the JAX JIT: :class:`Bessel`,
    :class:`Matern`, :class:`Pink`, :class:`Taylor`.

Index
-----
"""

# index of kernels
kernels2 = list(kernels)
kernels = kernels[::-1]
for superclass, title in zip(classes, titles):
    out += f"""
{title}
{'^' * len(title)}
"""
    for i in reversed(range(len(kernels))):
        obj = getattr(lgp, kernels[i])
        if issubclass(obj, superclass):
            out += f"""\
  * :func:`{kernels[i]}`
"""
            del kernels[i]

out += """
Documentation
-------------
"""

class Formula:
    
    def __init__(self, formula):
        self.formula = formula
    
    def __repr__(self):
        return self.formula
    
    def __call__(self, x):
        return eval(self.formula, vars(np), dict(x=x))

meta = dict(
    BagOfWords = dict(skip=True),
    Bessel = dict(range=[0, 10], kwlist=[dict(nu=v) for v in [0, 1, 2, 3]]),
    BrownianBridge = dict(range=[0, 1]),
    Categorical = dict(skip=True),
    Cauchy = dict(kwlist=[dict(alpha=1), dict(alpha=2), dict(beta=10)]),
    CausalExpQuad = dict(kwlist=[dict(alpha=a) for a in [0, 1, 2]]),
    Celerite = dict(kwlist=[dict(B=0), dict(B=1), dict(gamma=5)]),
    Circular = dict(kwlist=[dict(c=c, tau=t) for c, t in [(1/2, 4), (1/2, 10), (1/4, 4)]], range=[0, 2]),
    Constant = dict(skip=True),
    Cos = dict(range=[0, 4 * np.pi]),
    Decaying = dict(range=[0, 2], srange=[0, 5]),
    Fourier = dict(range=[0, 2], kwlist=[dict(n=n) for n in [1, 2, 3]]),
    FracBrownian = dict(kwlist=[dict(H=H, K=K) for H, K in [(0.1, 1), (0.5, 1), (0.9, 1), (0.9, 0.3)]], range=[-5, 5]),
    GammaExp = dict(kwlist=[dict(gamma=g) for g in [0.1, 1, 1.9]]),
    Gibbs = dict(kwlist=[dict(scalefun=Formula('where((0 < x) & (x < 0.1), 0.02, 1)'))], range=[-1, 1]),
    Harmonic = dict(range=[0, 4 * np.pi], kwlist=[dict(Q=Q) for Q in [1/20, 1, 20]]),
    MA = dict(x=np.arange(50), kwlist=[dict(w=w) for w in [
        2 * np.array([1, -1, 1, -1, 1, -1]),
        np.array([5, 4, 3, 2, 1]),
        2 * np.array([1, 1, 1, 1, 1]),
    ]]),
    Matern = dict(kwlist=[dict(nu=v) for v in [0.1, 1, 1.5, 2]]),
    Maternp = dict(kwlist=[dict(p=p) for p in [0, 1, 2]]),
    Log = dict(range=[0, 10]),
    OrnsteinUhlenbeck = dict(range=[0, 3], srange=[0, 10]),
    Periodic = dict(range=[0, 4 * np.pi], kwlist=[dict(outerscale=s) for s in [1, 0.2]]),
    Pink = dict(range=[0, 10], kwlist=[dict(dw=d) for d in [0.1, 1, 10]]),
    Rescaling = dict(skip=True),
    StationaryFracBrownian = dict(kwlist=[dict(H=H) for H in [0.1, 0.5, 0.9]]),
    Taylor = dict(range=[-2, 2]),
    Wendland = dict(range=[0, 2], kwlist=[dict(k=k, alpha=alpha) for k in [0, 2] for alpha in [1, 2]]),
    Wiener = dict(range=[0, 2]),
    WienerIntegral = dict(range=[0, 2]),
)

fig = plt.figure(num='kernelsref', clear=True)

gen = np.random.default_rng(202206281251)

# documentation
for kernel in kernels2:
    out += f"""\
.. autofunction:: {kernel}
"""
    fig.clf()
    ax = fig.subplots()
    
    m = meta.get(kernel, {})
    
    if m.get('skip', False):
        continue

    k = getattr(lgp, kernel)
    if 'x' in m:
        x = m['x']
        l = np.min(x)
        r = np.max(x)
    else:
        l, r = m.get('range', [0, 5])
        x = np.linspace(l, r, 1000)
    
    legend = m.get('kwlist')
    for kw in m.get('kwlist', [{}]):
        covfun = k(**kw)
        label = ', '.join(f'{k} = {v}' for k, v in kw.items())
        if issubclass(k, lgp.StationaryKernel):
            acf = covfun(x[0], x)
            ax.plot(x, acf, label=label)
        else:
            cov = covfun(x[None, :], x[:, None])
            dx = (x[1] - x[0]) / 2
            vmin = min(0, np.min(cov))
            im = ax.imshow(cov, aspect='equal', origin='lower', vmin=vmin, extent=(x[0] - dx, x[-1] + dx, x[0] - dx, x[-1] + dx))
            legend = False
            break
    
    if legend:
        ax.legend()
    ax.set_title('Covariance function')
    ax.set_xlabel('x')
    if issubclass(k, lgp.StationaryKernel):
        ax.set_ylabel(f'Cov[f(x), f({l})]')
        b, t = ax.get_ylim()
        dy = 0.1
        lim = max([abs(b), abs(t), 1 + dy])
        ax.set_ylim(-lim, lim)
        ax.axvspan(l, r, -1, 0.5, color='#ddd')
        ax.set_xlim(l, r)
    else:
        ax.set_ylabel("x'")
        fig.colorbar(im, label="Cov[f(x), f(x')]")
    figname = f'kernelsref-{kernel}.png'
    fig.savefig(figname)
    
    out += f"""\
.. image:: {figname}
"""
    
    fig.clf()
    ax = fig.subplots()

    if 'srange' in m:
        l, r = m['srange']
        x = np.linspace(l, r, len(x))
    
    nsamples = 1 if m.get('kwlist', False) else 2
    for kw in m.get('kwlist', [{}]):
        covfun = k(**kw)
        label = ', '.join(f'{k} = {v}' for k, v in kw.items())
        cov = covfun(x[None, :], x[:, None])
        try:
            dec = lgp._linalg.CholGersh(cov)
        except np.linalg.LinAlgError:
            dec = lgp._linalg.EigCutFullRank(cov)
        iid = gen.standard_normal(x.shape + (nsamples,))
        samples = dec.correlate(iid)
        for j, y in enumerate(samples.T):
            ax.plot(x, y, label=None if j else label)
    
    if m.get('kwlist'):
        ax.legend()
    ax.set_title('Samples')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    figname = f'kernelsref-{kernel}-samples.png'
    fig.savefig(figname)

    out += f"""\
.. image:: {figname}
"""

print(f'writing to {outputfile}...')
with open(outputfile, 'w') as file:
    file.write(out)
