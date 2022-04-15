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

sys.path = ['.', '..'] + sys.path
import lsqfitgp as lgp

kernels = []
for name, obj in vars(lgp).items():
    if inspect.isclass(obj) and issubclass(obj, lgp.Kernel):
        if obj not in (lgp.Kernel, lgp.IsotropicKernel):
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
are `r` or `r2` if the kernel is isotropic, or `x`, `y` if it isn't, and with
only the keyword arguments specific to the kernel. All kernels also understand
the general keyword arguments of :class:`Kernel` or :class:`IsotropicKernel`,
while there are no positional arguments when instantiating the kernel and the
call signature of instances is always `x`, `y`.

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

   Taking second or higher order derivatives might give problems with the
   :class:`Fourier` kernel and isotropic kernels with signature parameter `r`,
   while those with `r2` won't have any issue.

Index
-----
"""

# index of kernels
for kernel in kernels:
    out += f"""\
  * :func:`{kernel}`
"""
out += """
Documentation
-------------
"""

# documentation
for kernel in kernels:
    out += f"""\
.. autofunction:: {kernel}
"""

print(f'writing to {outputfile}...')
with open(outputfile, 'w') as file:
    file.write(out)
