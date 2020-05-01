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

.. warning::

    Taking second or higher order derivatives might give problems with
    isotropic kernels with signature parameter `r`, while those with `r2` won't
    have any issue.

"""
for kernel in kernels:
    out += f"""\
.. autofunction:: {kernel}
"""

print(f'writing to {outputfile}...')
with open(outputfile, 'w') as file:
    file.write(out)
