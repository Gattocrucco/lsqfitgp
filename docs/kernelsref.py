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
Kernels reference
=================



"""
for kernel in kernels:
    out += f"""\
.. autofunction:: lsqfitgp.{kernel}
"""

print(f'writing to {outputfile}...')
with open(outputfile, 'w') as file:
    file.write(out)
