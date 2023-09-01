# lsqfitgp/_Kernel/_kernel.py
#
# Copyright (c) 2020, 2022, 2023, Giacomo Petrillo
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

from . import _util
from . import _crosskernel

class Kernel(_crosskernel.CrossKernel):
    r"""

    Subclass of `CrossKernel` to represent the kernel of a single function:

    .. math::
        \mathrm{kernel}(x, y) = \mathrm{Cov}[f(x), f(y)].

    """

    def _swap(self):
        return self

Kernel.inherit_transf('xtransf')
Kernel.inherit_transf('diff')
# other transformations are registered by IsotropicKernel

@Kernel.register_transf
def forcekron(tcls, self):
    r"""
    
    Force the kernel to be a separate product over dimensions:

    .. math::
        \mathrm{newkernel}(x, y) = \prod_i \mathrm{kernel}(x_i, y_i)
    
    Returns
    -------
    newkernel : Kernel
        The transformed kernel.

    """

    core = self.core
    newcore = lambda x, y, **kw: _util.prod_recurse_dtype(core, x, y, **kw)
    return self._clone(tcls, core=newcore)

_crosskernel.Kernel = Kernel
