# lsqfitgp/_Kernel/_where.py
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

from jax import numpy as jnp

from . import _kernel

def where(condfun, kernel1, kernel2):
    """

    Make the kernel of the choice between two independent processes.
    
    Make a kernel(x, y) that yields:
    
      * ``kernel1(x, y)`` where ``condfun(x) & condfun(y)``
    
      * ``kernel2(x, y)`` where ``~condfun(x) & ~condfun(y)``
    
      * zero where ``condfun(x) ^ condfun(y)``
    
    Parameters
    ----------
    condfun : callable
        Function that is applied on an array of points and must return
        a boolean array with the same shape.
    kernel1 : Kernel
        Kernel of the process used where `condfun` yields ``True``.
    kernel2 : Kernel
        Kernel of the process used where `condfun` yields ``False``.
    
    Returns
    -------
    kernel : Kernel
        The conditional kernel.
    
    """
    assert isinstance(kernel1, _kernel.Kernel)
    assert isinstance(kernel2, _kernel.Kernel)
    assert callable(condfun)

    k1 = kernel1._core
    k2 = kernel2._core
        
    def core(x, y):        
        xcond = condfun(x)
        ycond = condfun(y)
        r = jnp.where(xcond & ycond, k1(x, y), k2(x, y))
        return jnp.where(xcond ^ ycond, 0, r)
    
    return _kernel.Kernel(core)

# TODO add a function `choose` to extend `where`. Interface:
# choose(keyfun, mapping)
# example where `comp` is an integer field selecting the kernel:
# choose(lambda comp: comp, [kernel0, kernel1, kernel2, ...], dim='comp')
# example where `comp` is a string field, and without using `dim`:
# choose(lambda x: x['comp'], {'a': kernela, 'b': kernelb})

# TODO consider making an extension of `transf` that allows multi-kernel
# operations. where is a linop.
