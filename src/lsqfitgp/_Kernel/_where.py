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

def where(condfun, kernel1, kernel2, dim=None):
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
    dim : str or None
        If specified, when the input arrays are structured, `condfun` is
        applied only to the field `dim`. If the field has a shape, the
        array passed to `condfun` still has `dim` as explicit field.
    
    Returns
    -------
    kernel : Kernel
        If both `kernel1` and `kernel2` are `IsotropicKernel`, the class is
        `IsotropicKernel`.
    
    """
    assert isinstance(kernel1, _kernel.Kernel)
    assert isinstance(kernel2, _kernel.Kernel)
    assert callable(condfun)
    
    assert isinstance(dim, (str, type(None)))
    if isinstance(dim, str):
        def transf(x):
            if x.dtype.names is None:
                raise ValueError('kernel called on non-structured array but condition dim="{}"'.format(dim))
            elif x.dtype.fields[dim][0].shape:
                # TODO should probably use subdtype, such that when the user
                # explicitly specifies an empty shape the behaviour is the
                # same as with nontrivial shapes. This applies also to Kernel.
                return x[[dim]]
            else:
                return x[dim]
        condfun = lambda x, condfun=condfun: condfun(transf(x))
    
    def kernel_op(k1, k2):
        def kernel(xy):
            # TODO this is inefficient, kernels should be computed only on
            # the relevant points. To support this with autograd, make a
            # custom np.where that uses assignment and define its vjp.
            
            # TODO this may produce a very sparse matrix, when I implement
            # sparse support do it here too.
            
            # TODO it will probably often be the case that the result is
            # either dense or all zero. The latter case can be optimized this
            # way: if it's zero, broadcast a 0-d array to the required shape,
            # and flag it as all zero with an instance variable.
            
            x, y = xy
            xcond = condfun(x)
            ycond = condfun(y)
            r = jnp.where(xcond & ycond, k1(xy), k2(xy))
            return jnp.where(xcond ^ ycond, 0, r)
        return kernel
    
    return kernel1._binary(kernel2, kernel_op)

# TODO add a function `choose` to extend `where`. Interface:
# choose(keyfun, mapping)
# example where `comp` is an integer field selecting the kernel:
# choose(lambda comp: comp, [kernel0, kernel1, kernel2, ...], dim='comp')
# example where `comp` is a string field, and without using `dim`:
# choose(lambda x: x['comp'], {'a': kernela, 'b': kernelb})

# TODO consider making an extension of `transf` that allows multi-kernel
# operations. where is a linop.
#
# Class rules: The condition for trying to make it a Kernel is that all
# involved kernels must be instances of Kernel.
