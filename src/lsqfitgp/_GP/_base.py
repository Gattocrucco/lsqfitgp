# lsqfitgp/_GP/_base.py
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

import functools

import jax
from jax import numpy as jnp

from .. import _jaxext
from .. import _utils

class GPBase:

    def __init__(self, *, checkfinite=True, checklin=True):
        self._checkfinite = bool(checkfinite)
        self._checklin = bool(checklin)

    def _clone(self):
        newself = object.__new__(self.__class__)
        newself._checkfinite = self._checkfinite
        newself._checklin = self._checklin
        return newself

    class _SingletonMeta(type):
    
        def __repr__(cls):
            return cls.__name__

    class _Singleton(metaclass=_SingletonMeta):
    
        def __new__(cls):
            raise NotImplementedError(f"{cls.__name__} can not be instantiated")

    class DefaultProcess(_Singleton):
        """ Key of the default process. """
        pass

    def _checklinear(self, func, inshapes, elementwise=False):
        
        # Make input arrays.
        rkey = jax.random.PRNGKey(202206091600)
        inp = []
        for shape in inshapes:
            rkey, subkey = jax.random.split(rkey)
            inp.append(jax.random.normal(subkey, shape))
        
        # Put zeros into the arrays to check they are preserved.
        if elementwise:
            shape = jnp.broadcast_shapes(*inshapes)
            rkey, subkey = jax.random.split(rkey)
            zeros = jax.random.bernoulli(subkey, 0.5, shape)
            for i, a in enumerate(inp):
                inp[i] = a.at[zeros].set(0)
            
        # Compute JVP and check it is identical to the function itself.
        with _jaxext.skipifabstract():
            out0, out1 = jax.jvp(func, inp, inp)
            if out1.dtype == jax.float0:
                cond = jnp.allclose(out0, 0)
            else:
                cond = jnp.allclose(out0, out1)
            if not cond:
                raise RuntimeError('the transformation is not linear')
        
            # Check that the function is elementwise.
            if elementwise:
                if out0.shape != shape or not (jnp.allclose(out0[zeros], 0) and jnp.allclose(out1[zeros], 0)):
                    raise RuntimeError('the transformation is not elementwise')
    
def newself(meth):
    """ Decorator to create a new GP object and pass it to the method. """

    @functools.wraps(meth)
    def newmeth(self, *args, **kw):
        self = self._clone()
        meth(self, *args, **kw)
        return self

    # append return value description to docstring
    doctail = """\
    Returns
    -------
    gp : GP
        A new GP object with the applied modifications.
    """
    newmeth.__doc__ = _utils.append_to_docstring(meth.__doc__, doctail)

    return newmeth
