# lsqfitgp/_Kernel/_decorators.py
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

import types
import warnings
import inspect

from . import _crosskernel
from . import _kernel
from . import _stationary
from . import _isotropic

def makekernelsubclass(core, bases, **prekw):

    named_object = getattr(core, 'pyfunc', core) # np.vectorize objects
    name = getattr(named_object, '__name__', 'DecoratedKernel')

    bases = tuple(bases)

    def exec_body(ns):
        
        def __new__(cls, **kw):
            kwargs = prekw.copy()
            kwargs.update(kw)
            if len(kwargs) < len(prekw) + len(kw):
                shared_keys = set(prekw).intersection(kw)
                warnings.warn(f'overriding init argument(s) '
                    f'{shared_keys} of kernel {name}')
            self = super(newclass, cls).__new__(cls, core, **kwargs)
            if isinstance(self, bases[-1]) and set(kw).issubset(self.initkw):
                self = self._clone(cls)
            return self
        
        ns['__new__'] = __new__
        ns['__wrapped__'] = named_object
        ns['__doc__'] = named_object.__doc__

    newclass = types.new_class(name, bases, exec_body=exec_body)
    assert issubclass(newclass, _crosskernel.CrossKernel)
    return newclass

def crosskernel(*args, bases=None, **kw):
    """
    
    Decorator to convert a function to a subclass of `CrossKernel`.

    Parameters
    ----------
    *args :
        Either a function to decorate, or no arguments. The function is used
        as the `core` argument to `CrossKernel`.
    bases : tuple of types, optional
        The bases of the new class. If not specified, use `CrossKernel`.
    **kw :
        Additional arguments are passed to `CrossKernel`.

    Returns
    -------
    class_or_dec : callable or type
        If `args` is empty, a decorator ready to be applied, else the kernel
        class.

    Examples
    --------
    
    >>> @lgp.crosskernel(derivable=True)
    ... def MyKernel(x, y, a=0, b=0):
    ...     return (x - a) * (y - b)

    Notes
    -----
    Arguments passed to the class constructor may modify the class. If the
    object returned by the the constructor is a subclass of the superclass
    targeted by the decorator, and all the arguments passed at instantiation
    are passed down to the decorated function, the class of the object is
    enforced to be the new class.
    
    """
    if bases is None:
        bases = _crosskernel.CrossKernel,
    functional = lambda core: makekernelsubclass(core, bases, **kw)
    if len(args) == 0:
        return functional
    elif len(args) == 1:
        return functional(*args)
    else:
        raise ValueError(len(args))

def kernel(*args, **kw):
    """
    
    Like `crosskernel` but makes a subclass of `Kernel`.

    Examples
    --------
    
    >>> @lgp.kernel(loc=10) # the default loc will be 10
    ... def MyKernel(x, y, cippa=1, lippa=42):
    ...     return cippa * (x * y) ** lippa
    
    """
    return crosskernel(*args, bases=(_kernel.Kernel,), **kw)

def crossstationarykernel(*args, **kw):
    """
    
    Like `crosskernel` but makes a subclass of `CrossStationaryKernel`.
    
    """
    return crosskernel(*args, bases=(_stationary.CrossStationaryKernel,), **kw)

def stationarykernel(*args, **kw):
    """
    
    Like `crosskernel` but makes a subclass of `StationaryKernel`.

    Examples
    --------
    
    >>> @lgp.stationarykernel(input='posabs')
    ... def MyKernel(absdelta, cippa=1, lippa=42):
    ...     return cippa * sum(
    ...         jnp.exp(-absdelta[name] / lippa)
    ...         for name in absdelta.dtype.names
    ...     )
    
    """
    return crosskernel(*args, bases=(_stationary.StationaryKernel,), **kw)

def crossisotropickernel(*args, **kw):
    """
    
    Like `crosskernel` but makes a subclass of `CrossIsotropicKernel`.
    
    """
    return crosskernel(*args, bases=(_isotropic.CrossIsotropicKernel,), **kw)

def isotropickernel(*args, **kw):
    """
    
    Like `crosskernel` but makes a subclass of `IsotropicKernel`.

    Examples
    --------
    
    >>> @lgp.isotropickernel(derivable=True)
    ... def MyKernel(distsquared, cippa=1, lippa=42):
    ...     return cippa * jnp.exp(-distsquared) + lippa
    
    """
    return crosskernel(*args, bases=(_isotropic.IsotropicKernel,), **kw)
