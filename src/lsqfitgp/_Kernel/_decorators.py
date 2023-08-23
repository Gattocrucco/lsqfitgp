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

from . import _kernel
from . import _stationary
from . import _isotropic

def _makekernelsubclass(kernel, superclass, **prekw):
    assert issubclass(superclass, _kernel.Kernel)
    
    if hasattr(kernel, 'pyfunc'): # np.vectorize objects
        named_object = kernel.pyfunc
    else:
        named_object = kernel
    
    name = getattr(named_object, '__name__', 'DecoratedKernel')
    newclass = types.new_class(name, (superclass,))

    # TODO use the exec_body parameter of new_class instead of doing everything
    # after creation, since an eventual __init_subclass__ wouldn't see the
    # definitions.
        
    prekwset = set(prekw)
    def __new__(cls, **kw):
        kwargs = prekw.copy()
        shared_keys = prekwset & set(kw)
        if shared_keys:
            msg = 'overriding init argument(s) ' + ', '.join(shared_keys)
            msg += ' of kernel ' + name
            warnings.warn(msg)
        kwargs.update(kw)
        return super(newclass, cls).__new__(cls, kernel, **kwargs)
    
    newclass.__new__ = __new__
    newclass.__wrapped__ = named_object
    newclass.__doc__ = named_object.__doc__
    newclass.__qualname__ = getattr(named_object, '__qualname__', name)
    
    # TODO functools.wraps raised an error, but maybe functools.update_wrapper
    # with appropriate arguments would work even on a class.
    
    return newclass

def _kerneldecoratorimpl(cls, *args, **kw):
    functional = lambda kernel: _makekernelsubclass(kernel, cls, **kw)
    if len(args) == 0:
        return functional
    elif len(args) == 1:
        return functional(*args)
    else:
        raise ValueError(len(args))

def kernel(*args, **kw):
    """
    
    Decorator to convert a function to a subclass of `Kernel`. Example::
    
        @kernel(loc=10) # the default loc will be 10
        def MyKernel(x, y, cippa=1, lippa=42):
            return cippa * (x * y) ** lippa
    
    """
    return _kerneldecoratorimpl(_kernel.Kernel, *args, **kw)

def stationarykernel(*args, **kw):
    """
    
    Decorator to convert a function to a subclass of `StationaryKernel`.
    Example::
    
        @stationarykernel(input='soft')
        def MyKernel(absdelta, cippa=1, lippa=42):
            return cippa * sum(
                np.exp(-absdelta[name] / lippa)
                for name in absdelta.dtype.names
            )
    
    """
    return _kerneldecoratorimpl(_stationary.StationaryKernel, *args, **kw)

def isotropickernel(*args, **kw):
    """
    
    Decorator to convert a function to a subclass of `IsotropicKernel`.
    Example::
    
        @isotropickernel(derivable=True)
        def MyKernel(distsquared, cippa=1, lippa=42):
            return cippa * np.exp(-distsquared) + lippa
    
    """
    return _kerneldecoratorimpl(_isotropic.IsotropicKernel, *args, **kw)
