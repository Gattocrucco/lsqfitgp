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

from . import _crosskernel
from . import _kernel
from . import _stationary
from . import _isotropic

def _makekernelsubclass(kernel, superclass, **prekw):

    # get cross superclass
    assert issubclass(superclass, _kernel.Kernel)
    cross_superclass = next(
        c for c in superclass.mro()
        if not issubclass(c, _kernel.Kernel)
    )
    
    if hasattr(kernel, 'pyfunc'): # np.vectorize objects
        named_object = kernel.pyfunc
    else:
        named_object = kernel
    
    name = getattr(named_object, '__name__', 'DecoratedKernel')

    newcrossclass = types.new_class('Cross' + name, (cross_superclass,))

    def exec_body(ns):
        
        prekwset = set(prekw)
        def __new__(cls, **kw):
            kwargs = prekw.copy()
            shared_keys = prekwset & set(kw)
            if shared_keys:
                warnings.warn(f'overriding init argument(s) '
                    f'{", ".join(shared_keys)} of kernel {name}, subclass '
                    'may not be preserved')
            kwargs.update(kw)
            self = super(newclass, cls).__new__(cls, kernel, **kwargs)
            if not shared_keys:
                self = self._clone(cls=cls)
            return self
        
        ns['__new__'] = __new__
        ns['__wrapped__'] = named_object
        ns['__doc__'] = named_object.__doc__

    newclass = types.new_class(name, (newcrossclass, superclass), exec_body=exec_body)

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
    
    Decorator to convert a function to a subclass of `Kernel`.

    Parameters
    ----------
    *args :
        Either a function to decorate, or no arguments.
    **kw :
        Additional arguments are passed to `Kernel`.

    Returns
    -------
    class_or_dec : callable or class
        If `args` is empty, a decorator ready to be applied, else the kernel
        class.

    Examples
    --------
    
    >>> @lgp.kernel(loc=10) # the default loc will be 10
    ... def MyKernel(x, y, cippa=1, lippa=42):
    ...     return cippa * (x * y) ** lippa

    See also
    --------
    stationarykernel, isotropickernel

    Notes
    -----
    The decorator also creates a class hierarchy on top of the new class.
    The first non-`Kernel`-subclass superclass of the the target superclass is
    used as superclass of a new class which is the first base of the returned
    class. The second base is the target superclass. In code, this::

        @lgp.<kind>kernel
        def Pino(x, y):
            return 0

    is equivalent to::

        class CrossPino(lgp.Cross<kind>Kernel):
            pass

        class Pino(CrossPino, lgp.<kind>Kernel):
            def __new__(cls, **kw):
                return super().__new__(cls, lambda x, y: 0, **kw)
    
    """
    return _kerneldecoratorimpl(_kernel.Kernel, *args, **kw)

def stationarykernel(*args, **kw):
    """
    
    Like `kernel` but makes a subclass of `StationaryKernel`.

    Examples
    --------
    
    >>> @lgp.stationarykernel(input='soft')
    ... def MyKernel(absdelta, cippa=1, lippa=42):
    ...     return cippa * sum(
    ...         jnp.exp(-absdelta[name] / lippa)
    ...         for name in absdelta.dtype.names
    ...     )
    
    """
    return _kerneldecoratorimpl(_stationary.StationaryKernel, *args, **kw)

def isotropickernel(*args, **kw):
    """
    
    Like `kernel` but makes a subclass of `IsotropicKernel`.

    Examples
    --------
    
    >>> @lgp.isotropickernel(derivable=True)
    ... def MyKernel(distsquared, cippa=1, lippa=42):
    ...     return cippa * jnp.exp(-distsquared) + lippa
    
    """
    return _kerneldecoratorimpl(_isotropic.IsotropicKernel, *args, **kw)
