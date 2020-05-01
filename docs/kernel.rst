.. currentmodule:: lsqfitgp

Generic kernel classes
======================

All kernels in :mod:`lsqfitgp` are subclasses of :class:`Kernel` or
:class:`IsotropicKernel` (which is itself a subclass of :class:`Kernel`).

The two general classes can be used directly by instantiating them with a
callable which will do the actual computation. However, this can be done in a
simpler and more functional way using the decorators :func:`kernel` and
:func:`isotropickernel`.

Kernel
------

.. autoclass:: Kernel
    :inherited-members: diff
    
IsotropicKernel
---------------

.. autoclass:: IsotropicKernel
