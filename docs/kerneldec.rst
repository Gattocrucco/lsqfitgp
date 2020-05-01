.. currentmodule:: lsqfitgp

.. _kerneldec:

Kernel decorators
=================

These decorators convert a callable to a subclass of :class:`Kernel`. The
decorators can be used both with and without keyword arguments. The keyword
arguments will be passed as initialization arguments to the superclass, and
will be overwritten by keyword arguments given at initialization of the
subclass.

.. autofunction:: kernel

.. autofunction:: isotropickernel
