.. currentmodule:: lsqfitgp

.. _kernels:

Kernels reference
=================

This is a list of all the specific kernels implemented in :mod:`lsqfitgp`.

Kernels are reported with a simplified signature where the positional arguments
are `r` or `r2` if the kernel is isotropic, or `x`, `y` if it isn't, and with
only the keyword arguments specific to the kernel. All kernels also understand
the general keyword arguments of :class:`Kernel` or :class:`IsotropicKernel`,
while there are no positional arguments when instantiating the kernel and the
call signature of instances is always `x`, `y`.

Example: the kernel :class:`GammaExp` is listed as ``GammaExp(r, gamma=1)``.
This means you could use it this way::

    import lsqfitgp as lgp
    import numpy as np
    kernel = lgp.GammaExp(loc=0.3, scale=2, gamma=1.4)
    x = np.random.randn(100)
    covmat = kernel(x[:, None], x[None, :])

On multidimensional input, isotropic kernels will compute the euclidean
distance. In general non-isotropic kernels will act separately on each
dimension, i.e. :math:`k(x_1,y_1,x_2,y_2) = k(x_1,y_1) k(x_2,y_2)`, apart from
kernels defined in terms of the dot product.

For all isotropic kernels :math:`k(x, x) = 1`, and the typical lengthscale
is approximately 1 for default values of the keyword parameters, apart from
some specific cases like :class:`Constant`.

.. warning::

    Taking second or higher order derivatives might give problems with
    isotropic kernels with signature parameter `r`, while those with `r2` won't
    have any issue.

.. autofunction:: Categorical
.. autofunction:: Constant
.. autofunction:: Cos
.. autofunction:: ExpQuad
.. autofunction:: Fourier
.. autofunction:: FracBrownian
.. autofunction:: GammaExp
.. autofunction:: Gibbs
.. autofunction:: Linear
.. autofunction:: Matern
.. autofunction:: Matern12
.. autofunction:: Matern32
.. autofunction:: Matern52
.. autofunction:: NNKernel
.. autofunction:: PPKernel
.. autofunction:: Periodic
.. autofunction:: Polynomial
.. autofunction:: RatQuad
.. autofunction:: Rescaling
.. autofunction:: Taylor
.. autofunction:: White
.. autofunction:: Wiener
.. autofunction:: WienerIntegral
