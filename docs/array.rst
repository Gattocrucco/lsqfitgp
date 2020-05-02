.. currentmodule:: lsqfitgp

.. _array:

Structured arrays wrapping
==========================

Taking derivatives on structured numpy arrays is not supported by
:mod:`autograd`, so structured arrays are internally wrapped with
:class:`StructuredArray`.

StructuredArray
---------------

.. autoclass:: StructuredArray

Functions
---------

.. autofunction:: asarray
.. autofunction:: broadcast
.. autofunction:: broadcast_arrays
.. autofunction:: broadcast_shapes
.. autofunction:: broadcast_to
