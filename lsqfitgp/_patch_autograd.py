"""Add vjp of asarray, fix #552 (creating object arrays from lists)"""

import builtins

import autograd

autograd.extend.defvjp(
    autograd.numpy.asarray,
    lambda ans, *args, **kw: lambda g: g
)

autograd.numpy.numpy_boxes.ArrayBox.item = lambda self: self[(0,) * len(self.shape)]

def array(A, *args, **kwargs):
    t = builtins.type(A)
    if t in (list, tuple):
        return autograd.numpy.numpy_wrapper.array_from_args(args, kwargs, *map(lambda a: a if a.shape else a.item(), map(array, A)))
    else:
        return autograd.numpy.numpy_wrapper._array_from_scalar_or_array(args, kwargs, A)
autograd.numpy.numpy_wrapper.array = array
autograd.numpy.array = array
