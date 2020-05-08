"""Add vjp of asarray, fix #552 (creating object arrays from lists),
add missing scipy functions."""

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

try:
    from scipy import special as special_noderiv
    from autograd.scipy import special
    
    missing_functions = [
        'bernoulli',
        'binom',
        'kv',
        'kvp',
        'factorial',
        'zeta'
    ]
    for f in missing_functions:
        if not hasattr(special, f):
            wf = autograd.extend.primitive(getattr(special_noderiv, f))
            setattr(special, f, wf)

    autograd.extend.defvjp(
        special.kvp,
        lambda ans, v, z, n: lambda g: g * special.kvp(v, z, n + 1),
        argnums=[1]
    )    

except (ImportError, ModuleNotFoundError):
    pass
    