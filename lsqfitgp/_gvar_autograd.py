import builtins

import autograd
import gvar
from autograd import numpy as np

__doc__ = """
Module that replaces gvar.numpy with autograd.numpy.
"""

def switch_numpy(module):
    oldmodule = gvar.numpy
    gvar.numpy = module 
    for s in [
        '_bufferdict', '_gvarcore', '_svec_smat', '_utilities', 
        'cspline', 'dataset', 'linalg', 'ode', 'powerseries','root'
        ]:
        if not hasattr(gvar, s):
            print('*** {} not found'.format(s))
            continue 
        gvar_s = getattr(gvar, s)
        if hasattr(gvar_s, 'numpy'):
            gvar_s.numpy = module 
        else:
            print('*** no numpy in', s)
    return oldmodule 

def switch_functions(module):
    oldmodule = gvar.numpy
    gvar.numpy = module 
    for s in [
        'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'fabs',
        'sinh', 'cosh', 'tanh', 'arcsin', 'arccos', 'arctan', 'arctan2',
        'arcsinh', 'arccosh', 'arctanh', 'square',
        ]:
        if not hasattr(gvar, s) or not hasattr(module, s):
            print('*** {} not found'.format(s))
            continue
        setattr(gvar, s, getattr(module, s))
    return oldmodule

switch_numpy(autograd.numpy)
switch_functions(autograd.numpy)

gvar.erf = autograd.extend.primitive(gvar.erf)
erf_jvp = lambda ans, x: lambda g: g * 2 / np.sqrt(np.pi) * np.exp(-x ** 2)
autograd.extend.defvjp(gvar.erf, erf_jvp)
autograd.extend.defjvp(gvar.erf, erf_jvp)

gvar.BufferDict.extension_fcn['log'] = gvar.exp
gvar.BufferDict.extension_fcn['sqrt'] = gvar.square
gvar.BufferDict.extension_fcn['erfinv'] = gvar.erf

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
