# lsqfitgp/_gvar_autograd.py
#
# Copyright (c) 2020, Giacomo Petrillo
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

import builtins

# do not use _imports here, because _imports imports this module
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
