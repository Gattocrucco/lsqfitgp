# lsqfitgp/_imports.py
#
# Copyright (c) 2020, 2022, Giacomo Petrillo
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

"""Import packages only if they are available, including autograd wrappers."""

import importlib
import builtins
import collections

def try_import(module):
    try:
        return importlib.import_module(module)
    except (ImportError, ModuleNotFoundError):
        return None

gvar = try_import('gvar')
autograd = try_import('autograd')
sparse = try_import('scipy.sparse')
try_import('scipy.sparse.linalg')
optimize = try_import('scipy.optimize')

if autograd is None:
    import numpy
    scipy = try_import('scipy')
    special = try_import('scipy.special')
    linalg = try_import('scipy.linalg')
    if linalg is None:
        from numpy import linalg
    isinstance = builtins.isinstance
    class dummy:
        pass
    numpy.numpy_boxes = dummy()
    numpy.numpy_boxes.ArrayBox = dummy
    autograd = dummy()
    autograd.extend = dummy()
    autograd.extend.defvjp = lambda *args, **kw: None
    autograd.extend.defjvp = lambda *args, **kw: None
    autograd.extend.primitive = lambda fun: fun
    autograd.wrap_util = dummy()
    autograd.wrap_util.unary_to_nary = lambda fun: fun
else:
    from autograd import numpy
    scipy = try_import('autograd.scipy')
    special = try_import('autograd.scipy.special')
    linalg = try_import('autograd.scipy.linalg')
    if linalg is None:
        linalg = autograd.numpy.linalg
    from . import _patch_autograd
    from . import _gvar_autograd
    isinstance = autograd.builtins.isinstance

if gvar is None:
    class dummy:
        pass
    gvar = dummy()
    gvar.BufferDict = dict

elif hasattr(gvar.SVD, '_analyzers') and scipy is None:
    gvar.SVD._analyzers = collections.OrderedDict([
        ('numpy_eigh', gvar.SVD._numpy_eigh),
        ('numpy_svd', gvar.SVD._numpy_svd) 
    ])

elif hasattr(gvar.SVD, '_analyzers'):
    def scipy_eigh(x):
        w, v = linalg.eigh(x)
        w = numpy.abs(w)
        si = numpy.argsort(w)
        return w[si], v.T[si]
    
    gvar.SVD._analyzers = collections.OrderedDict([
        ('scipy_eigh', scipy_eigh)
    ])
