"""Import packages only if they are available, including autograd wrappers."""

import importlib
import builtins

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
