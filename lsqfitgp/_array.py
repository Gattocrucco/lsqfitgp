import builtins

from autograd import numpy as np
from autograd.builtins import isinstance

__all__ = [
    'StructuredArray'
]

def _readonlyview(x):
    if not builtins.isinstance(x, (StructuredArray, np.numpy_boxes.ArrayBox)):
        x = x.view()
        x.flags['WRITEABLE'] = False
    return x

def _wrapifstructured(x):
    if x.dtype.names is None:
        return x
    else:
        return StructuredArray(x)

def _broadcast_shapes_2(s1, s2):
    assert isinstance(s1, tuple)
    assert isinstance(s2, tuple)
    if len(s1) < len(s2):
        s1 = (len(s2) - len(s1)) * (1,) + s1
    elif len(s2) < len(s1):
        s2 = (len(s1) - len(s2)) * (1,) + s2
    out = ()
    for a, b in zip(s1, s2):
        if a == b:
            out += (a,)
        elif a == 1 or b == 1:
            out += (a * b,)
        else:
            raise ValueError('can not broadcast shape {} with {}'.format(s1, s2))
    return out

def broadcast_shapes(shapes):
    """
    Return the broadcasted shape from a list of shapes.
    """
    out = ()
    for shape in shapes:
        try:
            out = _broadcast_shapes_2(out, shape)
        except ValueError:
            msg = 'can not broadcast shapes '
            msg += ', '.join(str(s) for s in shapes)
            raise ValueError(msg)
    return out

class broadcast:
    """
    Version of np.broadcast that works with StructuredArray.
    """
    
    def __init__(self, *arrays):
        shapes = [a.shape for a in arrays]
        self.shape = broadcast_shapes(shapes)

def broadcast_to(x, shape, **kw):
    """
    Version of np.broadcast_to that works with StructuredArray.
    """
    if isinstance(x, StructuredArray):
        return x.broadcast_to(shape, **kw)
    else:
        return np.broadcast_to(x, shape, **kw)

def broadcast_arrays(*arrays, **kw):
    """
    Version of np.broadcast_arrays that works with StructuredArray.
    """
    shapes = [a.shape for a in arrays]
    shape = broadcast_shapes(shapes)
    return tuple(broadcast_to(a, shape, **kw) for a in arrays)

class StructuredArray:
    """
    Autograd-friendly imitation of a numpy structured array. It behaves like
    a read-only numpy array, with the exception that you can set a whole field.
    Example:
    
    >>> a = np.empty(3, dtype=[('f', float), ('g', float)])
    >>> a = StructuredArray(a)
    >>> a['f'] = np.arange(3) # this is allowed
    >>> a[1] = (0.3, 0.4) # this raises an error
    """
    
    @classmethod
    def _fromarrayanddict(cls, x, d):
        out = super().__new__(cls)
        out.dtype = x.dtype
        out._dict = d
        f0 = x.dtype.names[0]
        a0 = d[f0]
        subshape = x.dtype.fields[f0][0].shape
        out.shape = a0.shape[:len(a0.shape) - len(subshape)]
        out.size = np.prod(out.shape)
        return out
    
    def __new__(cls, array):
        assert isinstance(array, (np.ndarray, cls))
        assert array.dtype.names is not None
        d = {
            name: _readonlyview(_wrapifstructured(array[name]))
            for name in array.dtype.names
        }
        return cls._fromarrayanddict(array, d)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self._dict[key]
        elif isinstance(key, list) and all(isinstance(k, str) for k in key):
            d = {
                name: self._dict[name]
                for name in key
            }
        else:
            d = {
                name: x[key]
                for name, x in self._dict.items()
            }
        return type(self)._fromarrayanddict(self, d)
    
    def __setitem__(self, key, val):
        assert key in self.dtype.names
        assert isinstance(val, (np.ndarray, StructuredArray))
        prev = self._dict[key]
        # TODO support casting and broadcasting
        assert prev.dtype == val.dtype
        assert prev.shape == val.shape
        self._dict[key] = _readonlyview(val)
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        d = {
            name: x.reshape(shape + self.dtype.fields[name][0].shape)
            for name, x in self._dict.items()
        }
        return type(self)._fromarrayanddict(self, d)
    
    def broadcast_to(self, shape, **kw):
        _broadcast_shapes_2(self.shape, shape) # raises if not broadcastable
        d = {
            name: broadcast_to(x, shape + self.dtype.fields[name][0].shape, **kw)
            for name, x in self._dict.items()
        }
        return type(self)._fromarrayanddict(self, d)
