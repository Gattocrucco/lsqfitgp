import pathlib
import functools
import gzip

import numpy as np

USE_JSON = False

if USE_JSON:

    import json

    # open = gzip.open
    # I do not use gzip because it does not support str, only bytes, and json
    # only outputs str. I'd have to write the binding code but I'm not using
    # json right now.

    class JSONEncoder(json.JSONEncoder):
        
        def default(self, obj):
            if isinstance(obj, complex):
                return dict(__class__='complex', args=(obj.real, obj.imag))
            return super().default(obj)

    def object_hook(obj):
        if '__class__' in obj:
            if obj['__class__'] == 'complex':
                return complex(*obj['args'])
        return obj

    def load(path, *args, **kw):
        with open(path, 'r') as file:
            return json.load(file, *args, object_hook=object_hook, **kw)
    
    def dump(obj, path, *args, **kw):
        with open(path, 'w') as file:
            return json.dump(obj, file, cls=JSONEncoder, **kw)
    
    dumps = lambda *args, **kw: json.dumps(*args, cls=JSONEncoder, **kw)

else:

    import pickle

    PROTOCOL = 5
    
    open = gzip.open

    def load(path, *args, **kw):
        with open(path, 'rb') as file:
            return pickle.load(file, *args, **kw)
    
    def dump(obj, path, *args, **kw):
        with open(path, 'wb') as file:
            return pickle.dump(obj, file, protocol=PROTOCOL, **kw)
    
    dumps = lambda *args, **kw: pickle.dumps(*args, protocol=PROTOCOL, **kw)

class DiskCacheDict(dict):
    """
    dict subclass that updates at initialization from a json file
    """

    def __init__(self, path, *args, **kw):
        super().__init__(*args, **kw)
        self.path = pathlib.Path(path)
        if self.path.exists():
            contents = load(self.path)
            self.update(contents)

    def save(self):
        dump(self, self.path)

class diskcachefunc:
    """
    Decorator to cache function results to file. Example:

    cache = DiskCacheDict('cache.json')
    @diskcachefunc(cache)
    def func(a, b):
        return a + b
    """

    def __init__(self, cachedict):
        self.cachedict = cachedict

    def __call__(self, func):

        @functools.wraps(func)
        def wrapper(*args, **kw):
            key = (func.__name__, args, kw)
            key = self.remove_numpy_recursive(key)
                # for json: numpy can not be encoded; for pickle: numpy scalars
                # takes up more space than ordinary types
            key = dumps(key)
            result = self.cachedict.get(key)
            if result is None:
                result = func(*args, **kw)
                self.cachedict[key] = result
            return result
        
        return wrapper

    def remove_numpy_recursive(self, obj):
        if isinstance(obj, dict):
            return obj.__class__({
                k: self.remove_numpy_recursive(v)
                for k, v in obj.items()
            })
        elif isinstance(obj, (list, tuple)):
            return obj.__class__([
                self.remove_numpy_recursive(v)
                for v in obj
            ])
        elif isinstance(obj, np.generic):
            return self.remove_numpy_recursive(obj.item())
        elif isinstance(obj, np.ndarray):
            if obj.shape == ():
                return self.remove_numpy_recursive(obj.item())
            else:
                return self.remove_numpy_recursive(obj.tolist())
        else:
            return obj
