import pathlib
import gzip
import pickle
import json
import functools
import collections
import types

import numpy as np

USE_JSON = True

class JSONEncoder(json.JSONEncoder):
    
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, complex):
            return dict(__class__='complex', args=(obj.real, obj.imag))
        return super().default(obj)

def object_hook(obj):
    if '__class__' in obj:
        if obj['__class__'] == 'complex':
            return complex(*obj['args'])
    return obj

def key_to_str(key):
    # I'm hoping that the json-encoded string does not change from the oldest
    # python version I support onwards.
    return json.dumps(key, cls=JSONEncoder).replace('9.999999999999999e-101', '1e-100')

if USE_JSON:

    def load(path, *args, **kw):
        with gzip.open(path, 'rt') as file:
            return json.load(file, *args, object_hook=object_hook, **kw)
    
    def dump(obj, path, *args, **kw):
        with gzip.open(path, 'wt') as file:
            return json.dump(obj, file, cls=JSONEncoder, **kw)

else:

    # This breaks on Windows because it can not instantiate pathlib.PosixPath,
    # despite the fact that I pickle a dict[bytes, float|complex]. A workaround
    # would be to make a context manager (see contextlib for a convenient
    # decorator) that temporarily replaces pathlib.PosixPath with WindowsPath.

    def load(path, *args, **kw):
        with gzip.open(path, 'rb') as file:
            return pickle.load(file, *args, **kw)
    
    def dump(obj, path, *args, **kw):
        with gzip.open(path, 'wb') as file:
            return pickle.dump(obj, file, protocol=5, **kw)
    
class CommittedDict:
    """
    Dictionary backed by a file under version control.

    If `path` is an existing file, read the dict contents from the file, the
    dict is immutable, and `save` has no effect.

    If `path` does not exist, the dict is mutable, and the dict contents will be
    written to the file when calling `save`.
    """

    def __new__(cls, path):
        path = pathlib.Path(path)
        if path.exists():
            contents = load(path)
            return ImmutableCommittedDict(contents)
        else:
            return MutableCommittedDict(path)

class MutableCommittedDict(dict):

    def __init__(self, path):
        super().__init__()
        self.path = path

    def save(self):
        dump(self, self.path)

class ImmutableCommittedDict(collections.UserDict):

    def __init__(self, contents):
        super().__init__()
        self.data = types.MappingProxyType(contents)

    def save(self):
        pass

class cache:
    """
    Decorator to cache function results with a dict.

    The dictionary `cachedict` is used to cache the results of the decorated
    function. The function name and arguments are serialized to a string to be
    used as dictionary keys. The same dictionary can be used for multiple
    functions if the functions have different names.

    Example:

    >>> cache = CommittedDict('cache.json.gz')
    >>> @cache(cache)
    >>> def func(a, b):
    >>>     return a + b
    """

    def __init__(self, cachedict):
        self.cachedict = cachedict

    def __call__(self, func):

        @functools.wraps(func)
        def wrapper(*args, **kw):
            key = (func.__name__, args, kw)
            key = key_to_str(key)
            try:
                return self.cachedict[key]
            except KeyError:
                result = func(*args, **kw)
                self.cachedict[key] = result
                return result
        
        return wrapper
