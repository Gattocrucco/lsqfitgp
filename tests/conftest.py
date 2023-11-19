# lsqfitgp/tests/conftest.py
#
# Copyright (c) 2023, Giacomo Petrillo
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

import pathlib
import json
import gzip

import pytest
import gvar
import numpy as np

@pytest.fixture(autouse=True)
def clean_gvar_env():
    """ Create a new hidden global covariance matrix for primary gvars, restore
    the previous one during teardown. Otherwise the global covariance matrix
    grows arbitrarily. """
    yield gvar.switch_gvar()
    gvar.restore_gvar()

@pytest.fixture
def rng(request):
    """ A random generator with a deterministic per-test seed """
    nodeid = request.node.nodeid
    seed = np.array([nodeid], np.bytes_).view(np.uint8)
    return np.random.default_rng(seed)

@pytest.fixture(autouse=True)
def reset_random_seeds(rng):
    """ Set seeds of global state random generators for tests that still use
    them. Prefer `rng` for new tests. """
    bitgen0 = rng.bit_generator
    bitgen1 = bitgen0.jumped(1)
    bitgen2 = bitgen1.jumped(2)
    def toseed(bitgen):
        return np.array([bitgen.random_raw()], np.uint64).view(np.uint32)
    np.random.seed(toseed(bitgen1))
    gvar.ranseed(toseed(bitgen2))

class JSONEncoder(json.JSONEncoder):
    
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return dict(__class__='array', args=(obj.tolist(),), kw=dict(dtype=obj.dtype.str))
        if isinstance(obj, complex):
            return dict(__class__='complex', args=(obj.real, obj.imag))
        return super().default(obj)

def object_hook(obj):
    if classname := obj.get('__class__'):
        constructor = dict(
            array=np.array,
            complex=complex
        )[classname]
        args = obj.get('args', ())
        kw = obj.get('kw', {})
        return constructor(*args, **kw)
    return obj

@pytest.fixture
def testpath(request):
    """
    - relative Path of the file where the test is defined
    - dotted name of the test, including classes and parametrization
    """
    path = request.node.path
    root = request.session.path
    path = path.relative_to(root)
    name = request.node.location[2]
    assert path.parts[0] == 'tests'
    assert path.suffix == '.py'
    return path, name

@pytest.fixture
def cached(testpath):
    """
    A function that caches the result of a function call in a file.

    The cache is per test (including parametrizations) and is stored in
    a compressed json file. The cache is a dictionary where the keys must be
    specified by the user; the cache does not try to use the function arguments
    to distinguish different invocations.

    If the cache already exists, it is immutable. Delete the file to reset the
    cache. The files are saved under tests/cached.

    Example usage:
    >>> def expensive_function(x):
    >>>     time.sleep(1)
    >>>     return abs(x)
    >>> @pytest.mark.parametrize('arg1,arg2', [(-1, 1), (-2, 2)])
    >>> def test_cippa(arg1, arg2, cached):
    >>>     lippa = cached('lippa', expensive_function, arg1)
    >>>     turlipu = cached('turlipu', expensive_function, arg2)
    >>>     assert lippa == turlipu
    """

    # determine cache file location
    file = pathlib.Path('tests') / 'cached'
    path, name = testpath
    assert path.parts[0] == 'tests'
    for part in path.parts[1:]:
        file = file.joinpath(part)
    file = file.with_suffix('').joinpath(name).with_suffix('.json.gz')

    # if the file exists, load the cache and consider it immutable
    if file.exists():
        with gzip.open(file, 'rt') as stream:
            cache = json.load(stream, object_hook=object_hook)
        def cached(name, func, *args, **kw):
            return cache[name]
        yield cached

    # if the file does not exist, keep the cache in a dictionary, and save it
    # to file on teardown
    else:
        cache = {}
        def cached(name, func, *args, **kw):
            assert isinstance(name, str)
            if name not in cache:
                cache[name] = func(*args, **kw)
            return cache[name]
        yield cached
        
        file.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(file, 'wt') as stream:
            json.dump(cache, stream, cls=JSONEncoder)
