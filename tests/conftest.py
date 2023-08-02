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
def nodepath(request):
    """ List of the node names of the current test """
    node = request.node
    names = []
    while node is not None:
        names.append(node.name)
        node = node.parent
    names = names[::-1]
    assert names[0] == 'lsqfitgp'
    assert names[1].startswith('tests')
    return names

@pytest.fixture
def rng(nodepath):
    """ A random generator with a deterministic per-test seed """
    path = '::'.join(nodepath)
    seed = np.array([path], np.bytes_).view(np.uint8)
    return np.random.default_rng(seed)

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
def cached(nodepath):
    """
    A function that caches the result of a function call in a file.

    The cache is per test (including parametrizations) and is stored in
    a compressed json file. The cache is a dictionary where the keys must be
    specified by the user; the cache does not try to use the function arguments
    to distinguish different invocations.

    If the cache already exists, it is immutable. Delete the file to reset the
    cache.

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

    # root cache directory
    file = pathlib.Path('tests') / 'cached'

    # drop tests prefix from first path component
    comp = pathlib.Path(nodepath[1])
    assert comp.parts[0] == 'tests'
    for name in comp.parts[1:]:
        file = file.joinpath(name)

    # add all other path components
    for name in nodepath[2:-1]:
        comp = pathlib.Path(name)
        file = file.joinpath(comp)

    # drop module extension, add test name and extension
    file = file.with_suffix('').joinpath(nodepath[-1]).with_suffix('.json.gz')

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
