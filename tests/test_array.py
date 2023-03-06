# lsqfitgp/tests/test_array.py
#
# Copyright (c) 2020, 2022, 2023, Giacomo Petrillo
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

import sys
import itertools

import numpy as np
from numpy.lib import recfunctions
from jax import tree_util
from jax import numpy as jnp
import pytest
import pandas as pd
import polars as pl

sys.path.insert(0, '.')
import lsqfitgp as lgp
from lsqfitgp import _array
import util

rng = np.random.default_rng(202301231524)

def test_ellipsis():
    x = np.empty((2, 3), dtype=[('a', float, 4)])
    y = lgp.StructuredArray(x)
    z = y[..., 0]
    assert z.shape == y.shape[:1]
    assert z['a'].shape == y.shape[:1] + x.dtype.fields['a'][0].shape

def fill_array_at_random(x, rng):
    if x.dtype.names:
        for name in x.dtype.names:
            fill_array_at_random(x[name], rng)
    elif x.dtype == bool:
        x[...] = rng.integers(0, 2, x.shape)
    elif np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        x[...] = rng.integers(info.min, info.max, x.shape, dtype=x.dtype, endpoint=True)
    elif np.issubdtype(x.dtype, np.floating):
        x[...] = 100 * rng.standard_normal(x.shape)
    elif np.issubdtype(x.dtype, np.character):
        basetype = x.dtype.type
        size = x.dtype.itemsize // np.dtype((basetype, 1)).itemsize
        total = x.size * size
        data = rng.integers(1, 128, total, dtype='u1')
        data = data.view((np.bytes_, size))
        data = data.astype((basetype, size))
        x[...] = data.reshape(x.shape)
    elif x.dtype == object:
        for idx in np.ndindex(*x.shape):
            x[idx] = dict(idx=idx)
    else:
        raise NotImplementedError

def random_array(shape, dtype, rng=rng):
    x = np.empty(shape, dtype)
    fill_array_at_random(x, rng)
    return x

def crosscheck_operation(op, *arrays, **kw):
    """
    arrays = structured numpy arrays
    applies op both to arrays and to the conversion to StructuredArrays, then
    converts back to numpy arrays and check the result is the same. Op can be
    a multivalued function.
    """
    
    r1 = op(*arrays, **kw)
    r2 = op(*map(lgp.StructuredArray, map(np.copy, arrays)), **kw)
    
    if not isinstance(r1, (tuple, list)):
        r1 = (r1,)
        r2 = (r2,)
    
    assert len(r1) == len(r2)
    for res1, res2 in zip(r1, r2):
    
        if isinstance(res1, lgp.StructuredArray):
            res1 = np.asarray(res1)
        if isinstance(res2, lgp.StructuredArray):
            res2 = np.asarray(res2)

        util.assert_equal(res1, res2)

def test_broadcast_to():
    op = np.broadcast_to # uses __array_function__
    dtypes = [
        [('f0', float)],
        'f,f',
        'f,d,f',
        'f,O',
        'S25,U10,?',
        [('a', 'f,f', (3, 1)), ('b', [('c', 'd,d'), ('e', '?', 15)])],
    ]
    shapes = [
        (),
        (0,),
        (1,),
        (1, 0),
        (0, 1),
        (10,),
        (1, 2, 3),
        (2, 3, 4),
    ]
    transfs = [
        lambda s: s,
        lambda s: (1,) + s,
        lambda s: (1, 1) + s,
        lambda s: (8,) + s,
        lambda s: tuple(7 if i == 1 else i for i in s),
        lambda s: (4,) + tuple(3 if i == 1 else i for i in s),
    ]
    for dtype, shape, transf in itertools.product(dtypes, shapes, transfs):
        array = random_array(shape, dtype)
        crosscheck_operation(op, array, shape=transf(shape))

def test_broadcast_arrays():
    op = np.broadcast_arrays # uses __array_function__
    dtypes = [
        [('f0', float)],
        'f,f',
        'f,d,f',
        'f,O',
        'S25,U10,?',
        [('a', 'f,f', (3, 1)), ('b', [('c', 'd,d'), ('e', '?', 15)])],
    ]
    shapes = [
        (),
        (0,),
        (1,),
        (1, 0),
        (0, 1),
        (10,),
        (1, 2, 3),
        (2, 3, 4),
    ]
    transfs = [
        lambda s: s,
        lambda s: (1,) + s,
        lambda s: (1, 1) + s,
        lambda s: (8,) + s,
        lambda s: tuple(7 if i == 1 else i for i in s),
        lambda s: (4,) + tuple(3 if i == 1 else i for i in s),
    ]
    for dtype, shape, transf in itertools.product(dtypes, shapes, transfs):
        array1 = random_array(shape, dtype)
        array2 = random_array(transf(shape), dtype)
        crosscheck_operation(op, array1, array2)

def test_broadcast():
    def op(array1, array2):
        return lgp.broadcast(array1, array2).shape
    dtypes = [
        [('f0', float)],
        'f,f',
        'f,d,f',
        'f,O',
        'S25,U10,?',
        [('a', 'f,f', (3, 1)), ('b', [('c', 'd,d'), ('e', '?', 15)])],
    ]
    shapes = [
        (),
        (0,),
        (1,),
        (1, 0),
        (0, 1),
        (10,),
        (1, 2, 3),
        (2, 3, 4),
    ]
    transfs = [
        lambda s: s,
        lambda s: (1,) + s,
        lambda s: (1, 1) + s,
        lambda s: (8,) + s,
        lambda s: tuple(7 if i == 1 else i for i in s),
        lambda s: (4,) + tuple(3 if i == 1 else i for i in s),
    ]
    for dtype, shape, transf in itertools.product(dtypes, shapes, transfs):
        array1 = random_array(shape, dtype)
        array2 = random_array(transf(shape), dtype)
        crosscheck_operation(op, array1, array2)

def test_asarray():
    def op(array):
        return lgp.asarray(array)
    dtypes = [
        [('f0', float)],
        'f,f',
        'f,d,f',
        'f,O',
        'S25,U10,?',
        [('a', 'f,f', (3, 1)), ('b', [('c', 'd,d'), ('e', '?', 15)])],
    ]
    shapes = [
        (),
        (0,),
        (1,),
        (1, 0),
        (0, 1),
        (10,),
        (1, 2, 3),
        (2, 3, 4),
    ]
    for dtype, shape in itertools.product(dtypes, shapes):
        array = random_array(shape, dtype)
        crosscheck_operation(op, array)

def test_multiindex():
    def op(a):
        names = list(a.dtype.names)
        return a[names], a[names[:2]], a[names[-2:]], a[names[:1]]
    dtypes = [
        [('f0', float)],
        'f,f',
        'f,d,f',
        'f,O',
        'S25,U10,?',
        [('a', 'f,f', (3, 1)), ('b', [('c', 'd,d'), ('e', '?', 15)])],
    ]
    shapes = [
        (),
        (0,),
        (1,),
        (1, 0),
        (0, 1),
        (10,),
        (1, 2, 3),
        (2, 3, 4),
    ]
    for dtype, shape in itertools.product(dtypes, shapes):
        array = random_array(shape, dtype)
        crosscheck_operation(op, array)

def array_meta_hash(a):
    d = a.dtype
    shape_hash = sum(a.shape) + a.size + a.ndim
    type_hash = len(d) + d.itemsize + sum(d[i].num for i in range(len(d))) + d.num
    return shape_hash + type_hash

def test_setfield():
    def op(a):
        name = a.dtype.names[0]
        a0 = a[name]
        rng = np.random.default_rng(array_meta_hash(a))
        val = random_array(a0.shape, a0.dtype, rng)
        if hasattr(a, 'at'):
            a = a.at[name].set(val)
        else:
            a[name] = val
        return a
    dtypes = [
        [('f0', float)],
        'f,f',
        'f,d,f',
        'f,O',
        'S25,U10,?',
        [('a', 'f,f', (3, 1)), ('b', [('c', 'd,d'), ('e', '?', 15)])],
    ]
    shapes = [
        (),
        (0,),
        (1,),
        (1, 0),
        (0, 1),
        (10,),
        (1, 2, 3),
        (2, 3, 4),
    ]
    for dtype, shape in itertools.product(dtypes, shapes):
        array = random_array(shape, dtype)
        crosscheck_operation(op, array)

def test_tree():
    dtypes = [
        [('f0', float)],
        'f,f',
        'f,d,f',
        'f,O',
        'S25,U10,?',
        [('a', 'f,f', (3, 1)), ('b', [('c', 'd,d'), ('e', '?', 15)])],
    ]
    shapes = [
        (),
        (0,),
        (1,),
        (1, 0),
        (0, 1),
        (10,),
        (1, 2, 3),
        (2, 3, 4),
    ]
    for dtype, shape in itertools.product(dtypes, shapes):
        array1 = random_array(shape, dtype)
        array = lgp.StructuredArray(array1)
        children, aux_data = tree_util.tree_flatten(array)
        array = tree_util.tree_unflatten(aux_data, children)
        array2 = np.asarray(array)
        util.assert_equal(array1, array2)

def test_tree_reshaped():
    dtypes = [
        [('f0', float)],
        'f,5f',
        'f,d,f',
        'f,O',
        'S25,U10,?',
        [('a', 'f,f', (3, 1)), ('b', [('c', 'd,d'), ('e', '?', 15)])],
    ]
    shapes = [
        (),
        (0,),
        (1,),
        (1, 0),
        (0, 1),
        (10,),
        (1, 2, 3),
        (2, 3, 4),
    ]
    for dtype, shape in itertools.product(dtypes, shapes):
        
        basearray = random_array(shape, dtype)
        
        slices = tuple(slice(rng.integers(size + 1)) for size in shape)
        array1 = basearray[slices]
        
        array = lgp.StructuredArray(basearray)
        children, aux_data = tree_util.tree_flatten(array)
        children = tuple(x[slices] for x in children)
        array = tree_util.tree_unflatten(aux_data, children)
        array2 = np.asarray(array)
        
        util.assert_equal(array1, array2)

        lead = (3,)
        array1 = np.broadcast_to(basearray, lead + shape)
        
        array = lgp.StructuredArray(basearray)
        children, aux_data = tree_util.tree_flatten(array)
        children = tuple(np.broadcast_to(x, lead + x.shape) for x in children)
        array = tree_util.tree_unflatten(aux_data, children)
        array2 = np.asarray(array)
        
        util.assert_equal(array1, array2)

        if shape and shape[0]:

            array1 = basearray[0]
            
            array = lgp.StructuredArray(basearray)
            children, aux_data = tree_util.tree_flatten(array)
            children = tuple(x[0] for x in children)
            array = tree_util.tree_unflatten(aux_data, children)
            array2 = np.asarray(array)
            
            util.assert_equal(array1, array2)

def test_reshape():
    def op1(array, shape):
        return array.reshape(shape)
    def op2(array, shape):
        return array.reshape(*shape) if shape else array.reshape(shape)
    dtypes = [
        [('f0', float)],
        'f,f',
        'f,d,f',
        'f,O',
        'S25,U10,?',
        [('a', 'f,f', (3, 1)), ('b', [('c', 'd,d'), ('e', '?', 15)])],
    ]
    shapes = [
        (),
        (0,),
        (1,),
        (1, 0),
        (0, 1),
        (10,),
        (1, 2, 3),
        (2, 3, 4),
    ]
    transfs = [
        lambda s: s,
        lambda s: (1,) + s,
        lambda s: s + (1,),
        lambda s: (1, 1) + s,
        lambda s: (-1,) + s,
        lambda s: s + (-1,),
        lambda s: s[:-1] + (-1,),
        lambda s: s[:-2] + (-1,),
        lambda s: (-1,) + s[1:],
        lambda s: (-1,) + s[2:],
    ]
    for dtype, shape, transf in itertools.product(dtypes, shapes, transfs):
        array = random_array(shape, dtype)
        newshape = transf(shape)
        if -1 in newshape and (not shape or 0 in shape):
            continue
        crosscheck_operation(op1, array, shape=newshape)
        crosscheck_operation(op2, array, shape=newshape)

def test_repr():
    x = np.array(
      [[(  79.4607   ,   34.16494104,  True),
        (-122.71211  ,   70.48242559, False)],
       [( -95.21081  ,   80.3448    , False),
        (  52.144142 ,   57.35083659, False)],
       [(  97.3018   ,  -65.78833393,  True),
        (  88.46741  ,  -85.54110175, False)],
       [(  -2.7272243,  -16.86393124, False),
        (   0.946183 ,  -50.12316733, False)],
       [( -19.002777 , -114.6684194 ,  True),
        ( -59.26707  ,   87.90692279, False)]],
      dtype=[('f0', '<f4'), ('f1', '<f8'), ('f2', '?')]
    )
    y = lgp.StructuredArray(x)
    s = """\
StructuredArray({
    'f0':
    array([[  79.4607   , -122.71211  ],
           [ -95.21081  ,   52.144142 ],
           [  97.3018   ,   88.46741  ],
           [  -2.7272243,    0.946183 ],
           [ -19.002777 ,  -59.26707  ]], dtype=float32),
    'f1':
    array([[  34.16494104,   70.48242559],
           [  80.3448    ,   57.35083659],
           [ -65.78833393,  -85.54110175],
           [ -16.86393124,  -50.12316733],
           [-114.6684194 ,   87.90692279]]),
    'f2':
    array([[ True, False],
           [False, False],
           [ True, False],
           [False, False],
           [ True, False]]),
})"""
    assert repr(y) == s
    
    x = np.array(
      [(43.52967   , -258.21209855, False),
       ( 0.80091816,   80.99479392, False)],
      dtype=[('f0', '<f4'), ('f1', '<f8'), ('f2', '?')]
    )
    y = lgp.StructuredArray(x)
    s = "StructuredArray({'f0': array([43.52967   ,  0.80091816], dtype=float32), 'f1': array([-258.21209855,   80.99479392]), 'f2': array([False, False])})"
    assert repr(y) == s

    x = np.array(
      [([ 102.43267  ,   25.42825  ,  -76.084114 ,   39.536003 ],   85.10061608),
       ([  -6.0310173, -108.25371  ,  -20.205214 , -233.0124   ], -152.64632798)],
      dtype=[('f0', '<f4', (4,)), ('f1', '<f8')]
    )
    y = lgp.StructuredArray(x)
    s = """\
StructuredArray({
    'f0':
    array([[ 102.43267  ,   25.42825  ,  -76.084114 ,   39.536003 ],
           [  -6.0310173, -108.25371  ,  -20.205214 , -233.0124   ]],
          dtype=float32),
    'f1': array([  85.10061608, -152.64632798]),
})"""
    assert repr(y) == s
    
    x = np.zeros(5, [])
    y = lgp.StructuredArray(x)
    s = "StructuredArray({})"
    assert repr(y) == s

def test_set_subfield():
    y = random_array(5, [('a', [('b', float)])])
    x = lgp.StructuredArray(y)
    a = x['a']
    x = x.at['a'].set(random_array(a.shape, a.dtype))
    b = x['a']['b']
    newb = random_array(b.shape, b.dtype)
    x1 = x.at['a'].set(x['a'].at['b'].set(newb))
    x2 = x.at['a']['b'].set(newb)
    util.assert_equal(np.array(x1), np.array(x2))

def test_s2u():
    dtypes = [
        [('f0', float)],
        'f,f',
        'f,d,f',
        # 'f,O',
        # structured_to_unstructured does not work on object arrays, see
        # numpy issue #21990
        'S25,U10,?',
        [('a', 'f,f', (3, 1)), ('b', [('c', 'd,d'), ('e', '?', 15)])],
        [('a', [('aa', 'f,f', 2), ('ab', float)], 3)],
    ]
    shapes = [
        (),
        (0,),
        (1,),
        (1, 0),
        (0, 1),
        (10,),
        (1, 2, 3),
        (2, 3, 4),
    ]
    for dtype, shape in itertools.product(dtypes, shapes):
        array = random_array(shape, dtype)
        crosscheck_operation(recfunctions.structured_to_unstructured, array)

def test_squeeze():
    dtypes = [
        [('f0', float)],
        'f,f',
        'f,d,f',
        # 'f,O',
        # structured_to_unstructured does not work on object arrays, see
        # numpy issue #21990
        'S25,U10,?',
        [('a', 'f,f', (3, 1)), ('b', [('c', 'd,d'), ('e', '?', 15)])],
        [('a', [('aa', 'f,f', 2), ('ab', float)], 3)],
    ]
    shapes = [
        (),
        (0,),
        (1,),
        (1, 0),
        (0, 1),
        (10, 1),
        (1, 2, 3),
        (2, 3, 4, 1),
    ]
    for dtype, shape in itertools.product(dtypes, shapes):
        array = random_array(shape, dtype)
        crosscheck_operation(np.squeeze, array)

    shapes = [
        (1,),
        (1, 0),
        (1, 2),
        (1, 3, 7),
    ]
    for dtype, shape in itertools.product(dtypes, shapes):
        array = random_array(shape, dtype)
        crosscheck_operation(lambda a: a.squeeze(axis=0), array)
        crosscheck_operation(lambda a: a.squeeze(axis=(0,)), array)
    
    shapes = [
        (1, 0, 1),
        (1, 2, 1),
        (1, 3, 1, 7),
    ]
    for dtype, shape in itertools.product(dtypes, shapes):
        array = random_array(shape, dtype)
        crosscheck_operation(lambda a: a.squeeze(axis=(0, 2)), array)
        crosscheck_operation(lambda a: a.squeeze(axis=(2, 0)), array)

@pytest.mark.parametrize('cls', [pl.DataFrame, pd.DataFrame])
def test_dataframe(cls):
    """ test DataFrame -> StructuredArray, both with Pandas and Polars """
    a = random_array(10, 'f4,f8,i1,i2,i4,i8,u1,u2,u4,u8,?,U16,S16')
    df = cls(pd.DataFrame.from_records(a))
    s = lgp.StructuredArray.from_dataframe(df)
    a2 = np.array(s)
    util.assert_equal(a, a2)

def test_not_handled():
    with pytest.raises(TypeError):
        np.concatenate([lgp.StructuredArray(np.zeros(1, 'd,d'))])

def test_asjax():
    
    x = random_array(4, float)
    assert isinstance(x, np.ndarray)
    x = _array._asarray_jaxifpossible(x)
    assert isinstance(x, jnp.ndarray)

    x = random_array(5, 'S5')
    x = _array._asarray_jaxifpossible(x)
    assert isinstance(x, np.ndarray)

    x = random_array(2, 'd,U10')
    x = _array._asarray_jaxifpossible(x)
    assert isinstance(x['f0'], jnp.ndarray)
    assert isinstance(x['f1'], np.ndarray)

def test_shortkey():
    x = random_array((2, 3), 'd,d')
    x = lgp.StructuredArray(x)
    elem = x[0]
    assert elem.shape == (3,)
    assert elem.dtype == 'd,d'

def test_longkey():
    x = random_array((2, 3), '5d,5d')
    x = lgp.StructuredArray(x)
    with pytest.raises(IndexError):
        elem = x[0, 1, 2]

def test_unflatten_dummy():
    x = random_array((2, 3), 'f,d,?')
    x = lgp.StructuredArray(x)
    _, aux = tree_util.tree_flatten(x)
    children = (8., False, None)
    y = tree_util.tree_unflatten(aux, children)
    assert y.shape == ()
    assert y.dtype == [('f0', float), ('f1', bool), ('f2', object)]

def test_incompatible_shapes():
    x = random_array((2, 3), 'f,2d,?')
    x = lgp.StructuredArray(x)
    _, aux = tree_util.tree_flatten(x)
    children = (8., False, None)
    y = tree_util.tree_unflatten(aux, children)
    assert y.shape == ()
    assert y.dtype == [('f0', float), ('f1', bool), ('f2', object)]
