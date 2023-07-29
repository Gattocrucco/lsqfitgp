# lsqfitgp/tests/util.py
#
# Copyright (c) 2022, 2023, Giacomo Petrillo
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

import functools
import warnings

from jax import tree_util
import numpy as np
from jax import numpy as jnp
import gvar
import pytest
from scipy import linalg

import lsqfitgp as lgp

def jaxtonumpy(x):
    """
    Recursively convert jax arrays in x to numpy arrays.
    """
    children, meta = tree_util.tree_flatten(x)
    children = (np.array(x) if isinstance(x, jnp.ndarray) else x for x in children)
    return tree_util.tree_unflatten(meta, children)

def assert_equal(*args):
    """
    Version of assert_equal that works with jax arrays and StructuredArray, and
    which fixes numpy issue #21739
    """
    assert_array_equal(*jaxtonumpy(args))

def assert_array_equal(*args):
    args = [np.array(a) if isinstance(a, lgp.StructuredArray) else a for a in args]
    a = args[0]
    if isinstance(a, np.ndarray) and a.size == 0 and a.dtype.names:
        assert all(b.dtype == a.dtype for b in args)
        assert all(b.shape == a.shape for b in args)
    elif isinstance(a, np.ndarray) and a.dtype.names:
        # old versions of numpy force structured dtypes to be equal when
        # comparing, instead of casting
        assert all(b.dtype.names == a.dtype.names for b in args)
        for n in a.dtype.names:
            assert_array_equal(*(x[n] for x in args))
    elif isinstance(a, np.ndarray) and all(np.issubdtype(b.dtype, np.character) for b in args):
        # old versions of numpy do not compare as equal strings with different
        # allocated space
        basetype = a.dtype.type
        assert all(b.dtype.type == basetype for b in args) # check it's the same kind of string type
        basesize = np.dtype((basetype, 1)).itemsize
        maxsize = max(b.dtype.itemsize // basesize for b in args)
        newtype = np.dtype((basetype, maxsize))
        args = (b.astype(newtype) for b in args)
        np.testing.assert_equal(*args)
    else:
        np.testing.assert_equal(*args)

def mark(cls, meth, mark):
    """
    Function to mark a test method.
    """
    impl = getattr(cls, meth)
    if not meth.startswith('test_'):
        warnings.warn(f'method {cls.__name__}.{meth} not prefixed with test_')
    marker = getattr(pytest.mark, mark)
    @marker
    @functools.wraps(impl) # `wraps` needed because pytest uses the method name
    def newimpl(self):
        # wrap because otherwise the superclass method would be marked too,
        # but set the mark temporarily to allow introspection
        marker(impl)
        try:
            impl(self)
        finally:
            impl.pytestmark.pop()
    setattr(cls, meth, newimpl)

def xfail(cls, meth):
    mark(cls, meth, 'xfail')

def skip(cls, meth):
    mark(cls, meth, 'skip')

def tryagain(fun, rep=2, method=False):
    """
    Decorates test `fun` to make it try again in case of failure. Inhibited
    by xfails.
    """
    meta = {}
    @functools.wraps(fun)
    def newfun(*args, _meta=meta, **kw):

        job = lambda: fun(*args, **kw)

        marks = getattr(_meta['newfun'], 'pytestmark', [])
        if any(m.name == 'xfail' for m in marks):
            return job()

        if method:
            self = args[0]
            name = f'{self.__class__.__name__}.{fun.__name__}'
        else:
            name = fun.__name__

        for i in range(rep):
            try:
                x = job()
                if i > 0:
                    warnings.warn(f'Test {name} failed {i} times with exception {exc.__class__.__name__}: ' + ", ".join(exc.args))
                return x
            except Exception as e:
                exc = e

        # if rep > 1:
        #     warnings.warn(f'Test {name} failed {rep} times')
        raise exc
    
    if hasattr(fun, 'pytestmark'):
        newfun.pytestmark = fun.pytestmark
    meta['newfun'] = newfun
    return newfun

def assert_close_matrices(actual, desired, *, rtol=0, atol=0, tozero=False):
    """
    Check if two matrices are similar.

    Scalars and vectors are intepreted as 1x1 and Nx1 matrices, but the two
    arrays must have the same shape beforehand.

    The closeness condition is:

        ||actual - desired|| <= atol + rtol * ||desired||,

    where the norm is the matrix 2-norm, i.e., the maximum (in absolute value)
    singular value. The tolerances are 0 by default.

    Parameters
    ----------
    actual, desired : array_like
        The two matrices to be compared. Must be scalars, vectors, or 2d arrays.
    rtol, atol : scalar
        Relative and absolute tolerances for the comparison.
    tozero : bool
        Default False. If True, use the following codition instead:

            ||actual|| <= atol + rtol * ||desired||

    Raises
    ------
    AssertionError :
        If the condition is not satisfied.
    """

    actual = np.asarray(actual)
    desired = np.asarray(desired)
    assert actual.shape == desired.shape
    if actual.size == 0:
        return
    actual = np.atleast_1d(actual)
    desired = np.atleast_1d(desired)
    
    if tozero:
        expr = 'actual'
        ref = 'zero'
    else:
        expr = 'actual - desired'
        ref = 'desired'

    dnorm = linalg.norm(desired, 2)
    adnorm = linalg.norm(eval(expr), 2)
    ratio = adnorm / dnorm if dnorm else np.nan

    msg = f"""\
matrices actual and {ref} are not close in 2-norm
norm(desired) = {dnorm:.2g}
norm({expr}) = {adnorm:.2g}  (atol = {atol:.2g})
ratio = {ratio:.2g}  (rtol = {rtol:.2g})"""

    assert adnorm <= atol + rtol * dnorm, msg

def _assert_similar_gvars(g, h, rtol, atol):
    assert_allclose(gvar.mean(g), gvar.mean(h), rtol=rtol, atol=atol)
    g = np.reshape(g, -1)
    h = np.reshape(h, -1)
    assert_close_matrices(gvar.evalcov(g), gvar.evalcov(h), rtol=rtol, atol=atol)

def assert_similar_gvars(*gs, rtol=0, atol=0):
    if gs:
        for g in gs[1:]:
            _assert_similar_gvars(g, gs[0], rtol, atol)

def assert_same_gvars(actual, desired, *, rtol=0, atol=0):
    z = np.reshape(actual - desired, -1)
    desired = np.reshape(desired, -1)
    kw = dict(tozero=True, rtol=rtol, atol=atol)
    assert_close_matrices(gvar.mean(z), gvar.mean(desired), **kw)
    assert_close_matrices(gvar.evalcov(z), gvar.evalcov(desired), **kw)

def assert_close_decomps(actual, desired, *, rtol=0, atol=0):
    assert actual.n == desired.n
    def compare(a, b):
        I = np.eye(a.n)
        Z = a.correlate(I)
        zzbz = a.correlate(b.ginv_quad(Z)) # ZZ'K⁻Z
        assert_close_matrices(zzbz, Z, rtol=rtol, atol=atol)
    compare(actual, desired)
    compare(desired, actual)

def assert_allclose(actual, desired, *, rtol=0, atol=0, equal_nan=False, **kw):
    """ change the default arguments of np.testing.assert_allclose """
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, equal_nan=equal_nan, **kw)
