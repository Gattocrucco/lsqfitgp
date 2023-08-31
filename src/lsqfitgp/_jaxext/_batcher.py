# lsqfitgp/_jaxext/_batcher.py
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

import functools
import math

from jax import lax
from jax import numpy as jnp
import numpy

def batchufunc(func, *, maxnbytes):
    """

    Make a batched version of an universal function.

    The function is modified to process its inputs in chunks.

    Parameters
    ----------
    func : callable
        A jax-traceable universal function. All positional arguments are assumed
        to be arrays which are broadcasted to determine the shape.
    maxnbytes : number
        The maximum number of bytes in each input chunck over all input arrays
        after broadcasting.

    Return
    ------
    batched_func : callable
        The batched version of `func`. Keywords arguments are passed as-is to
        the function.

    """

    maxnbytes = int(maxnbytes)
    assert maxnbytes > 0

    @functools.wraps(func)
    def batched_func(*args, **kw):

        shape = jnp.broadcast_shapes(*(arg.shape for arg in args))
        if not shape or any(size == 0 for size in shape):
            return func(*args)

        rowsize = math.prod(shape[1:])
        rownbytes = rowsize * sum(arg.dtype.itemsize for arg in args)
        totalnbytes = shape[0] * rownbytes
        if totalnbytes <= maxnbytes:
            return func(*args)

        args = [arg.reshape((1,) * (len(shape) - arg.ndim) + arg.shape) for arg in args]
        short_args_idx = [i for i, arg in enumerate(args) if arg.shape[0] == 1]
        long_args_idx = [i for i, arg in enumerate(args) if arg.shape[0] > 1]
        short_args = [args[i].squeeze(0) for i in short_args_idx]
        long_args = [args[i] for i in long_args_idx]

        def combine_args(short_args, long_args):
            args = [None] * (len(short_args) + len(long_args))
            for i, arg in zip(short_args_idx, short_args):
                args[i] = arg
            for i, arg in zip(long_args_idx, long_args):
                args[i] = arg
            return args

        if rownbytes <= maxnbytes:
            # batch over leading axis

            batchsize = maxnbytes // rownbytes
            nbatches = shape[0] // batchsize
            batchedsize = nbatches * batchsize
            
            sliced_args = [arg[:batchedsize] for arg in long_args]
            batched_args = [
                arg.reshape((nbatches, batchsize) + arg.shape[1:])
                for arg in sliced_args
            ]
            def scan_loop_body(short_args, batched_args):
                assert all(arg.ndim == len(shape) - 1 for arg in short_args)
                assert all(arg.ndim == len(shape) for arg in batched_args)
                args = combine_args(short_args, batched_args)
                out = func(*args, **kw)
                assert out.shape == (batchsize,) + shape[1:]
                return short_args, out
            _, out = lax.scan(scan_loop_body, short_args, batched_args)
            assert out.shape == (nbatches, batchsize) + shape[1:]
            out = out.reshape((batchedsize,) + shape[1:])
            
            remainder_args = [arg[batchedsize:] for arg in long_args]
            args = combine_args(short_args, remainder_args)
            remainder = func(*args)
            assert remainder.shape == (shape[0] - batchedsize,) + shape[1:]
            
            out = jnp.concatenate([out, remainder])

        else:
            # cycle over leading axis, recurse

            def scan_loop_body(short_args, long_args):
                args = combine_args(short_args, long_args)
                assert all(arg.ndim == len(shape) - 1 for arg in args)
                out = batched_func(*args, **kw)
                assert out.shape == shape[1:]
                return short_args, out
            _, out = lax.scan(scan_loop_body, short_args, long_args)

        assert out.shape == shape
        return out

    return batched_func
