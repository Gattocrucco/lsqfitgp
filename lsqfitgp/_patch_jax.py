# lsqfitgp/_patch_jax.py
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

from jax.config import config
config.update("jax_enable_x64", True)

import functools

from jax import core
from jax.interpreters import ad
from jax.interpreters import batching
from scipy import special

def makejaxufunc(ufunc, *derivs):
    prim = core.Primitive(ufunc.__name__)
    
    @functools.wraps(ufunc)
    def func(*args):
        return prim.bind(*args)

    @prim.def_impl
    def impl(*args):
        return ufunc(*args)

    @prim.def_abstract_eval
    def abstract_eval(*args):
        return x

    jvps = (
        None if d is None
        else lambda g, *args: d(*args) * g
        for d in derivs
    )
    ad.defjvp(prim, *jvps)

    batching.defbroadcasting(prim)
    
    return func

j0 = makejaxufunc(special.j0, lambda x: -j1(x))
j1 = makejaxufunc(special.j1, lambda x: (j0(x) - jn(2, x)) / 2.0)
jn = makejaxufunc(special.jn, None, lambda n, x: (jn(n - 1, x) - jn(n + 1, x)) / 2.0)
kv = makejaxufunc(special.kv, None, lambda v, z: kvp(v, z, 1))
kvp = makejaxufunc(special.kvp, None, lambda v, z, n: kvp(v, z, n + 1), None)
