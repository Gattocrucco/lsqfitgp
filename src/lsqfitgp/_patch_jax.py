# lsqfitgp/_patch_jax.py
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

""" modifications to the global state of jax """

from jax import config
from jax import tree_util
import gvar
import numpy

config.update("jax_enable_x64", True)

class BufferDictPyTreeDef:

    @staticmethod
    def _skeleton(bd):
        """ Return a memoryless BufferDict with the same layout as `bd` """
        return gvar.BufferDict(bd, buf=numpy.empty(bd.buf.shape, []))
        # BufferDict mirrors the data type of the _buf attribute, so we do not
        # need to preserve it to maintain consistency. buf is not copied.

    def __init__(self, bd):
        self.skeleton = self._skeleton(bd)
        self.layout = {k: tuple(bd.slice_shape(k)) for k in bd.keys()}
        # it is not necessary to save the data type because that's in buf

    def __eq__(self, other):
        if not isinstance(other, __class__):
            return NotImplemented
        return self.layout == other.layout

    def __hash__(self):
        return hash(self.layout)

    def __repr__(self):
        return repr(self.layout)

    @classmethod
    def flatten(cls, bd):
        return (bd.buf,), cls(bd)

    @classmethod
    def unflatten(cls, self, children):
        buf, = children
        new = cls._skeleton(self.skeleton)
        # copy the skeleton to permit multiple unflattening
        new._extension = {}
        new._buf = buf
        return new

# register BufferDict as a pytree
tree_util.register_pytree_node(gvar.BufferDict, BufferDictPyTreeDef.flatten, BufferDictPyTreeDef.unflatten)

# TODO the current implementation of BufferDict as pytree is not really
# consistent with how JAX handles trees, because JAX expects to be allowed to
# put arbitrary objects in the leaves; in particular, internally it sometimes
# creates dummy trees filled with None. Maybe the current impl is fine with
# this; _buf gets set to None, and assuming the BufferDict is never really
# used in that crooked state, everything goes fine. The thing that this breaks
# is a subsequent flattening of the dummy, I think JAX never does this. (The
# reason for switching to buf-as-leaf in place of dict-values-as-leaves is that
# the latter breaks tracing.)
#
# Maybe since BufferDict is simple and stable, I could read its code, bypass its
# initialization altogether and set all the internal attributes to make it a
# proper pytree but also compatible with tracing.

# TODO try to drop BufferDict altogether. Currently I use it only in bcf and
# bart to pass stuff to a precompiled function. In empbayes_fit it is rebuilt
# by custom code.
