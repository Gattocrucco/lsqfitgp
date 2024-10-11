# lsqfitgp/_signature.py
#
# Copyright (c) 2023, 2024 Giacomo Petrillo
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

""" define Signature """

import inspect

try:
    from numpy.lib import function_base # numpy 1
except ImportError:
    from numpy.lib import _function_base_impl as function_base # numpy 2
import jax
from jax import numpy as jnp

class Signature:
    """ Class to parse a numpy gufunc signature. """

    def __init__(self, signature):
        self.signature = signature
        self.incores, self.outcores = function_base._parse_gufunc_signature(signature)

    @classmethod
    def from_tuples(cls, incores, outcores):
        self = cls.__new__(cls)
        tuplestr = lambda t: '(' + ','.join(map(str, t)) + ')'
        self.signature = ','.join(map(tuplestr, incores)) + '->' + ','.join(map(tuplestr, outcores))
        self.incores = incores
        self.outcores = outcores
        return self

    def __repr__(self):
        return self.signature

    def check_nargs(self, func):
        """ Check that the function has the correct number of arguments. """
        sig = inspect.signature(func)
        if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()):
            return
        if len(sig.parameters) != len(self.incores):
            raise ValueError(f'function {func} has {len(sig.parameters)} '
                f'arguments, but signature {self.signature} '
                f'requires {len(self.incores)}')

    @property
    def nin(self):
        return len(self.incores)

    @property
    def nout(self):
        return len(self.outcores)

    def eval(self, *args):
        """

        Evaluate the signature with the given arguments.

        Parameters
        ----------
        args : sequence of numpy.ndarray or None
            A missing argument can be replaced with None, provided the other
            arguments are sufficient to infer all dimension sizes.

        Returns
        -------
        sig : EvaluatedSignature
            An object with attributes `broadcast_shape`, `sizes`,
            `core_out_shapes`, `out_shapes`, `core_in_shapes`, `in_shapes`.

        """
        return self.EvaluatedSignature(self, *args)
    
    class EvaluatedSignature:

        def __init__(self, sig, *args):

            assert len(args) == len(sig.incores)
            
            known_args = []
            known_cores = []
            missing_cores = []
            for arg, core in zip(args, sig.incores):
                if arg is None:
                    missing_cores.append(core)
                else:
                    known_args.append(jax.ShapeDtypeStruct(arg.shape, 'd'))
                    known_cores.append(core)
            
            self.broadcast_shape, self.sizes = function_base._parse_input_dimensions(known_args, known_cores)

            missing_indices = set(sum(missing_cores, ()))
            missing_indices.difference_update(self.sizes)
            if missing_indices:
                raise ValueError(f'cannot infer sizes of dimesions {missing_indices} from signature {sig.signature}')

            self.core_out_shapes, self.out_shapes = self._compute_shapes(sig.outcores)
            self.core_in_shapes, self.in_shapes = self._compute_shapes(sig.incores)

        def _compute_shapes(self, cores):
            coreshapes = []
            shapes = []
            for core in cores:
                core = tuple(self.sizes[i] for i in core)
                coreshapes.append(core)
                shapes.append(self.broadcast_shape + core)
            return tuple(coreshapes), tuple(shapes)

        def _repr(self, shapes):
            return ','.join(map(str, shapes)).replace(' ', '')

        def __repr__(self):
            return self._repr(self.in_shapes) + '->' + self._repr(self.out_shapes)

# I use numpy's internals to parse the signature, but these do not correspond to
# the description in
# https://numpy.org/doc/stable/reference/c-api/generalized-ufuncs.html. In
# particular:
# - integers are parsed like identifiers
# - ? is not accepted
# Looking at the issues, it seems a long standing issue amongst many with
# vectorize and is not going to be solved.
# See https://github.com/HypothesisWorks/hypothesis/blob/4e675dee1a4cba9d6902290bbc5719fd72072ec7/hypothesis-python/src/hypothesis/extra/_array_helpers.py#L289
# for a more complete implementation
