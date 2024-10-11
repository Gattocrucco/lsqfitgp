# lsqfitgp/_Deriv.py
#
# Copyright (c) 2020, 2022, 2024, Giacomo Petrillo
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

import collections

import numpy as np

class Deriv:
    """
    Class for specifying derivatives. Behaves like a dictionary str -> int,
    where the keys represent variables and values the derivation order. An
    empty Deriv means no derivatives. A Deriv with one single key None means
    that the variable is implicit.

    Deriv(int) -> specified order derivative
    
    Deriv(str) -> first derivative w.r.t. specified variable
    
    Deriv(iter of str) -> derivative w.r.t. specified variables
    
    Deriv(iter of int, str) -> an int before a str acts as a multiplier
    
    Deriv(Deriv) -> pass through
    
    Deriv(None) -> Deriv(0)
    
    Example: Deriv(['a', 'b', 'b', 'c']) is equivalent to
    Deriv(['a', 2, 'b', 'c']).
    
    Raises
    ------
    TypeError
        If ``*args`` is not of the specified form.
    ValueError
        If ``*args`` ends with an integer or if there are consecutive
        integers.
    
    Attributes
    ----------
    implicit
    order
    max

    """
   
    def __new__(cls, *args):
        c = collections.Counter()
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, cls):
                return arg
            elif isinstance(arg, (int, np.integer)):
                if arg < 0:
                    raise ValueError(f'degree {arg} is negative')
                if arg:
                    c.update({None: arg})
            elif isinstance(arg, str):
                c.update([arg])
            elif np.iterable(arg):
                integer = None
                for obj in arg:
                    if isinstance(obj, str):
                        if integer is not None:
                            if integer:
                                c.update({obj: integer})
                            integer = None
                        else:
                            c.update([obj])
                    elif isinstance(obj, (int, np.integer)):
                        assert obj >= 0, obj
                        if integer is not None:
                            raise ValueError('consecutive integers in iterable')
                        integer = int(obj)
                    else:
                        raise TypeError('objects in iterable must be int or str')
                if integer is not None:
                    raise ValueError('dangling derivative order')
            elif arg is not None:
                raise TypeError('argument must be None, int, str, or iterable')
        elif len(args) != 0:
            raise ValueError(len(args))
        assert all(c.values())
        self = super().__new__(cls)
        self._counter = c
        return self
    
    def __getitem__(self, key):
        return self._counter[key]
    
    def __iter__(self):
        return iter(self._counter)
    
    def __len__(self):
        return len(self._counter)
    
    def __bool__(self):
        return bool(self._counter)

    def __eq__(self, val):
        if isinstance(val, Deriv):
            return self._counter == val._counter
        else:
            return NotImplemented
    
    def __repr__(self):
        return dict.__repr__(self._counter)
    
    @property
    def implicit(self):
        """
        True if the derivative is trivial or the variable is implicit.
        """
        return not self or next(iter(self._counter)) is None
    
    @property
    def order(self):
        """
        The total derivation order, i.e., the sum of the values.
        """
        # return self._counter.total() # works only in Python >=3.10
        return sum(self._counter.values())
    
    @property
    def max(self):
        """
        The maximum derivation order for any single variable.
        """
        return max(self._counter.values(), default=0)
