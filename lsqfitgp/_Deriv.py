import collections

from ._imports import numpy as np

__all__ = [
    'Deriv'
]

class Deriv:
    """
    Class for specifying derivatives. Behaves like a dictionary str -> int,
    where the keys represent variables and values the derivation order. An
    empty Deriv means no derivatives. A Deriv with one single key None means
    that the variable is implicit.
    """
    
    def __new__(cls, *args):
        """
        Deriv(int) -> specified order derivative
        
        Deriv(str) -> first derivative w.r.t specified variable
        
        Deriv(iter of str) -> derivative w.r.t specified variables
        
        Deriv(iter of int, str) -> an int before a str acts as a multiplier
        
        Deriv(Deriv) -> pass through
        
        Example: Deriv(['a', 'b', 'b', 'c']) is equivalent to
        Deriv(['a', 2, 'b', 'c']).
        
        Raises
        ------
        TypeError
            If `*args` is not of the specified form.
        ValueError
            If `*args` ends with an integer or if there are consecutive
            integers.
        
        Attributes
        ----------
        implicit
        order
        """
        c = collections.Counter()
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, Deriv):
                return arg
            elif isinstance(arg, (int, np.integer)):
                assert arg >= 0
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
                        assert obj >= 0
                        if integer is not None:
                            raise ValueError('consecutive integers in iterable')
                        integer = int(obj)
                    else:
                        raise TypeError('objects in iterable must be int or str')
                if integer is not None:
                    raise ValueError('dangling derivative order')
            else:
                raise TypeError('argument must be int, str, or iterable')
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
    
    def __bool__(self):
        return bool(self._counter)

    def __eq__(self, val):
        if isinstance(val, Deriv):
            return self._counter == val._counter
        else:
            return NotImplemented
    
    @property
    def implicit(self):
        """
        True if the derivative is trivial or the variable is implicit.
        """
        return not self or next(iter(self._counter)) is None
    
    @property
    def order(self):
        """
        The total derivation order, id est the sum of the values.
        """
        return sum(self._counter.values())
