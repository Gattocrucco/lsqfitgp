# lsqfitgp/copula/_base.py
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

""" define DistrBase """

import abc
import functools
import collections

import gvar
import numpy

class DistrBase(metaclass=abc.ABCMeta):
    r"""

    Abstract base class to represent (trees of) probability distributions.

    Attributes
    ----------
    in_shape : tuple of int
        The core shape of the input array to `partial_invfcn`.
    shape : (tree of) tuple of int
        The core shape of the output array of `partial_invfcn`.
    dtype : (tree of) dtype
        The dtype of the output array of `partial_invfcn`.
    distrshape : (tree of) tuple of int
        The sub-core shape of the output array of `partial_invfcn` that
        represents the atomic shape of the distribution.

    Methods
    -------
    partial_invfcn :
        Transformation function from a (multivariate) Normal variable to the
        target random variable.
    add_distribution :
        Register the distribution for usage with `gvar.BufferDict`.
    gvars :
        Return an array of gvars with the appropriate shape for usage with
        `gvar.BufferDict`.

    See also
    --------
    Distr, Copula

    """

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        cls._named = {}

    in_shape = NotImplemented
    shape = NotImplemented
    dtype = NotImplemented
    distrshape = NotImplemented

    def partial_invfcn(self, x):
        """
            
        Map independent Normal variables to the desired distribution.

        This function is a generalized ufunc. It is jax traceable and
        differentiable one time. It supports arrays of gvars as input. The
        attributes `in_shape` and `shape` give the core shapes.

        Parameters
        ----------
        x : ``(..., *in_shape)`` array
            An array of values representing draws of i.i.d. Normal variates.

        Returns
        -------
        y : (tree of) ``(..., *shape)`` array
            An array of values representing draws of the desired distribution.

        """
        return self._partial_invfcn(x)

    @abc.abstractmethod
    def _partial_invfcn(self, x):
        pass
    
    def _is_same_family(self, invfcn):
        return getattr(invfcn, '__self__', None).__class__ is self.__class__

    def add_distribution(self, name):
        """

        Register the distribution for usage with `gvar.BufferDict`.

        Parameters
        ----------
        name : str
            The name to use for the distribution. It must be globally unique,
            and it should not contain parentheses. To redefine a distribution
            with the same name, use `gvar.BufferDict.del_distribution` first.
            However, it is allowed to reuse the name if the distribution family,
            shape and parameters are identical to those used for the existing
            definition.

        See also
        --------
        gvar.BufferDict.add_distribution, gvar.BufferDict.del_distribution

        """

        if gvar.BufferDict.has_distribution(name):
            invfcn = gvar.BufferDict.invfcn[name]
            if not self._is_same_family(invfcn):
                raise ValueError(f'distribution {name} already defined')
            existing = self._named[name]
            if existing != self._staticdescr:
                raise ValueError('Attempt to overwrite existing'
                    f' {self.__class__.__name__} distribution with name {name}')
                # cls._named is not updated by
                # gvar.BufferDict.del_distribution, but it is not a problem
        
        else:
            gvar.BufferDict.add_distribution(name, self.partial_invfcn)
            self._named[name] = self._staticdescr
    
    def gvars(self):
        """

        Return an array of gvars intended as value in a `gvar.BufferDict`.

        Returns
        -------
        gvars : array of gvars
            An array of i.i.d. standard Normal primary gvars with shape
            `in_shape`.

        """

        return gvar.gvar(numpy.zeros(self.in_shape), numpy.ones(self.in_shape))

    @abc.abstractmethod
    def __repr__(self, path='', cache=None):
        """ produce a representation where no object appears more than once,
        later appearances are replaced by a user-friendly identifier """
        if cache is None:
            cache = {}
        if self in cache:
            return cache[self]
        cache[self] = f'<{path}>'
        return cache

    class _Path(collections.namedtuple('Path', ['path'])): pass

    @abc.abstractmethod
    def _compute_staticdescr(self, path, cache):
        """ compute static description of self, can be compared """
        if self in cache:
            return cache[self]
        cache[self] = self._Path(path)

    @functools.cached_property
    def _staticdescr(self):
        return self._compute_staticdescr([], {})

    @abc.abstractmethod
    def _compute_in_size(self, cache):
        """ compute input size to partial_invfcn, without double counting """
        if self in cache:
            return 0
        cache.add(self)

    @abc.abstractmethod
    def _partial_invfcn_internal(self, x, i, cache):
        assert x.ndim == 1
        if self in cache:
            return cache[self], i
