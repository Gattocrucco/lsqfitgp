# lsqfitgp/_GP/_processes.py
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

import abc
import numbers

import numpy
from jax import numpy as jnp

from .. import _Kernel
from .. import _Deriv

from . import _base

class GPProcesses(_base.GPBase):

    def __init__(self, *, covfun):
        self._procs = {} # proc key -> _Proc
        self._kernels = {} # (proc key, proc key) -> CrossKernel
        if covfun is not None:
            if not isinstance(covfun, _Kernel.Kernel):
                raise TypeError('covariance function must be of class Kernel')
            self._procs[self.DefaultProcess] = self._ProcKernel(covfun)

    def _clone(self):
        newself = super()._clone()
        newself._procs = self._procs.copy()
        newself._kernels = self._kernels.copy()
        return newself

    class _Proc(abc.ABC):
        """
        Abstract base class for an object holding information about a process
        in a GP object.
        """
    
        @abc.abstractmethod
        def __init__(self): # pragma: no cover
            pass
    
    class _ProcKernel(_Proc):
        """An independent process defined with a kernel"""
    
        def __init__(self, kernel, deriv=0):
            assert isinstance(kernel, _Kernel.Kernel)
            self.kernel = kernel
            self.deriv = deriv
    
    class _ProcTransf(_Proc):
        """A process defined as a linear transformation of other processes"""
    
        def __init__(self, ops, deriv):
            """ops = dict proc key -> callable"""
            self.ops = ops
            self.deriv = deriv

    class _ProcLinTransf(_Proc):
    
        def __init__(self, transf, keys, deriv):
            self.transf = transf
            self.keys = keys
            self.deriv = deriv

    class _ProcKernelTransf(_Proc):
        """A process defined by an operation on the kernel of another process"""
    
        def __init__(self, proc, transfname, arg):
            """proc = proc key, transfname = Kernel transfname, arg = argument to transf """
            self.proc = proc
            self.transfname = transfname
            self.arg = arg
        
    _zerokernel = _Kernel.Zero()

    @_base.newself
    def defproc(self, key, kernel=None, *, deriv=0):
        """
        
        Add an independent process.
        
        Parameters
        ----------
        key : hashable
            The name that identifies the process in the GP object.
        kernel : Kernel
            A kernel for the process. If None, use the default kernel. The
            difference between the default process and a process defined with
            the default kernel is that, although they have the same kernel,
            they are independent.
        deriv : Deriv-like
            Derivatives to take on the process defined by the kernel.
                
        """
        
        if key in self._procs:
            raise KeyError(f'process key {key!r} already used in GP')
        
        if kernel is None:
            kernel = self._procs[self.DefaultProcess].kernel
        
        deriv = _Deriv.Deriv(deriv)
        
        self._procs[key] = self._ProcKernel(kernel, deriv)
    
    @_base.newself
    def deftransf(self, key, ops, *, deriv=0):
        """
        
        Define a new process as a linear combination of other processes.
        
        Let f_i(x), i = 1, 2, ... be already defined processes, and g_i(x) be
        deterministic functions. The new process is defined as
        
            h(x) = g_1(x) f_1(x) + g_2(x) f_2(x) + ...
        
        Parameters
        ----------
        key : hashable
            The name that identifies the new process in the GP object.
        ops : dict
            A dictionary mapping process keys to scalars or scalar
            functions. The functions must take an argument of the same kind
            of the domain of the process.
        deriv : Deriv-like
            The linear combination is derived as specified by this
            parameter.
        
        """
        
        for k, func in ops.items():
            if k not in self._procs:
                raise KeyError(f'process key {k!r} not in GP object')
            if not _Kernel.is_numerical_scalar(func) and not callable(func):
                raise TypeError(f'object of type {type(func)!r} for key {k!r} is neither scalar nor callable')

        if key in self._procs:
            raise KeyError(f'process key {key!r} already used in GP')
        
        deriv = _Deriv.Deriv(deriv)
        
        self._procs[key] = self._ProcTransf(ops, deriv)
        
        # we could implement deftransf in terms of deflintransf with
        # the following code, but deftransf has linear kernel building
        # cost so I'm leaving it around (probably not significant anyway)
        # functions = [
        #     op if callable(op)
        #     else (lambda x: lambda _: x)(op)
        #     for op in ops.values()
        # ]
        # def equivalent_lintransf(*procs):
        #     def fun(x):
        #         out = None
        #         for fun, proc in zip(functions, procs):
        #             this = fun(x) * proc(x)
        #             out = this if out is None else out + this
        #         return out
        #     return fun
        # self.deflintransf(key, equivalent_lintransf, list(ops.keys()), deriv=deriv, checklin=False)
    
    @_base.newself
    def deflintransf(self, key, transf, procs, *, deriv=0, checklin=False):
        """
        
        Define a new process as a linear combination of other processes.
        
        Let f_i(x), i = 1, 2, ... be already defined processes, and T
        a linear map from processes to a single process. The new process is
        
            h(x) = T(f_1, f_2, ...)(x).
        
        Parameters
        ----------
        key : hashable
            The name that identifies the new process in the GP object.
        transf : callable
            A function with signature ``transf(callable, callable, ...) -> callable``.
        procs : sequence
            The keys of the processes to be passed to the transformation.
        deriv : Deriv-like
            The linear combination is derived as specified by this
            parameter.
        checklin : bool
            If True, check if the transformation is linear. Default False.
        
        Notes
        -----
        The linearity check may fail if the transformation does nontrivial
        operations with the inner function input.
        
        """
        
        # TODO support procs being a single key
        
        if key in self._procs:
            raise KeyError(f'process key {key!r} already used in GP')
    
        for k in procs:
            if k not in self._procs:
                raise KeyError(k)
        
        deriv = _Deriv.Deriv(deriv)
        
        if len(procs) == 0:
            self._procs[key] = self._ProcKernel(self._zerokernel)
            return
        
        if checklin is None:
            checklin = self._checklin
        if checklin:
            mockup_function = lambda a: lambda _: a
            # TODO this array mockup fails with jax functions
            class Mockup(numpy.ndarray):
                __getitem__ = lambda *_: Mockup((0,))
                __getattr__ = __getitem__
            def checktransf(*arrays):
                functions = [mockup_function(a) for a in arrays]
                return transf(*functions)(Mockup((0,)))
            shapes = [(11,)] * len(procs)
            self._checklinear(checktransf, shapes, elementwise=True)
        
        self._procs[key] = self._ProcLinTransf(transf, procs, deriv)

    @_base.newself
    def deflinop(self, key, transfname, arg, proc):
        """
        
        Define a new process as the transformation of an existing one.
        
        Parameters
        ----------
        key : hashable
            Key for the new process.
        transfname : hashable
            A transformation recognized by the `~CrossKernel.transf` method
            of the kernel.
        arg :
            A valid argument to the transformation.
        proc : hashable
            Key of the process to be transformed.
        
        """
        
        if key in self._procs:
            raise KeyError(f'process key {key!r} already used in GP')
        if proc not in self._procs:
            raise KeyError(f'process {proc!r} not found')
        self._procs[key] = self._ProcKernelTransf(proc, transfname, arg)
    
    def defderiv(self, key, deriv, proc):
        """
        
        Define a new process as the derivative of an existing one.
        
        .. math::
            g(x) = \\frac{\\partial^n}{\\partial x^n} f(x)
        
        Parameters
        ----------
        key : hashable
            The key of the new process.
        deriv : Deriv-like
            Derivation order.
        proc : hashable
            The key of the process to be derived.
        
        Returns
        -------
        gp : GP
            A new GP object with the applied modifications.

        """
        deriv = _Deriv.Deriv(deriv)
        return self.deflinop(key, 'diff', deriv, proc)
    
    def defxtransf(self, key, transf, proc):
        """
        
        Define a new process by transforming the inputs of another one.
        
        .. math::
            g(x) = f(T(x))
        
        Parameters
        ----------
        key : hashable
            The key of the new process.
        transf : callable
            A function mapping the new kind input to the input expected by the
            transformed process.
        proc : hashable
            The key of the process to be transformed.
        
        Returns
        -------
        gp : GP
            A new GP object with the applied modifications.

        """
        assert callable(transf)
        return self.deflinop(key, 'xtransf', transf, proc)
    
    def defrescale(self, key, scalefun, proc):
        """
        
        Define a new process as a rescaling of an existing one.
        
        .. math::
            g(x) = s(x)f(x)
        
        Parameters
        ----------
        key : hashable
            The key of the new process.
        scalefun : callable
            A function from the domain of the process to a scalar.
        proc : hashable
            The key of the process to be transformed.
        
        Returns
        -------
        gp : GP
            A new GP object with the applied modifications.

        """
        assert callable(scalefun)
        return self.deflinop(key, 'rescale', scalefun, proc)
    
    def _crosskernel(self, xpkey, ypkey):
        
        # Check if the kernel is in cache.
        cache = self._kernels.get((xpkey, ypkey))
        if cache is not None:
            return cache
        
        # Compute the kernel.
        xp = self._procs[xpkey]
        yp = self._procs[ypkey]
        
        if isinstance(xp, self._ProcKernel) and isinstance(yp, self._ProcKernel):
            kernel = self._crosskernel_kernels(xpkey, ypkey)
        elif isinstance(xp, self._ProcTransf):
            kernel = self._crosskernel_transf_any(xpkey, ypkey)
        elif isinstance(yp, self._ProcTransf):
            kernel = self._crosskernel_transf_any(ypkey, xpkey)._swap()
        elif isinstance(xp, self._ProcLinTransf):
            kernel = self._crosskernel_lintransf_any(xpkey, ypkey)
        elif isinstance(yp, self._ProcLinTransf):
            kernel = self._crosskernel_lintransf_any(ypkey, xpkey)._swap()
        elif isinstance(xp, self._ProcKernelTransf):
            kernel = self._crosskernel_kerneltransf_any(xpkey, ypkey)
        elif isinstance(yp, self._ProcKernelTransf):
            kernel = self._crosskernel_kerneltransf_any(ypkey, xpkey)._swap()
        else: # pragma: no cover
            raise TypeError(f'unrecognized process types {type(xp)!r} and {type(yp)!r}')
        
        # Save cache.
        self._kernels[xpkey, ypkey] = kernel
        self._kernels[ypkey, xpkey] = kernel._swap()
        
        return kernel
        
    def _crosskernel_kernels(self, xpkey, ypkey):
        xp = self._procs[xpkey]
        yp = self._procs[ypkey]
        
        if xp is yp:
            return xp.kernel.linop('diff', xp.deriv, xp.deriv)
        else:
            return self._zerokernel
    
    def _crosskernel_transf_any(self, xpkey, ypkey):
        xp = self._procs[xpkey]
        yp = self._procs[ypkey]
        
        kernelsum = self._zerokernel
        
        for pkey, factor in xp.ops.items():
            kernel = self._crosskernel(pkey, ypkey)
            if kernel is self._zerokernel:
                continue
            
            if not callable(factor):
                factor = (lambda f: lambda _: f)(factor)
            kernel = kernel.linop('rescale', factor, None)
            
            if kernelsum is self._zerokernel:
                kernelsum = kernel
            else:
                kernelsum += kernel
                
        return kernelsum.linop('diff', xp.deriv, 0)
    
    def _crosskernel_lintransf_any(self, xpkey, ypkey):
        xp = self._procs[xpkey]
        yp = self._procs[ypkey]
        
        kernels = [self._crosskernel(pk, ypkey) for pk in xp.keys]
        kernel = _Kernel.CrossKernel._nary(xp.transf, kernels, _Kernel.CrossKernel._side.LEFT)
        kernel = kernel.linop('diff', xp.deriv, 0)
        
        return kernel
    
    def _crosskernel_kerneltransf_any(self, xpkey, ypkey):
        xp = self._procs[xpkey]
        yp = self._procs[ypkey]
        
        if xp is yp:
            basekernel = self._crosskernel(xp.proc, xp.proc)
            # I could avoid handling this case separately but it allows to
            # skip defining two-step transformations A -> CrossAT -> T
        else:
            basekernel = self._crosskernel(xp.proc, ypkey)
        
        if basekernel is self._zerokernel:
            return self._zerokernel
        elif xp is yp:
            return basekernel.linop(xp.transfname, xp.arg)
        else:
            return basekernel.linop(xp.transfname, xp.arg, None)
