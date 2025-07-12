# lsqfitgp/_GP/_elements.py
#
# Copyright (c) 2020, 2022, 2023, 2025, Giacomo Petrillo
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
import functools
import warnings
import math

import gvar
import numpy
from scipy import sparse
import jax
from jax import numpy as jnp

from .. import _Deriv
from .. import _array
from .. import _jaxext
from .. import _gvarext
from .. import _linalg

from . import _base

class GPElements(_base.GPBase):

    def __init__(self, *, checkpos, checksym, posepsfac, halfmatrix):
        self._elements = dict() # key -> _Element
        self._covblocks = dict() # (key, key) -> matrix (2d flattened)
        self._priordict = {} # key -> gvar array (shaped)
        self._checkpositive = bool(checkpos)
        self._posepsfac = float(posepsfac)
        self._checksym = bool(checksym)
        self._halfmatrix = bool(halfmatrix)
        self._dtype = None
        assert not (halfmatrix and checksym)

    def _clone(self):
        newself = super()._clone()
        newself._elements = self._elements.copy()
        newself._covblocks = self._covblocks.copy()
        newself._priordict = self._priordict.copy()
        newself._checkpositive = self._checkpositive
        newself._posepsfac = self._posepsfac
        newself._checksym = self._checksym
        newself._halfmatrix = self._halfmatrix
        newself._dtype = self._dtype
        return newself

    @staticmethod
    def _concatenate(alist):
        """
        Decides to use numpy.concatenate or jnp.concatenate depending on the
        input to support gvars.
        """
        if any(a.dtype == object for a in alist):
            return numpy.concatenate(alist)
        else:
            return jnp.concatenate(alist)

    @staticmethod
    def _triu_indices_and_back(n):
        """
        Return indices to get the upper triangular part of a matrix, and indices
        to convert a flat array of upper triangular elements to a symmetric
        matrix.
        """
        ix, iy = jnp.triu_indices(n)
        q = jnp.empty((n, n), ix.dtype)
        a = jnp.arange(ix.size)
        q = q.at[ix, iy].set(a)
        q = q.at[iy, ix].set(a)
        return ix, iy, q

    class _Element(abc.ABC):
        """
        Abstract class for an object holding information associated to a key in
        a GP object.
        """
    
        @property
        @abc.abstractmethod
        def shape(self): # pragma: no cover
            """Output shape"""
            pass
    
        @property
        def size(self):
            return math.prod(self.shape)

    class _Points(_Element):
        """Points where the process is evaluated"""
    
        def __init__(self, x, deriv, proc):
            assert isinstance(x, (numpy.ndarray, jnp.ndarray, _array.StructuredArray))
            assert isinstance(deriv, _Deriv.Deriv)
            self.x = x
            self.deriv = deriv
            self.proc = proc
    
        @property
        def shape(self):
            return self.x.shape

    class _LinTransf(_Element):
        """Linear transformation of other _Element objects"""
    
        shape = None
    
        def __init__(self, transf, keys, shape):
            self.transf = transf
            self.keys = keys
            self.shape = shape
    
        def matrices(self, gp):
            """
            Matrix coefficients of the transformation (with flattened inputs
            and output)
            """
            elems = [gp._elements[key] for key in self.keys]
            matrices = []
            transf = jax.vmap(self.transf, 0, 0)
            for i, elem in enumerate(elems):
                inputs = [
                    jnp.eye(elem.size).reshape((elem.size,) + elem.shape)
                    if j == i else
                    jnp.zeros((elem.size,) + ej.shape)
                    for j, ej in enumerate(elems)
                ]
                output = transf(*inputs).reshape(elem.size, self.size).T
                matrices.append(output)
            return matrices

    class _Cov(_Element):
        """User-provided covariance matrix block(s)"""
    
        shape = None
    
        def __init__(self, blocks, shape):
            """ blocks = dict (key, key) -> matrix """
            self.blocks = blocks
            self.shape = shape

    @_base.newself
    def addx(self, x, key=None, *, deriv=0, proc=_base.GPBase.DefaultProcess):
        """
        
        Add points where the Gaussian process is evaluated.
        
        The GP object keeps the various x arrays in a dictionary. If ``x`` is an
        array, you have to specify its dictionary key with the ``key`` parameter.
        Otherwise, you can directly pass a dictionary for ``x``.
        
        To specify that on the given ``x`` a derivative of the process instead of
        the process itself should be evaluated, use the parameter ``deriv``.
        
        `addx` may or may not copy the input arrays.
        
        Parameters
        ----------
        x : array or dictionary of arrays
            The points to be added.
        key : hashable
            If ``x`` is an array, the dictionary key under which ``x`` is added.
            Can not be specified if ``x`` is a dictionary.
        deriv : Deriv-like
            Derivative specification. A `Deriv` object or something that
            can be converted to `Deriv`.
        proc : hashable
            The process to be evaluated on the points. If not specified, use
            the default process.
        
        """
        
        # TODO after I implement block solving, add per-key covariance matrix
        # flags.
        
        # TODO add `copy` parameter, default False, to copy the input arrays
        # if they are numpy arrays.
        
        # this interface does not allow adding a single dictionary as x element
        # unless it's wrapped as a 0d numpy array, but this is for the best
        
        deriv = _Deriv.Deriv(deriv)

        if proc not in self._procs:
            raise KeyError(f'process named {proc!r} not found')
                
        if hasattr(x, 'keys'):
            if key is not None:
                raise ValueError('can not specify key if x is a dictionary')
            if None in x:
                raise ValueError('None key in x not allowed')
        else:
            if key is None:
                raise ValueError('x is not dictionary but key is None')
            x = {key: x}
        
        for key in x:
            if key in self._elements:
                raise KeyError('key {!r} already in GP'.format(key))
            
            gx = x[key]
            
            # Convert to JAX array, numpy array or StructuredArray.
            # convert eagerly to jax to avoid problems with tracing.
            gx = _array._asarray_jaxifpossible(gx)

            # Check dtype is compatible with previous arrays.
            # TODO since we never concatenate arrays we could allow a less
            # strict compatibility. In principle we could allow really anything
            # as long as the kernel eats it, but this probably would let bugs
            # through without being really ever useful. What would make sense
            # is checking the dtype structure matches recursively and check
            # concrete dtypes of fields can be casted.
            # TODO result_type is too lax. Examples: str, float -> str,
            # object, float -> object. I should use something like the
            # ordering function in updowncast.py.
            if self._dtype is not None:
                try:
                    self._dtype = numpy.result_type(self._dtype, gx.dtype)
                    # do not use jnp.result_type, it does not support
                    # structured types
                except TypeError:
                    msg = 'x[{!r}].dtype = {!r} not compatible with {!r}'
                    msg = msg.format(key, gx.dtype, self._dtype)
                    raise TypeError(msg)
            else:
                self._dtype = gx.dtype

            # Check that the derivative specifications are compatible with the
            # array data type.
            if gx.dtype.names is None:
                if not deriv.implicit:
                    raise ValueError('x has no fields but derivative has')
            else:
                for dim in deriv:
                    if dim not in gx.dtype.names:
                        raise ValueError(f'deriv field {dim!r} not in x')
            
            self._elements[key] = self._Points(gx, deriv, proc)

    def _get_x_dtype(self):
        """ Get the data type of x points """
        return self._dtype
        
    def addtransf(self, tensors, key, *, axes=1):
        """
        
        Apply a linear transformation to already specified process points. The
        result of the transformation is represented by a new key.
        
        Parameters
        ----------
        tensors : dict
            Dictionary mapping keys of the GP to arrays/scalars. Each array is
            matrix-multiplied with the process array represented by its key,
            while scalars are just multiplied. Finally, the keys are summed
            over.
        key : hashable
            A new key under which the transformation is placed.
        axes : int
            Number of axes to be summed over for matrix multiplication,
            referring to trailing axes for tensors in ` tensors``, and to
            heading axes for process points. Default 1.
        
        Returns
        -------
        gp : GP
            A new GP object with the applied modifications.
       
        Notes
        -----
        The multiplication between the tensors and the process is done with
        np.tensordot with, by default, 1-axis contraction. For >2d arrays this
        is different from numpy's matrix multiplication, which would act on the
        second-to-last dimension of the second array.
        
        """        
        # Note: it may seem nice that when an array has less axes than `axes`,
        # the summation would be restricted only on the existing axes. However
        # this brings about the ambiguous case where only one of the factors has
        # not enough axes. How many axes do you sum over on the other?
        
        # Check axes.
        assert isinstance(axes, int) and axes >= 0, axes
        
        # Check key.
        if key is None:
            raise ValueError('key can not be None')
        if key in self._elements:
            raise KeyError(f'key {key!r} already in GP')
        
        # Check keys.
        for k in tensors:
            if k not in self._elements:
                raise KeyError(k)
        
        # Check tensors and convert them to jax arrays.
        if len(tensors) == 0:
            raise ValueError('empty tensors, undetermined output shape')
        tens = {}
        for k, t in tensors.items():
            t = jnp.asarray(t)
            # no need to check dtype since jax supports only numerical arrays
            with _jaxext.skipifabstract():
                if self._checkfinite and not jnp.all(jnp.isfinite(t)):
                    raise ValueError(f'tensors[{k!r}] contains infs/nans')
            rshape = self._elements[k].shape
            if t.shape and t.shape[t.ndim - axes:] != rshape[:axes]:
                raise ValueError(f'tensors[{k!r}].shape = {t.shape!r} can not be multiplied with shape {rshape!r} with {axes}-axes contraction')
            tens[k] = t
        
        # Check shapes broadcast correctly.
        arrays = tens.values()
        elements = (self._elements[k] for k in tens)
        shapes = (
            t.shape[:t.ndim - axes] + e.shape[axes:] if t.shape else e.shape
            for t, e in zip(arrays, elements)
        )
        try:
            shape = jnp.broadcast_shapes(*shapes)
        except ValueError:
            msg = 'can not broadcast tensors with shapes ['
            msg += ', '.join(repr(t.shape) for t in arrays)
            msg += '] contracted with arrays with shapes ['
            msg += ', '.join(repr(e.shape) for e in elements) + ']'
            raise ValueError(msg)
        
        # Define linear transformation.
        def equiv_lintransf(*args):
            assert len(args) == len(tens)
            out = None
            for a, (k, t) in zip(args, tens.items()):
                if t.shape:
                    b = jnp.tensordot(t, a, axes)
                else:
                    b = t * a
                if out is None:
                    out = b
                else:
                    out = out + b
            return out
        keys = list(tens.keys())
        return self.addlintransf(equiv_lintransf, keys, key, checklin=False)
    
    @_base.newself
    def addlintransf(self, transf, keys, key, *, checklin=None):
        """
        
        Define a finite linear transformation of the evaluated process.
        
        Parameters
        ----------
        transf : callable
            A function with signature ``f(array1, array2, ...) -> array`` which
            computes the linear transformation. The function must be
            jax-traceable, i.e., use jax.numpy instead of numpy.
        keys : sequence
            Keys of parts of the process to be passed as inputs to the
            transformation.
        key : hashable
            The key of the newly defined points.
        checklin : bool
            If True (default), check that the given function is linear in its
            inputs. The default can be overridden at initialization of the GP
            object. Note that an affine function (x -> a + bx) is not linear.
        
        Raises
        ------
        RuntimeError :
            The transformation seems not to be linear. To disable the linearity
            check, initialize the GP with ``checklin=False``.
        
        """
        
        # TODO elementwise operations can be applied more efficiently to
        # primary gvars (tipical case), so the method could use an option
        # `elementwise`. What is the reliable way to check it is indeed
        # elementwise with a single random vector? Zero items of the tangent
        # at random with p=0.5 and check they stay zero? (And of course check
        # the shape is preserved.)
        
        # Check key.
        if key is None:
            raise ValueError('key can not be None')
        if key in self._elements:
            raise KeyError(f'key {key!r} already in GP')
        
        # Check keys.
        for k in keys:
            if k not in self._elements:
                raise KeyError(k)
        
        # Determine shape.
        class ArrayMockup:
            def __init__(self, elem):
                self.shape = elem.shape
                self.dtype = float
        inp = [ArrayMockup(self._elements[k]) for k in keys]
        out = jax.eval_shape(transf, *inp)
        shape = out.shape
        
        # Check that the transformation is linear.
        if checklin is None:
            checklin = self._checklin
        if checklin:
            shapes = [self._elements[k].shape for k in keys]
            self._checklinear(transf, shapes)
        
        self._elements[key] = self._LinTransf(transf, keys, shape)
    
    @_base.newself
    def addcov(self, covblocks, key=None, *, decomps=None):
        """
        
        Add user-defined prior covariance matrix blocks.
        
        Covariance matrices defined with `addcov` represent arbitrary
        finite-dimensional zero-mean Gaussian variables, assumed independent
        from all other variables in the GP object.
                
        Parameters
        ----------
        covblocks : array or dictionary of arrays
            If an array: a covariance matrix (or tensor) to be added under key
            ``key``. If a dictionary: a mapping from pairs of keys to the
            corresponding covariance matrix blocks. A missing off-diagonal
            block in the dictionary is interpreted as a matrix of zeros,
            unless the corresponding transposed block is specified.
        key : hashable
            If ``covblocks`` is an array, the dictionary key under which
            ``covblocks`` is added. Can not be specified if ``covblocks`` is a
            dictionary.
        decomps : Decomposition or dict of Decompositions
            Pre-computed decompositions of (not necessarily all) diagonal
            blocks, as produced by `decompose`. The keys are single
            GP keys and not pairs like in ``covblocks``.
        
        Raises
        ------
        KeyError :
            A key is already used in the GP.
        ValueError :
            ``covblocks`` and/or ``key`` and ``decomps`` are malformed or
            inconsistent.
        TypeError :
            Wrong type of ``covblocks`` or ``decomps``.
        
        """
        
        # TODO maybe allow passing only the lower/upper triangular part for
        # the diagonal blocks, like I meta-allow for out of diagonal blocks?
        
        # TODO with multiple blocks and a single decomp, the decomp could be
        # interpreted as the decomposition of the whole block matrix.
        
        # Check type of `covblocks` and standardize it to dictionary.
        if hasattr(covblocks, 'keys'):
            if key is not None:
                raise ValueError('can not specify key if covblocks is a dictionary')
            if None in covblocks:
                raise ValueError('None key in covblocks not allowed')
            if decomps is not None and not hasattr(decomps, 'keys'):
                raise TypeError('covblocks is dictionary but decomps is not')
        else:
            if key is None:
                raise ValueError('covblocks is not dictionary but key is None')
            covblocks = {(key, key): covblocks}
            if decomps is not None:
                decomps = {key: decomps}
        
        if decomps is None:
            decomps = {}
        
        # Convert blocks to jax arrays and determine shapes from diagonal
        # blocks.
        shapes = {}
        preblocks = {}
        for keys, block in covblocks.items():
            # TODO maybe check that keys is a 2-tuple
            for key in keys:
                if key in self._elements:
                    raise KeyError(f'key {key!r} already in GP')
            xkey, ykey = keys
            if block is None:
                raise TypeError(f'block {keys!r} is None')
                # because jnp.asarray(None) interprets None as nan
                # (see jax issue #14506)
            block = jnp.asarray(block)
            
            if xkey == ykey:
                
                if block.ndim % 2 == 1:
                    raise ValueError(f'diagonal block {key!r} has odd number of axes')
                
                half = block.ndim // 2
                head = block.shape[:half]
                tail = block.shape[half:]
                if head != tail:
                    raise ValueError(f'shape {block.shape!r} of diagonal block {key!r} is not symmetric')
                shapes[xkey] = head
                
                with _jaxext.skipifabstract():
                    if self._checksym and not jnp.allclose(block, block.T):
                        raise ValueError(f'diagonal block {key!r} is not symmetric')
                
            preblocks[keys] = block
        
        # Check decomps is consistent with covblocks.
        for key, dec in decomps.items():
            if key not in shapes:
                raise KeyError(f'key {key!r} in decomps not found in diagonal blocks')
            if not isinstance(dec, _linalg.Decomposition):
                raise TypeError(f'decomps[{key!r}] = {dec!r} is not a decomposition')
            n = math.prod(shapes[key])
            if dec.n != n:
                raise ValueError(f'decomposition matrix size {dec.n} != diagonal block size {n} for key {key!r}')
        
        # Reshape blocks to square matrices and check that the shapes of out of
        # diagonal blocks match those of diagonal ones.
        blocks = {}
        for keys, block in preblocks.items():
            with _jaxext.skipifabstract():
                if self._checkfinite and not jnp.all(jnp.isfinite(block)):
                    raise ValueError(f'block {keys!r} not finite')
            xkey, ykey = keys
            if xkey == ykey:
                size = math.prod(shapes[xkey])
                blocks[keys] = block.reshape((size, size))
            else:
                for key in keys:
                    if key not in shapes:
                        raise KeyError(f'key {key!r} from off-diagonal block {keys!r} not found in diagonal blocks')
                eshape = shapes[xkey] + shapes[ykey]
                if block.shape != eshape:
                    raise ValueError(f'shape {block.shape!r} of block {keys!r} is not {eshape!r} as expected from diagonal blocks')
                xsize = math.prod(shapes[xkey])
                ysize = math.prod(shapes[ykey])
                block = block.reshape((xsize, ysize))
                blocks[keys] = block
                revkeys = keys[::-1]
                blockT = preblocks.get(revkeys)
                if blockT is None:
                    blocks[revkeys] = block.T
        
        # Check symmetry of out of diagonal blocks.
        if self._checksym:
            with _jaxext.skipifabstract():
                for keys, block in blocks.items():
                    xkey, ykey = keys
                    if xkey != ykey:
                        blockT = blocks[ykey, xkey]
                        if not jnp.allclose(block.T, blockT):
                            raise ValueError(f'block {keys!r} is not the transpose of block {revkeys!r}')
        
        # Create _Cov objects.
        for key, shape in shapes.items():
            self._elements[key] = self._Cov(blocks, shape)
            decomp = decomps.get(key)
            if decomp is not None:
                self._decompcache[key,] = decomp

    def _makecovblock_points(self, xkey, ykey):
        x = self._elements[xkey]
        y = self._elements[ykey]
        
        assert isinstance(x, self._Points)
        assert isinstance(y, self._Points)
        
        kernel = self._crosskernel(x.proc, y.proc)
        if kernel is self._zerokernel:
            # TODO handle zero cov block efficiently
            return jnp.zeros((x.size, y.size))
        
        kernel = kernel.linop('diff', x.deriv, y.deriv)
        
        if x is y and not self._checksym and self._halfmatrix:
            ix, iy, back = self._triu_indices_and_back(x.size)
            flat = x.x.reshape(-1)
            ax = flat[ix]
            ay = flat[iy]
            halfcov = kernel(ax, ay)
            cov = halfcov[back]
            # TODO to avoid inefficiencies like in BART, maybe _Kernel should
            # have a method outer(x) that by default simply does self(x[None,
            # :], x[:, None]) but can be overwritten. This halfmatrix impl could
            # be moved there with an option outer(x, *, half=False). To carry
            # over custom implementations of outer, there should be a callable
            # attribute _outer, optionally set at initialization, that is
            # transformed by kernel operations.
        else:
            ax = x.x.reshape(-1)[:, None]
            ay = y.x.reshape(-1)[None, :]
            cov = kernel(ax, ay)
        
        return cov
    
    def _makecovblock_lintransf_any(self, xkey, ykey):
        x = self._elements[xkey]
        y = self._elements[ykey]
        assert isinstance(x, self._LinTransf)
        
        # Gather covariance matrices to be transformed.
        covs = []
        for k in x.keys:
            elem = self._elements[k]
            cov = self._covblock(k, ykey)
            assert cov.shape == (elem.size, y.size)
            cov = cov.reshape(elem.shape + (y.size,))
            covs.append(cov)
        
        # Apply transformation.
        t = jax.vmap(x.transf, -1, -1)
        cov = t(*covs)
        assert cov.shape == x.shape + (y.size,)
        return cov.reshape((x.size, y.size)) # don't leave out the ()!
        # the () probably was an obscure autograd bug, I don't think it will
        # be a problem again with jax
            
    def _makecovblock(self, xkey, ykey):
        x = self._elements[xkey]
        y = self._elements[ykey]
        if isinstance(x, self._Points) and isinstance(y, self._Points):
            cov = self._makecovblock_points(xkey, ykey)
        elif isinstance(x, self._LinTransf):
            cov = self._makecovblock_lintransf_any(xkey, ykey)
        elif isinstance(y, self._LinTransf):
            cov = self._makecovblock_lintransf_any(ykey, xkey)
            cov = cov.T
        elif isinstance(x, self._Cov) and isinstance(y, self._Cov) and x.blocks is y.blocks and (xkey, ykey) in x.blocks:
            cov = x.blocks[xkey, ykey]
        else:
            # TODO handle zero cov block efficiently
            cov = jnp.zeros((x.size, y.size))
        
        with _jaxext.skipifabstract():
            if self._checkfinite and not jnp.all(jnp.isfinite(cov)):
                raise RuntimeError(f'covariance block {(xkey, ykey)!r} is not finite')
            if self._checksym and xkey == ykey and not jnp.allclose(cov, cov.T):
                raise RuntimeError(f'covariance block {(xkey, ykey)!r} is not symmetric')

        return cov
    
    def _covblock(self, row, col):
        
        if (row, col) not in self._covblocks:
            block = self._makecovblock(row, col)
            if row != col:
                if self._checksym:
                    with _jaxext.skipifabstract():
                        blockT = self._makecovblock(col, row)
                        if not jnp.allclose(block.T, blockT):
                            msg = 'covariance block {!r} is not symmetric'
                            raise RuntimeError(msg.format((row, col)))
                self._covblocks[col, row] = block.T
            self._covblocks[row, col] = block

        return self._covblocks[row, col]
        
    def _assemblecovblocks(self, rowkeys, colkeys=None):
        if colkeys is None:
            colkeys = rowkeys
        blocks = [
            [self._covblock(row, col) for col in colkeys]
            for row in rowkeys
        ]
        return jnp.block(blocks)
    
    def _checkpos(self, cov):
        with _jaxext.skipifabstract():
            # eigv = jnp.linalg.eigvalsh(cov)
            # mineigv, maxeigv = jnp.min(eigv), jnp.max(eigv)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'Exited at iteration .+? with accuracies')
                warnings.filterwarnings('ignore', r'Exited postprocessing with accuracies')
                X = numpy.random.randn(len(cov), 1)
                A = numpy.asarray(cov)
                (mineigv,), _ = sparse.linalg.lobpcg(A, X, largest=False)
                (maxeigv,), _ = sparse.linalg.lobpcg(A, X, largest=True)
            assert mineigv <= maxeigv
            if mineigv < 0:
                bound = -len(cov) * jnp.finfo(cov.dtype).eps * maxeigv * self._posepsfac
                if mineigv < bound:
                    msg = 'covariance matrix is not positive definite: '
                    msg += 'mineigv = {:.4g} < {:.4g}'.format(mineigv, bound)
                    raise numpy.linalg.LinAlgError(msg)
    
    _checkpos_cache = functools.cached_property(lambda self: [])
    def _checkpos_keys(self, keys):
        # TODO go back to ancestors of _LinTransf?
        if not self._checkpositive:
            return
        keys = set(keys)
        for prev_keys in self._checkpos_cache:
            if keys.issubset(prev_keys):
                return
        cov = self._assemblecovblocks(list(keys))
        self._checkpos(cov)
        self._checkpos_cache.append(keys)
    
    def _priorpointscov(self, key):
        
        x = self._elements[key]
        classes = (self._Points, self._Cov)
        assert isinstance(x, classes)
        mean = numpy.zeros(x.size)
        cov = self._covblock(key, key).astype(float)
        assert cov.shape == 2 * mean.shape, cov.shape

        ##### temporary fix for gplepage/gvar#49 #####
        cov = numpy.array(cov)
        ##############################################
        
        # get preexisting primary gvars to be correlated with the new ones
        preitems = [
            k
            for k, px in self._elements.items()
            if isinstance(px, classes)
            and k in self._priordict
        ]
        if preitems:
            prex = numpy.concatenate([
                numpy.reshape(self._priordict[k], -1)
                for k in preitems
            ])
            precov = numpy.concatenate([
                self._covblock(k, key).astype(float)
                for k in preitems
            ])
            g = gvar.gvar(mean, cov, prex, precov, fast=True)
        else:
            g = gvar.gvar(mean, cov, fast=True)
        
        return g.reshape(x.shape)
    
    def _priorlintransf(self, key):
        x = self._elements[key]
        assert isinstance(x, self._LinTransf)
        
        # Gather all gvars to be transformed.
        elems = [
            self._prior(k).reshape(-1)
            for k in x.keys
        ]
        g = numpy.concatenate(elems)
        
        # Extract jacobian and split it.
        slices = self._slices(x.keys)
        jac, indices = _gvarext.jacobian(g)
        jacs = [
            jac[s].reshape(self._elements[k].shape + indices.shape)
            for s, k in zip(slices, x.keys)
        ]
        # TODO the jacobian can be extracted much more efficiently when the
        # elements are _Points or _Cov, since in that case the gvars are primary
        # and contiguous within each block, so each jacobian is the identity + a
        # range. Then write a function _gvarext.merge_jacobians to combine
        # them, which also can be optimized knowing the indices are
        # non-overlapping ranges.
        
        # Apply transformation.
        t = jax.vmap(x.transf, -1, -1)
        outjac = t(*jacs)
        assert outjac.shape == x.shape + indices.shape
        
        # Rebuild gvars.
        outg = _gvarext.from_jacobian(numpy.zeros(x.shape), outjac, indices)
        return outg
    
    def _prior(self, key):
        prior = self._priordict.get(key, None)
        if prior is None:
            x = self._elements[key]
            if isinstance(x, (self._Points, self._Cov)):
                prior = self._priorpointscov(key)
            elif isinstance(x, self._LinTransf):
                prior = self._priorlintransf(key)
            else: # pragma: no cover
                raise TypeError(type(x))
            self._priordict[key] = prior
        return prior
    
    def prior(self, key=None, *, raw=False):
        """
        
        Return an array or a dictionary of arrays of gvars representing the
        prior for the Gaussian process. The returned object is not unique but
        the gvars stored inside are, so all the correlations are kept between
        objects returned by different calls to `prior`.
        
        Calling without arguments returns the complete prior as a dictionary.
        If you specify ``key``, only the array for the requested key is returned.
        
        Parameters
        ----------
        key : None, key or list of keys
            Key(s) corresponding to one passed to `addx` or `addtransf`. None
            for all keys.
        raw : bool
            If True, instead of returning a collection of gvars return
            their covariance matrix as would be returned by `gvar.evalcov`.
            Default False.
        
        Returns
        -------
        If raw=False (default):
        
        prior : np.ndarray or dict
            A collection of gvars representing the prior.
        
        If raw=True:
        
        cov : np.ndarray or dict
            The covariance matrix of the prior.
        """
        raw = bool(raw)
        
        if key is None:
            outkeys = list(self._elements)
        elif isinstance(key, list):
            outkeys = key
        else:
            outkeys = None
        
        self._checkpos_keys([key] if outkeys is None else outkeys)
        
        if raw and outkeys is not None:
            return {
                (row, col):
                self._covblock(row, col).reshape(
                    self._elements[row].shape +
                    self._elements[col].shape
                )
                for row in outkeys
                for col in outkeys
            }
        elif raw:
            return self._covblock(key, key).reshape(2 * self._elements[key].shape)
        elif outkeys is not None:
            return {key: self._prior(key) for key in outkeys}
        else:
            return self._prior(key)
        
    def _slices(self, keylist):
        """
        Return list of slices for the positions of flattened arrays
        corresponding to keys in ``keylist`` into their concatenation.
        """
        sizes = [self._elements[key].size for key in keylist]
        stops = numpy.pad(numpy.cumsum(sizes), (1, 0))
        return [slice(stops[i - 1], stops[i]) for i in range(1, len(stops))]
