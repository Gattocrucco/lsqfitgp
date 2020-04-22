from __future__ import division

import itertools
import sys
import builtins

import gvar
from autograd import numpy as np
from autograd.scipy import linalg
from autograd.builtins import isinstance
import numpy # to bypass autograd

from . import _Kernel
from . import _linalg
from . import _array
from . import _Deriv

__all__ = [
    'GP'
]

def _concatenate_noop(alist, **kw):
    """
    Like np.concatenate, but does not make a copy when concatenating only one
    array.
    """
    if len(alist) == 1:
        return np.array(alist[0], copy=False)
    else:
        return np.concatenate(alist, **kw)

def _triu_indices_and_back(n):
    """
    Return indices to get the upper triangular part of a matrix, and indices to
    convert a flat array of upper triangular elements to a symmetric matrix.
    """
    indices = np.triu_indices(n)
    q = np.empty((n, n), int)
    a = np.arange(len(indices[0]))
    q[indices] = a
    q[tuple(reversed(indices))] = a
    return indices, q

def _block_matrix(blocks):
    """
    Like np.block, but is autograd-friendly and avoids a copy when there is
    only one block.
    """
    return _concatenate_noop([_concatenate_noop(row, axis=1) for row in blocks], axis=0)
    # TODO make a bug report to autograd because np.block does not work

def _isarraylike_nostructured(x):
    return isinstance(x, (list, np.ndarray))

def _isarraylike(x):
    return _isarraylike_nostructured(x) or isinstance(x, _array.StructuredArray)

def _asarray(x):
    if isinstance(x, _array.StructuredArray):
        return x
    else:
        return np.array(x, copy=False)
        # TODO won't work with object array-like due to autograd's array bug

def _isdictlike(x):
    return isinstance(x, (dict, gvar.BufferDict))

def _compatible_dtypes(d1, d2):
    """
    Function to check x arrays datatypes passed to GP.addx. If the dtype is
    structured, it checks the structure of the fields is the same, but allows
    casting of concrete dtypes (like, in one array a field can be int, in
    another float, as long as the field name and position is the same).
    Currently not used.
    """
    if d1.names != d2.names or d1.shape != d2.shape:
        return False
    if d1.names is not None:
        for name in d1.names:
            if not _compatible_dtypes(d1.fields[name][0], d2.fields[name][0]):
                return False
    else:
        try:
            np.result_type(d1, d2)
        except TypeError:
            return False
    return True

class _Element:
    """
    Abstract class for an object holding information associated to a key in a
    GP object.
    """
    def __init__(self):
        raise NotImplementedError()
    @property
    def shape(self):
        """Output shape"""
        return NotImplemented
    @property
    def size(self):
        return np.prod(self.shape)

class _Points(_Element):
    """Points where the process is evaluated"""
    def __init__(self, x, deriv):
        assert _isarraylike(x)
        assert isinstance(deriv, _Deriv.Deriv)
        self.x = x
        self.deriv = deriv
    @property
    def shape(self):
        return self.x.shape

class _Transf(_Element):
    """Trasformation over other _Element objects"""
    def __init__(self, tensors, shape):
        assert isinstance(tensors, dict)
        assert isinstance(shape, tuple)
        self.tensors = tensors # dict key -> array
        self._shape = shape
    @property
    def shape(self):
        return self._shape

class GP:
    """
    
    Object that represents a gaussian process over arbitrary input.
    
    Methods that accept arrays/dictionaries also recognize lists,
    StructuredArray and gvar.BufferDict. The output is always a np.ndarray or
    gvar.BufferDict.
    
    Methods
    -------
    addx :
        Add points where the gaussian process is evaluated.
    addtransf :
        Add a linear transformation of the process.
    prior :
        Compute the prior for the process.
    pred :
        Compute the posterior for the process.
    predfromfit, predfromdata :
        Convenience wrappers for `pred`.
    marginal_likelihood :
        Compute the "marginal likelihood", also known as "bayes factor".
    
    """
    
    def __init__(self, covfun, solver='eigcut+', checkpos=True, checksym=True, checkfinite=True, **kw):
        """
        
        Parameters
        ----------
        covfun : Kernel
            An instance of `Kernel` representing the covariance kernel.
        solver : str
            A solver used to invert the covariance matrix. See list below for
            the available solvers. Default is `eigcut+` which is slow but
            robust.
        checkpos : bool
            If True (default), raise a `ValueError` if the covariance matrix
            turns out non positive within numerical error. The check will be
            done only if you use in some way the `gvar` prior.
        checksym : bool
            If True (default), check that the covariance matrix is symmetric.
            If False, only half of the matrix is computed.
        checkfinite : bool
            If True (default), check that the covariance matrix does not
            contain infs or nans.
        
        Solvers
        -------
        eigcut+ :
            Promote small eigenvalues to a minimum value (default). What
            `lsqfit` does by default.
        eigcut- :
            Remove small eigenvalues.
        lowrank :
            Reduce the rank of the matrix. The complexity is O(n^2 r) where
            `n` is the matrix size and `r` the required rank, while other
            algorithms are O(n^3). Slow for small sizes.
        gersh :
            Cholesky decomposition after regularizing the matrix with a
            Gershgorin estimate of the maximum eigenvalue. The fastest of the
            O(n^3) algorithms.
        maxeigv :
            Cholesky decomposition regularizing the matrix with the maximum
            eigenvalue. Slow for small sizes.
        
        Keyword arguments
        -----------------
        eps : positive float
            For solvers `eigcut+`, `eigcut-`, `gersh` and `maxeigv`. Specifies
            the threshold for considering small the eigenvalues, relative to
            the maximum eigenvalue. The default is matrix size * float epsilon.
        rank : positive integer
            For the `lowrank` solver, the target rank. It should be much
            smaller than the matrix size for the method to be convenient.
        
        """
        if not isinstance(covfun, _Kernel.Kernel):
            raise TypeError('covariance function must be of class Kernel')
        self._covfun = covfun
        self._elements = dict() # key -> _Element
        self._canaddx = True
        self._checkpositive = bool(checkpos)
        decomp = {
            'eigcut+': _linalg.EigCutFullRank,
            'eigcut-': _linalg.EigCutLowRank,
            'lowrank': _linalg.ReduceRank,
            'gersh'  : _linalg.CholGersh,
            'maxeigv': _linalg.CholMaxEig
        }[solver]
        self._decompclass = lambda K, **kwargs: decomp(K, **kwargs, **kw)
        self._checkfinite = bool(checkfinite)
        self._checksym = bool(checksym)
    
    # TODO after I implement block solving, add per-key solver option
    def addx(self, x, key=None, deriv=0):
        """
        
        Add points where the gaussian process is evaluated. The GP objects
        keeps the various x arrays in a dictionary. If `x` is an array, you
        have to specify its dictionary key with the `key` parameter. Otherwise,
        you can directly pass a dictionary for `x`.
        
        To specify that on the given `x` a derivative of the process instead of
        the process itself should be evaluated, use the parameter `deriv`.
        
        `addx` never copies the input arrays if they are numpy arrays, so if
        you change their contents before doing something with the GP, the
        change will be reflected on the result. However, after the GP has
        computed internally its covariance matrix, the x are ignored.
        
        If you use in some way the `gvar` prior, e.g. by calling `prior` or
        `pred` using `gvar`s, you can't call `addx` any more.
        
        Parameters
        ----------
        x : array or dictionary of arrays
            The points to be added.
        key :
            If `x` is an array, the dictionary key under which `x` is added.
            Can not be specified if `x` is a dictionary.
        deriv :
            Derivative specification. A `Deriv` object or something that can
            be converted to `Deriv` (see Deriv's help).
        
        """
        if not self._canaddx:
            raise RuntimeError('can not add x any more to this process')
            # TODO remove if I implement the lazy gvar prior
        
        deriv = _Deriv.Deriv(deriv)
        
        if _isarraylike(x):
            if key is None:
                raise ValueError('x is array but key is None')
            x = {key: x}
        elif _isdictlike(x):
            if key is not None:
                raise ValueError('can not specify key if x is a dictionary')
            if None in x:
                raise ValueError('None key in x not allowed')
        else:
            raise TypeError('x must be array or dict')
        
        for key in x:
            if key in self._elements:
                raise RuntimeError('key {} already in GP'.format(repr(key)))
            
            gx = x[key]
            
            # Convert to numpy array or StructuredArray.
            if not _isarraylike(gx):
                raise TypeError('x[{}] is not array or list'.format(repr(key)))
            gx = _asarray(gx)

            # Check it is not empty.
            if not gx.size:
                raise ValueError('x[{}] is empty'.format(repr(key)))

            # Check dtype is compatible with previous arrays.
            # TODO since we never concatenate arrays we could allow a less
            # strict compatibility. In principle we could allow really anything
            # as long as the kernel eats it, but this probably would let bugs
            # through without being really ever useful. What would make sense
            # is checking the dtype structure matches recursively and check
            # concrete dtypes of fields can be casted.
            if hasattr(self, '_dtype'):
                try:
                    self._dtype = np.result_type(self._dtype, gx.dtype)
                except TypeError:
                    raise TypeError('x[{}].dtype = {} which is not compatible with {}'.format(repr(key), repr(gx.dtype), repr(self._dtype)))
            else:
                self._dtype = gx.dtype

            # Check that the derivative specifications are compatible with the
            # array data type.
            if gx.dtype.names is None:
                if not deriv.implicit:
                    raise ValueError('x has not fields but derivative has')
            else:
                for dim in deriv:
                    if dim not in gx.dtype.names:
                        raise ValueError('derivative field {} not in x'.format(repr(dim)))
            
            self._elements[key] = _Points(gx, deriv)
    
    def addtransf(self, tensors, key):
        """
        
        Apply a linear transformation to already specified process points. The
        result of the transformation is represented by a new key.
        
        Parameters
        ----------
        tensors : dict
            Dictionary mapping keys of the GP to arrays. Each array is
            matrix-multiplied with the process array represented by its key.
            Scalars are just multiplied. Finally, the keys are summed over.
        key :
            A new key under which the transformation is placed.
        
        """
        # TODO axes parameter like np.tensordot to allow fancy contractions
        
        # Check key.
        if key is None:
            raise ValueError('key can not be None')
        if key in self._elements:
            raise RuntimeError('key {} already in GP'.format(key))
        
        # Check keys.
        for k in tensors:
            if k not in self._elements:
                raise KeyError(k)
        
        # Check tensors and convert them to numpy arrays.
        for k, t in tensors.items():
            if not np.isscalar(t) and not _isarraylike_nostructured(t):
                raise TypeError('tensor for key {} is not scalar, array or list'.format(k))
            t = np.array(t, copy=False)
            if not np.issubdtype(t.dtype, np.number):
                raise TypeError('tensor for key {} has non-numeric dtype {}'.format(k, t.dtype))
            rshape = self._elements[k].shape
            if t.shape and t.shape[-1] != rshape[0]:
                raise RuntimeError('tensor with shape {} can not be matrix multiplied with shape {} of key {}'.format(t.shape, rshape, k))
            tensors[k] = t
        
        # Compute shape.
        arrays = tensors.values()
        elements = (self._elements[k] for k in tensors)
        shapes = [
            t.shape[:-1] + e.shape[1:] if t.shape else e.shape
            for t, e in zip(arrays, elements)
        ]
        try:
            shape = _array.broadcast_shapes(shapes)
        except ValueError:
            msg = 'can not broadcast tensors with shapes ['
            msg += ', '.join(str(t.shape) for t in arrays)
            msg += '] contracted with arrays with shapes ['
            msg += ', '.join(str(e.shape) for e in elements) + ']'
            raise ValueError(msg)
        
        self._elements[key] = _Transf(tensors, shape)
    
    def _makecovblock_points(self, xkey, ykey):
        x = self._elements[xkey]
        y = self._elements[ykey]
        assert isinstance(x, _Points)
        assert isinstance(y, _Points)
        kernel = self._covfun.diff(x.deriv, y.deriv)
        
        if x is y and not self._checksym:
            indices, back = _triu_indices_and_back(x.size)
            x = x.x.reshape(-1)[indices]
            y = y.x.reshape(-1)[indices]
            halfcov = kernel(x, y)
            cov = halfcov[back]
        else:
            x = x.x.reshape(-1)[:, None]
            y = y.x.reshape(-1)[None, :]
            cov = kernel(x, y)
        
        return cov
    
    def _makecovblock_transf_any(self, xkey, ykey):
        x = self._elements[xkey]
        y = self._elements[ykey]
        assert isinstance(x, _Transf)
        covsum = None
        for key, tensor in x.tensors.items():
            elem = self._elements[key]
            cov = self._covblock(key, ykey)
            assert cov.shape == (elem.size, y.size)
            cov = cov.reshape(elem.shape + y.shape)
            if tensor.shape:
                cov = np.tensordot(tensor, cov, axes=1)
            elif tensor.item != 1:
                cov = tensor * cov
            if covsum is not None:
                covsum = covsum + cov
            else:
                covsum = cov
        assert covsum.shape == x.shape + y.shape
        return covsum.reshape(x.size, y.size)
            
    def _makecovblock(self, xkey, ykey):
        x = self._elements[xkey]
        y = self._elements[ykey]
        if isinstance(x, _Points) and isinstance(y, _Points):
            cov = self._makecovblock_points(xkey, ykey)
        elif isinstance(x, _Transf):
            cov = self._makecovblock_transf_any(xkey, ykey)
        elif isinstance(y, _Transf):
            cov = self._makecovblock_transf_any(ykey, xkey)
            cov = cov.T

        if self._checkfinite and not np.all(np.isfinite(cov)):
            raise RuntimeError('covariance block ({}, {}) is not finite'.format(xkey, ykey))
        if self._checksym and xkey == ykey and not np.allclose(cov, cov.T):
            raise RuntimeError('covariance block ({}, {}) is not symmetric'.format(xkey, ykey))

        return cov

    def _covblock(self, row, col):
        if not self._elements:
            raise RuntimeError('process is empty, add values with `addx`')

        if not hasattr(self, '_covblocks'):
            self._covblocks = dict() # (key1, key2) -> matrix
        
        if (row, col) not in self._covblocks:
            block = self._makecovblock(row, col)
            _linalg.noautograd(block).flags['WRITEABLE'] = False
            if row != col:
                if self._checksym:
                    blockT = self._makecovblock(col, row)
                    if not np.allclose(block.T, blockT):
                        raise RuntimeError('covariance block ({}, {}) is not symmetric'.format(row, col))
                self._covblocks[col, row] = block.T
            self._covblocks[row, col] = block
        
        return self._covblocks[row, col]
        
    def _assemblecovblocks(self, rowkeys, colkeys=None):
        if colkeys is None:
            colkeys = rowkeys
        blocks = [[self._covblock(row, col) for col in colkeys] for row in rowkeys]
        return _block_matrix(blocks)
    
    def _solver(self, keys, ycov=0):
        """
        Return a decomposition of the covariance matrix of the keys in `keys`
        plus the matrix ycov.
        """
        # TODO Block matrix solving. Example: solve a subproblem with kronecker,
        # another plain. Cache decompositions of blocks. Caching is effective
        # with data if I can reuse the decomposition of Kxx to compute the
        # decomposition of Kxx + ycov, i.e. it works in all cases if ycov is
        # scalar, and in some cases if ycov is diagonal. Is there an efficient
        # way to update a Cholesky decomposition if I add a diagonal matrix?
        Kxx = self._assemblecovblocks(keys)
        assert np.allclose(Kxx, Kxx.T) # TODO remove
        return self._decompclass(Kxx + ycov)
        
    def _checkpos(self, cov):
        eigv = linalg.eigvalsh(_linalg.noautograd(cov))
        mineigv = np.min(eigv)
        if mineigv < 0:
            bound = -len(cov) * np.finfo(float).eps * np.max(eigv)
            if mineigv < bound:
                msg = 'covariance matrix is not positive definite: '
                msg += 'mineigv = {:.4g} < {:.4g}'.format(mineigv, bound)
                raise ValueError(msg)
        
    @property
    def _prior(self):
        # TODO I think that gvar internals would let me build the prior
        # one block at a time although everything is correlated, but I don't
        # know how to do it. In case it is possible, replace this with a
        # method that gets the prior for a key without necessarily generating
        # the whole prior. Finally, remove the _canaddx check in GP.addx.
        if not hasattr(self, '_priordict'):
            if self._checkpositive:
                fullcov = self._assemblecovblocks(list(self._elements))
                self._checkpos(fullcov)
            mean = {
                key: np.zeros(x.shape)
                for key, x in self._elements.items()
            }
            cov = {
                (row, col): self._covblock(row, col).reshape(x.shape + y.shape)
                for row, x in self._elements.items()
                for col, y in self._elements.items()
            }
            self._priordict = gvar.gvar(mean, cov)
            self._priordict.buf.flags['WRITEABLE'] = False
            self._canaddx = False
        return self._priordict
    
    def prior(self, key=None, raw=False):
        """
        
        Return an array or a dictionary of arrays of `gvar`s representing the
        prior for the gaussian process. The returned object is not unique but
        the `gvar`s stored inside are, so all the correlations are kept between
        objects returned by different calls to `prior`.
        
        Calling without arguments returns the complete prior as a dictionary.
        If you specify `key`, only the array for the requested key is returned.
        
        Parameters
        ----------
        key : None, key or list of keys
            Key(s) corresponding to one passed to `addx`. None for all keys.
        raw : bool
            If True, instead of returning a collection of `gvar`s return
            their covariance matrix as would be returned by `gvar.evalcov`.
            Default False.
        
        Returns
        -------
        If raw=False (default):
        
        prior : np.ndarray or gvar.BufferDict
            A collection of `gvar`s representing the prior.
        
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
        
        if raw and outkeys is not None:
            return {
                (row, col):
                self._covblock(row, col)
                for row in outkeys
                for col in outkeys
            }
        elif raw:
            return self._covblock(key, key)
        elif outkeys is not None:
            return gvar.BufferDict({
                key: self._prior[key] for key in outkeys
            })
        else:
            return self._prior[key]
        
    def _flatgiven(self, given, givencov):
        if _isarraylike_nostructured(given):
            if len(self._elements) == 1:
                given = {key: given for key in self._elements}
                assert len(given) == 1
                if givencov is not None:
                    assert _isarraylike_nostructured(givencov)
                    givencov = {(key, key): givencov for key in given}
            else:
                raise ValueError('`given` is an array but GP has multiple keys, provide a dictionary')
            
        elif _isdictlike(given):
            if givencov is not None:
                assert _isdictlike(givencov)
            
        else:
            raise TypeError('`given` must be array or dict')
        
        ylist = []
        keylist = []
        for key, l in given.items():
            if key not in self._elements:
                raise KeyError(key)

            if not _isarraylike_nostructured(l):
                raise TypeError('element given[{}] is not list or array'.format(repr(key)))
            
            l = numpy.array(l, copy=False)
            # TODO I use numpy instead of np == autograd.numpy because np.array
            # has this bug:
            # array([object()]) -> array([array([object()])])
            shape = self._elements[key].shape
            if l.shape != shape:
                raise ValueError('given[{}] has shape {} different from shape {}'.format(repr(key), l.shape, shape))
            if l.dtype != object and not np.issubdtype(l.dtype, np.number):
                    raise ValueError('given[{}] has non-numerical dtype {}'.format(repr(key), l.dtype))
            
            ylist.append(l.reshape(-1))
            keylist.append(key)
        
        if givencov is not None:
            covblocks = [
                [
                    givencov[keylist[i], keylist[j]].reshape(ylist[i].shape + ylist[j].shape)
                    for j in range(len(keylist))
                ]
                for i in range(len(keylist))
            ]
        else:
            covblocks = None
            
        return ylist, keylist, covblocks
    
    def _slices(self, keylist):
        """
        Return list of slices for the positions of flattened arrays
        corresponding to keys in `keylist` into their concatenation.
        """
        sizes = [self._elements[key].size for key in keylist]
        stops = np.concatenate([[0], np.cumsum(sizes)])
        return [slice(stops[i - 1], stops[i]) for i in range(1, len(stops))]
    
    def pred(self, given, key=None, givencov=None, fromdata=None, raw=False, keepcorr=None):
        """
        
        Compute the posterior for the gaussian process, either on all points,
        on a subset of points, or conditionally from a subset of points on
        another subset; and either directly from data or from a posterior
        obtained with a fit. The latter case is for when the gaussian process
        was used in a fit with other parameters.
        
        The output is a collection of `gvar`s, either an array or a dictionary
        of arrays. They are properly correlated with `gvar`s returned by
        `prior` and with the input data/fit.
        
        The input is a dictionary of arrays, `given`, with keys corresponding
        to the keys in the GP as added by `addx`. You can pass an array if
        there is only one key in the GP.
        
        Parameters
        ----------
        given : array or dictionary of arrays
            The data or fit result for some/all of the points in the GP.
            The arrays can contain either `gvar`s or normal numbers, the latter
            being equivalent to zero-uncertainty `gvar`s.
        key : None, key or list of keys
            If None, compute the posterior for all points in the GP (also those
            used in `given`). Otherwise only those specified by key.
        givencov : array or dictionary of arrays
            Covariance matrix of `given`. If not specified, the covariance
            is extracted from `given` with `gvar.evalcov(given)`.
        fromdata : bool
            Mandatory. Specify if the contents of `given` are data or already
            a posterior.
        raw : bool (default False)
            If True, instead of returning a collection of `gvar`s, return
            the mean and the covariance. When the mean is a dictionary, the
            covariance is a dictionary whose keys are pairs of keys of the
            mean (the same format used by `gvar.evalcov`).
        keepcorr : bool
            If True (default), the returned `gvar`s are correlated with the
            prior and the data/fit. If False, they have the correct covariance
            between themselves, but are independent from all other preexisting
            `gvar`s.
        
        Returns
        -------
        If raw=False (default):
        
        posterior : array or dictionary of arrays
            A collections of `gvar`s representing the posterior.
        
        If raw=True:
        
        pmean : array or dictionary of arrays
            The mean of the posterior. Equivalent to `gvar.mean(posterior)`.
        pcov : 2D array or dictionary of 2D arrays
            The covariance matrix of the posterior. If `pmean` is a dictionary,
            the keys of `pcov` are pairs of keys of `pmean`. Equivalent to
            `gvar.evalcov(posterior)`.
        
        """
        
        if fromdata is None:
            raise ValueError('you must specify if `given` is data or fit result')
        fromdata = bool(fromdata)
        raw = bool(raw)
        if keepcorr is None:
            keepcorr = not raw
        if keepcorr and raw:
            raise ValueError('both keepcorr=True and raw=True')
        
        strip = False
        if key is None:
            outkeys = list(self._elements)
        elif isinstance(key, list):
            outkeys = key
        else:
            outkeys = [key]
            strip = True
        outslices = self._slices(outkeys)
        
        ylist, inkeys, ycovblocks = self._flatgiven(given, givencov)
        y = _concatenate_noop(ylist)
        
        # I think it is good to have Kxxs row-major and Kxsx column-major
        Kxxs = self._assemblecovblocks(inkeys, outkeys)
        Kxsx = Kxxs.T
        
        # TODO remove
        assert np.allclose(Kxxs, self._assemblecovblocks(outkeys, inkeys).T)
        
        if ycovblocks is not None:
            ycov = _block_matrix(ycovblocks)
        elif (fromdata or raw or not keepcorr) and y.dtype == object:
            ycov = gvar.evalcov(gvar.gvar(y))
            # TODO use evalcov_block? If fromdata=True, it doesn't really
            # make a difference because I just use ycov in Kxx + ycov. If
            # fromdata=False, typically ycov will be dense. The only reason
            # is that maybe gvar.evalcov is not optimized to handle non-dense
            # cases, but in this case I should modify gvar.evalcov. Case under
            # which it makes a difference even for fromdata=True: I implement
            # caching of decompositions and ycov is diagonal. Or: ycov is zero,
            # then evalcov builds a matrix of zeros anyway -> make a method
            # to replace evalcov.
            if self._checkfinite and not np.all(np.isfinite(ycov)):
                raise ValueError('covariance matrix of `given` is not finite')
        else:
            ycov = 0
        
        if raw or not keepcorr:
            
            Kxsxs = self._assemblecovblocks(outkeys)

            assert np.allclose(Kxsxs, Kxsxs.T) # TODO remove
            
            ymean = gvar.mean(y)
            if self._checkfinite and not np.all(np.isfinite(ymean)):
                raise ValueError('mean of `given` is not finite')
            
            if fromdata:
                solver = self._solver(inkeys, ycov)
                cov = Kxsxs - solver.quad(Kxxs)
                mean = solver.solve(Kxxs).T @ ymean
            else:
                solver = self._solver(inkeys)
                A = solver.solve(Kxxs).T
                if np.isscalar(ycov) and ycov == 0:
                    cov = Kxsxs - solver.quad(Kxxs)
                elif np.isscalar(ycov) or len(ycov.shape) == 1:
                    ycov_mat = np.reshape(ycov, (1, -1))
                    cov = Kxsxs + (A * ycov_mat) @ A.T - solver.quad(Kxxs)
                else:
                    cov = Kxsxs + A @ ycov @ A.T - solver.quad(Kxxs)
                # equivalent formula:
                # cov = Kxsxs - A @ (Kxx - ycov) @ A.T
                mean = A @ ymean
            
        else: # (keepcorr and not raw)        
            yplist = [self._prior[key].reshape(-1) for key in inkeys]
            ysplist = [self._prior[key].reshape(-1) for key in outkeys]
            yp = _concatenate_noop(yplist)
            ysp = _concatenate_noop(ysplist)
            
            mat = ycov if fromdata else 0
            flatout = Kxsx @ self._solver(inkeys, mat).usolve(y - yp) + ysp
        
        if raw and not strip:
            meandict = {
                key: mean[slic].reshape(self._elements[key].shape)
                for key, slic in zip(outkeys, outslices)
            }
            
            covdict = {
                (row, col):
                cov[rowslice, colslice].reshape(self._elements[row].shape + self._elements[col].shape)
                for row, rowslice in zip(outkeys, outslices)
                for col, colslice in zip(outkeys, outslices)
            }
            
            return meandict, covdict
            
        elif raw:
            assert len(outkeys) == 1
            mean = mean.reshape(self._elements[outkeys[0]].shape)
            cov = cov.reshape(2 * self._elements[outkeys[0]].shape)
            return mean, cov
        
        elif not keepcorr:
            flatout = gvar.gvar(mean, cov)
        
        if not strip:
            return gvar.BufferDict({
                key: flatout[slic].reshape(self._elements[key].shape)
                for key, slic in zip(outkeys, outslices)
            })
        else:
            assert len(outkeys) == 1
            return flatout.reshape(self._elements[outkeys[0]].shape)
        
    def predfromfit(self, *args, **kw):
        """
        Like `pred` with `fromdata=False`.
        """
        return self.pred(*args, fromdata=False, **kw)
    
    def predfromdata(self, *args, **kw):
        """
        Like `pred` with `fromdata=True`.
        """
        return self.pred(*args, fromdata=True, **kw)

    def marginal_likelihood(self, given, givencov=None):
        """
        
        Compute (the logarithm of) the marginal likelihood given data, i.e. the
        probability of the data conditioned on the gaussian process prior and
        data error.
        
        Unlike `pred()`, you can't compute this with a fit result instead of
        data. If you used the gaussian process as latent variable in a fit,
        use the whole fit to compute the marginal likelihood. E.g. `lsqfit`
        always computes the logGBF (it's the same thing).
        
        The input is an array or dictionary of arrays, `given`. You can pass an
        array only if the GP has only one key. The contents of `given`
        represent the input data.
                
        Parameters
        ----------
        given : array or dictionary of arrays
            The data for some/all of the points in the GP. The arrays can
            contain either `gvar`s or normal numbers, the latter being
            equivalent to zero-uncertainty `gvar`s.
        givencov : array or dictionary of arrays
            Covariance matrix of `given`. If not specified, the covariance
            is extracted from `given` with `gvar.evalcov(given)`.
        
        Returns
        -------
        marglike : scalar
            The logarithm of the marginal likelihood.
            
        """        
        ylist, inkeys, ycovblocks = self._flatgiven(given, givencov)
        y = _concatenate_noop(ylist)

        if ycovblocks is not None:
            ycov = _block_matrix(ycovblocks)
            ymean = gvar.mean(y)
        elif y.dtype == object:
            gvary = gvar.gvar(y)
            # TODO this gvar.gvar(y) is here because gvar.evalcov is picky and
            # won't accept a non-gvar scalar in the array. I should modify
            # gvar.evalcov, since gvar.mean and gvar.sdev accept non-gvars.
            ycov = gvar.evalcov(gvary)
            ymean = gvar.mean(gvary)
        else:
            ycov = 0
            ymean = y
        
        if self._checkfinite and not np.all(np.isfinite(ymean)):
            raise ValueError('mean of `given` is not finite')
        if self._checkfinite and not np.all(np.isfinite(ycov)):
            raise ValueError('covariance matrix of `given` is not finite')
        
        decomp = self._solver(inkeys, ycov)
        return -1/2 * (decomp.quad(ymean) + decomp.logdet() + len(y) * np.log(2 * np.pi))
