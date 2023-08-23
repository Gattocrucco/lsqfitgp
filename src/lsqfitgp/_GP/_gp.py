# lsqfitgp/_GP/_gp.py
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

from . import _base, _compute, _elements, _processes

class GP(_compute.GPCompute, _elements.GPElements, _processes.GPProcesses):
    """
    
    Object that represents a Gaussian process.

    A `GP` is structured like a pair of dictionaries, one for "processes", and
    one for "elements". The processes represent independent Gaussian processes,
    i.e., infinite-dimensional Normally distributed variables. The elements
    represent finite-dimensional Normal variables, typically finite subsets of
    the processes.

    The methods to define processes start with "def", while those to define
    elements starts with "add". The basic methods are `defproc` and `addx`.
    
    A `GP` object is immutable. Methods that modify the Gaussian process return
    a new object which differs only in the requested modification, leaving the
    original untouched.

    Parameters
    ----------
    covfun : Kernel or None
        An instance of `Kernel` representing the covariance kernel of the
        default process of the GP object. It can be left unspecified.
    solver : str
        The algorithm used to decompose the prior covariance matrix. See
        `decompose` for the available solvers. Default is ``'chol'``.
    checkpos : bool
        If True (default), raise a `LinAlgError` if the prior covariance matrix
        turns out non positive within numerical error.
    checksym : bool
        If True (default), check that the prior covariance matrix is
        symmetric.
    checkfinite : bool
        If True (default), check that the prior covariance matrix does not
        contain infs or nans.
    checklin : bool
        If True (default), the method `addlintransf` will check that the
        given transformation is linear on a random input tensor.
    posepsfac : number
        The threshold used to check if the prior covariance matrix is positive
        definite is multiplied by this factor (default 1).
    halfmatrix : bool
        If True and ``checksym=False``, compute only half of the covariance
        matrices by unrolling their lower triangular part as flat arrays. This
        may actually be a large performance hit if the input arrays have large
        item size or if the implementation of the kernel takes advantage of
        non-broadcasted inputs, so it is False by default.
    **kw
        Additional keyword arguments are passed to the solver, see `decompose`.

    Methods
    -------
    addx
        Add points where a process is evaluated.
    addlintransf
        Define a finite linear transformation.
    addtransf
        Define a finite linear transformation with explicit coefficients.
    addcov
        Introduce a set of user-provided prior covariance matrix blocks.
    defproc
        Define a new independent process with a kernel.
    defproclintransf
        Define a pointwise linear transformation.
    defproctransf
        Define a pointwise linear transformation with explicit coefficients.
    defkernelop
        Define an arbitrary linear transformation through a kernel method.
    defprocderiv
        Define a process as the derivative of another one.
    defprocxtransf
        Define a process with transformed inputs.
    defprocrescale
        Rescale a process.
    prior
        Compute the prior.
    pred
        Compute the posterior.
    predfromfit
        Like `pred` with ``fromdata=False``.
    predfromdata
        Like `pred` with ``fromdata=True``.
    marginal_likelihood
        Compute the probability density.
    decompose
        Decompose a pos. semidef. matrix.

    Attributes
    ----------
    DefaultProcess :
        The unique process key used to represent the default process.
    
    """

    def __init__(self,
        covfun=None,
        *,
        solver='chol',
        checkpos=True,
        checksym=True,
        checkfinite=True,
        checklin=True,
        posepsfac=1,
        halfmatrix=False,
        **kw,
    ):
        _base.GPBase.__init__(self, checkfinite=checkfinite, checklin=checklin)
        _processes.GPProcesses.__init__(self, covfun=covfun)
        _elements.GPElements.__init__(self, checkpos=checkpos, checksym=checksym, posepsfac=posepsfac, halfmatrix=halfmatrix)
        _compute.GPCompute.__init__(self, solver=solver, solverkw=kw)
