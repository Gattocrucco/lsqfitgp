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
    
    Object that represents a Gaussian process over arbitrary input.
    
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
        symmetric. If False, only half of the matrix is computed.
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
        Add points where the Gaussian process is evaluated.
    addlintransf
        Define a finite linear transformation of the evaluated process.
    addtransf
        Define a finite linear transformation of the evaluated process with
        explicit coefficients.
    addcov
        Introduce a set of user-provided prior covariance matrix blocks.
    def
        Define a new independent component of the process.
    deftransf
        Define a pointwise linear transformation of the process with explicit
        coefficients.
    deflintransf
        Define a pointwise linear transformation of the process.
    defkernelop
        Define a transformation of the process through a kernel method.
    defderiv
        Define a derivative of the process.
    defxtransf
        Define a process with transformed inputs.
    defrescale
        Define a rescaled process.
    prior
        Compute the prior for the process.
    pred
        Compute the posterior for the process.
    predfromfit
        Like `pred` with ``fromdata=False``.
    predfromdata
        Like `pred` with ``fromdata=True``.
    marginal_likelihood
        Compute the marginal likelihood, i.e., the unconditional probability of
        data.
    decompose
        Decompose a pos. def. matrix.

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
