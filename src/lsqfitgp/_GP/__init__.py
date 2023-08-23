# lsqfitgp/_GP/__init__.py
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

from ._gp import GP

# TODO planned rewrite:
# Methods:
#     Processes
#         .defproc(key, *args, *, deriv=0)
#             *args is either (mean, kernel) or just (kernel,). The mean is an
#             ordinary callable.
#     Inspection
#         .elements() return a read-only mapping of keys -> named tuples with
#             fields ancestors=(tuple of keys), kind=str (x, cov, transf), shape,
#             x (array or None), proc (proc key or None). The mapping generates
#             the descriptions on the fly in __getitem__.
#         .processes() analogous with ancestors, kind (kernel, transf), kernel
#             (Kernel or None)
#         .dtype the x dtype, None if no x passed yet
#         .kernel(*args) if one key a Kernel, if two a CrossKernel
#         .meanfunc(prockey)
#         .__repr__() -> multiline readable description
#     Mean functions
#         .__init__(*args, ...) supports either only the kernel or the mean and
#              the kernel of the DefaultProcess.
#         .add(*args) arrays/dicts mean and cov or just cov, replaces .addcov.
#     Conditioning
#         .condition(given, givencov=None) -> new GP, with appended new keys and
#             values to condition on. given and givencov keep dict layout like
#             now.
#         .extend(given, givencov=None) -> new GP, like current predfromfit. It
#             sets a flag about the kind of conditioning in the list of given
#             stuff. I have to derive how to condition and extend together.
#     Distribution access
#         .mean(key) -> array, the key can be a pytree
#         .cov(*args) -> if one arg: square covariance, if two args: cross
#             covariance. The keys can be pytrees, and the result is the pytree
#             product with matrices as leaves.
#         .gvars(key, keepcorr=True) -> tree of arrays of gvars. keepcorr=True
#             won't work if condition was not passed gvars, so this default raises
#             an error if that happens.
#         .sample(key, n=1) -> array, key can be a pytree. caches the decomp?
#     Density
#         ._decomp(y, ycov=None) -> resid, cov (like _prior_decomp now)
#         .logpdf(y, ycov=None) -> like marginal_likelihood now

# TODO methods to implement linear models. The elegant way to do this is with
# kernels taking a formula, but that would not take advantage of the fact that
# the kernel is low-rank if I re-implement Woodbury. I need to define an element
# for the coefficients, and then the process works by definining finite
# transformations of this element instead of going through the usual kernel
# route.
#
# A more general route would be `features` mechanism for kernels: a kernel can
# provide itself a way to evaluate a covariance matrix as a low-rank product.
