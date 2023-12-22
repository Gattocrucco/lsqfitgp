# lsqfitgp/_gvarext/__init__.py
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

import contextlib

import gvar

from ._jacobian import jacobian, from_jacobian
from ._tabulate import tabulate_together
from ._ufunc import gvar_gufunc
from ._format import uformat, fmtspec_kwargs, gvar_format

@contextlib.contextmanager
def switchgvar():
    """
    Context manager to keep new gvars in a separate pool.

    Creating new primary gvars fills up memory permanently. This context manager
    keeps the gvars created within its context in a separate pool that is freed
    when all such gvars are deleted. They can not be mixed in operations with
    other gvars created outside of the context.
    
    Returns
    -------
    gvar : gvar.GVarFactory
        The new gvar-creating function that uses a new pool. The change is also
        reflected in the global `gvar.gvar`.

    See also
    --------
    gvar.switch_gvar, gvar.restore_gvar

    Examples
    --------

    >>> x = gvar.gvar(0, 1)
    >>> with lgp.switchgvar():
    >>>     y = gvar.gvar(0, 1)
    >>>     z = gvar.gvar(0, 1)
    >>> w = gvar.gvar(0, 1)
    >>> q = y + z  # allowed, y and z created in the same pool
    >>> p = x + w  # allowed, x and w created in the same pool
    >>> h = x + y  # x and y created in different pools: this will silently
    ...            # fail and possibly crash python immediately or later on

    """
    try:
        yield gvar.switch_gvar()
    finally:
        gvar.restore_gvar()
