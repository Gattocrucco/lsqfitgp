# lsqfitgp/_gvarext/_format.py
#
# Copyright (c) 2023, 2024, Giacomo Petrillo
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

import math
import re
import contextlib

import gvar

def exponent(x):
    return int(math.floor(math.log10(abs(x))))

def int_mantissa(x, n, e):
    return round(x * 10 ** (n - 1 - e))

def naive_ndigits(x, n):
    log10x = math.log10(abs(x))
    n_int = int(math.floor(n))
    n_frac = n - n_int
    log10x_frac = log10x - math.floor(log10x)
    return n_int + (log10x_frac < n_frac)

def ndigits(x, n):
    ndig = naive_ndigits(x, n)
    xexp = exponent(x)
    rounded_x = int_mantissa(x, ndig, xexp) * 10 ** xexp
    if rounded_x > x:
        rounded_ndig = naive_ndigits(rounded_x, n)
        if rounded_ndig > ndig:
            x = rounded_x
            ndig = rounded_ndig
    return x, ndig

def mantissa(x, n, e):
    m = int_mantissa(x, n, e)
    s = str(abs(int(m)))
    assert len(s) == n or len(s) == n + 1 or (m == 0 and n < 0)
    if n >= 1 and len(s) == n + 1:
        e = e + 1
        s = s[:-1]
    return s, e

def insert_dot(s, n, e, *, add_leading_zeros=True, trailing_zero_char='0'):
    e = e + len(s) - n
    n = len(s)
    if e >= n - 1:
        s = s + trailing_zero_char * (e - n + 1)
    elif e >= 0:
        s = s[:1 + e] + '.' + s[1 + e:]
    elif e <= -1 and add_leading_zeros:
        s = '0' * -e + s
        s = s[:1] + '.' + s[1:]
    return s

def tostring(x):
    return '0' if x == 0 else f'{x:#.6g}'

def uformat(mu, s, errdig=2, sep=None, *,
    shareexp=True,
    outersign=False,
    uniexp=False,
    minnegexp=6,
    minposexp=4,
    padzero=None,
    possign=False,
    ):
    """
    Format a number with uncertainty.
    
    Parameters
    ----------
    mu : number
        The central value.
    s : number
        The error.
    errdig : number
        The number of digits of the error to be shown. Must be >= 1. It can be
        a noninteger, in which case the number of digits switches between the
        lower nearest integer to the upper nearest integer as the first decimal
        digit (after rounding) crosses 10 raised to the fractional part of
        `errdig`. Default 1.5.
    sep : None or str
        The separator put between the central value and the error. Eventual
        spaces must be included. If None, put the error between parentheses,
        sharing decimal places/exponential notation with the central value.
        Default None.
    shareexp : bool, default True
        Applies if `sep` is not ``None``. When using exponential notation,
        whether to share the exponent between central value and error with outer
        parentheses.
    outersign : bool
        Applied when sep is not None and shareexp is True. Whether to put the
        sign outside or within the parentheses. Default False
    uniexp : bool
        When using exponential notation, whether to use unicode characters
        instead of the standard ASCII notation. Default False.
    minnegexp : int
        The number of places after the comma at which the notation switches
        to exponential notation. Default 4. The number of places from the
        greater between central value and error is considered.
    minposexp : int
        The power of ten of the least significant digit at which exponential
        notation is used. Default 0. Setting higher values may force padding
        the error with zeros, depending on `errdig`.
    padzero : str, optional
        If provided, a character representing 0 to pad with when not using
        exponential notation due to `minposexp` even if the least significant
        digit is not on the units, instead of showing more actual digits than
        those specified.
    possign : bool, default False
        Whether to put a `+` before the central value when it is positive.

    Returns
    -------
    r : str
        The quantity (mu +/- s) nicely formatted.
    """
    if errdig < 1:
        raise ValueError('errdig < 1')
    if not math.isfinite(mu) or not math.isfinite(s) or s <= 0:
        if sep is None:
            return f'{tostring(mu)}({tostring(s)})'
        else:
            return f'{tostring(mu)}{sep}{tostring(s)}'
    
    s, sndig = ndigits(s, errdig)
    sexp = exponent(s)
    muexp = exponent(mu) if mu != 0 else sexp - sndig - 1
    smant, sexp = mantissa(s, sndig, sexp)
    mundig = sndig + muexp - sexp
    mumant, muexp = mantissa(mu, mundig, muexp)
    musign = '-' if mu < 0 else '+' if possign else ''
    
    if mundig >= sndig:
        use_exp = muexp >= mundig + minposexp or muexp <= -minnegexp
        base_exp = muexp
    else:
        use_exp = sexp >= sndig + minposexp or sexp <= -minnegexp
        base_exp = sexp
    
    if use_exp:
        mumant = insert_dot(mumant, mundig, muexp - base_exp)
        smant = insert_dot(smant, sndig, sexp - base_exp, add_leading_zeros=sep is not None)
    elif base_exp >= max(mundig, sndig) and padzero is None:
        mumant = str(abs(round(mu)))
        smant = str(abs(round(s)))
    else:
        zerochar = '0' if padzero is None else padzero
        mumant = insert_dot(mumant, mundig, muexp, trailing_zero_char=zerochar)
        if len(mumant) >= 2 and mumant.startswith('0') and all(c == zerochar for c in mumant[1:]):
            mumant = zerochar + mumant[1:]
        smant = insert_dot(smant, sndig, sexp, add_leading_zeros=sep is not None, trailing_zero_char=zerochar)
    
    if not outersign:
        mumant = musign + mumant
    
    if use_exp:
        if uniexp:
            asc = '0123456789+-'
            uni = '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻'
            table = str.maketrans(asc, uni)
            exp = str(base_exp).translate(table)
            suffix = '×10' + exp
        else:
            suffix = f'e{base_exp:+}'
        if sep is None:
            r = mumant + '(' + smant + ')' + suffix
        elif shareexp:
            r = '(' + mumant + sep + smant + ')' + suffix
        else:
            r = mumant + suffix + sep + smant + suffix
    elif sep is None:
        r = mumant + '(' + smant + ')'
    else:
        r = mumant + sep + smant
    
    if outersign:
        r = musign + r
    
    return r

def fmtspec_kwargs(spec):
    """
    Parse a string formatting pattern to be used with `uformat`.

    Parameters
    ----------
    spec : str
        The format specification. It must follow the format

            [options](error digits)[:minimum exponent](mode)

        where brackets indicate optional parts.

    Returns
    -------
    kwargs : dict
        The keyword arguments to be passed to `uformat`.

    Notes
    -----
    Full format:

    Options: any combination these characters:
    
    '+' :
        Put a '+' before positive central values.
    '-' :
        Put the sign outside the parentheses used to group the central value and
        error mantissas in exponential notation.
    '#' :
        Do not show non-significative digits at all costs, replacing them with
        lowercase 'o', representing a rounding 0 rather than a significative 0.
    '$' :
        In exponential notation, repeat the exponent for the central value and
        error.

    Error digits: a decimal number expressing the number of leading error digits
    to show. Non-integer values indicate that the number of digits switches from
    the floor to the ceil at some value of the mantissa.

    Minimum exponent: a decimal number expressing the minimum exponent at which
    exponential notation is used.

    Mode: one of these characters:
    
    'p' :
        Put the error between parentheses.
    's' :
        Separate the central value from the error with '+/-'.
    'u' :
        Separate the central value from the error with '±'.
    'U' :
        Separate the central value from the error with '±', and use unicode
        superscript characters for exponential notation.
    """
    pat = r'([-+#$]*)(\d*\.?\d*)(:\d+)?(p|s|u|U)'
    m = re.fullmatch(pat, spec)
    if not m:
        raise ValueError(f'format specification {spec!r} not understood, format is r"{pat}"')
    kw = {}
    options = m.group(1)
    kw['possign'] = '+' in options
    kw['outersign'] = '-' in options
    kw['padzero'] = 'o' if '#' in options else None
    kw['shareexp'] = '$' not in options
    if m.group(2):
        kw['errdig'] = float(m.group(2))
    else:
        kw['errdig'] = 1.5
    if m.group(3):
        nexp = int(m.group(3)[1:])
    else:
        nexp = 5
    kw['minposexp'] = max(0, nexp - math.floor(kw['errdig']))
    kw['minnegexp'] = nexp
    mode = m.group(4)
    kw['sep'] = dict(p=None, s=' +/- ', u=' ± ', U=' ± ')[mode]
    kw['uniexp'] = mode == 'U'
    return kw

def gvar_formatter(g, spec):
    """
    A formatter for `gvar.GVar.set` that uses `uformat`.
    """
    mu = gvar.mean(g)
    s = gvar.sdev(g)
    kw = fmtspec_kwargs(spec)
    return uformat(mu, s, **kw)

@contextlib.contextmanager
def gvar_format(spec=None, *, lsqfitgp_format=True):
    """
    Context manager to set the default format specification of gvars.
    
    Parameters
    ----------
    spec : str, optional
        The format specification. If not specified and `lsqfitgp_format` is
        ``True``, use ``'#1.5p'``.
    lsqfitgp_format : bool, default True
        Whether to use a modified version of the `gvar` formatting
        specification, provided by `lsqfitgp`.

    Notes
    -----
    See `fmtspec_kwargs` for the format specification, and `uformat` for all
    details.

    See also
    --------
    gvar.fmt, gvar.GVar.set
    """
    if lsqfitgp_format:
        if spec is None:
            spec = '#1.5p'
        def formatter(g, spec, defaultspec=spec):
            if spec == '':
                spec = defaultspec
            return gvar_formatter(g, spec)
        kw = dict(formatter=formatter)
    else:
        kw = {} if spec is None else dict(default_format=spec)
    try:
        old_settings = gvar.GVar.set(**kw)
        yield
    finally:
        gvar.GVar.set(**old_settings)
