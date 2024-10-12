# lsqfitgp/_special/_gamma.py
#
# Copyright (c) 2022, 2023, 2024, Giacomo Petrillo
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

from jax import numpy as jnp
from jax.scipy import special as jspecial

from .. import _jaxext

def sgngamma(x):
    return jnp.where((x > 0) | (x % 2 < 1), 1, -1)

def gamma(x):
    return sgngamma(x) * jnp.exp(jspecial.gammaln(x))

def poch(x, k):
    return jnp.exp(jspecial.gammaln(x + k) - jspecial.gammaln(x)) # DLMF 5.2.5
    # TODO does not handle properly special cases with x and/or k nonpositive
    # integers

def gamma_incr(x, e):
    """
    Compute Γ(x+e) / (Γ(x)Γ(1+e)) - 1 accurately for x >= 2 and |e| < 1/2
    """
    
    # G(x + e) / G(x)G(1+e) - 1 =
    # = expm1(log(G(x + e) / G(x)G(1+e))) =
    # = expm1(log G(x + e) - log G(x) - log G(1 + e))
    
    t = _jaxext.float_type(x, e)
    n = 23 if t == jnp.float64 else 10
    # n such that 1/2^n 1/n! d^n/dx^n log G(x) |_x=2 < eps
    k = jnp.arange(n).reshape((n,) + (1,) * max(x.ndim, e.ndim))
    coef = jspecial.polygamma(k, x)
    fact = jnp.cumprod(1 + k, 0, t)
    coef /= fact
    gammaln = e * jnp.polyval(coef[::-1], e)
    return jnp.expm1(gammaln - gammaln1(e))
    
    # Like gammaln1, I thought that writing as log Γ(1+x+e) - log Γ(1+x) +
    # - log1p(e/x) would increase accuracy, instead it deteriorates

def gammaln1(x):
    """ compute log Γ(1+x) accurately for |x| <= 1/2 """
    
    t = _jaxext.float_type(x)
    coef = jnp.array(_gammaln1_coef_1[:48], t) # 48 found by trial and error
    return x * jnp.polyval(coef[::-1], x)
    
    # I thought that writing this as log Γ(2+x) - log1p(x) would be more
    # accurate but it isn't, probably there's a cancellation for some values of
    # x

_gammaln1_coef_1 = [        # = _gen_gammaln1_coef(53, 1)
    -0.5772156649015329,
    0.8224670334241132,
    -0.40068563438653143,
    0.27058080842778454,
    -0.20738555102867398,
    0.1695571769974082,
    -0.1440498967688461,
    0.12550966952474304,
    -0.11133426586956469,
    0.1000994575127818,
    -0.09095401714582904,
    0.083353840546109,
    -0.0769325164113522,
    0.07143294629536133,
    -0.06666870588242046,
    0.06250095514121304,
    -0.058823978658684585,
    0.055555767627403614,
    -0.05263167937961666,
    0.05000004769810169,
    -0.047619070330142226,
    0.04545455629320467,
    -0.04347826605304026,
    0.04166666915034121,
    -0.04000000119214014,
    0.03846153903467518,
    -0.037037037312989324,
    0.035714285847333355,
    -0.034482758684919304,
    0.03333333336437758,
    -0.03225806453115042,
    0.03125000000727597,
    -0.030303030306558044,
    0.029411764707594344,
    -0.02857142857226011,
    0.027777777778181998,
    -0.027027027027223673,
    0.02631578947377995,
    -0.025641025641072283,
    0.025000000000022737,
    -0.024390243902450117,
    0.023809523809529224,
    -0.023255813953491015,
    0.02272727272727402,
    -0.022222222222222855,
    0.021739130434782917,
    -0.021276595744681003,
    0.02083333333333341,
    -0.02040816326530616,
    0.020000000000000018,
    -0.019607843137254912,
    0.019230769230769235,
    -0.01886792452830189,
]

def _gen_gammaln1_coef(n, x): # pragma: no cover
    """ compute Taylor coefficients of log Γ(x) """
    import mpmath as mp
    with mp.workdps(32):
        return [float(mp.polygamma(k, x) / mp.fac(k + 1)) for k in range(n)]
