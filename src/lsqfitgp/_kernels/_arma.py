# lsqfitgp/_kernels/_arma.py
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

import jax
from jax import numpy as jnp
from jax import lax
import numpy

from .. import _linalg
from .._linalg import _toeplitz
from .. import _jaxext
from .._Kernel import stationarykernel

# use positive delta because negative indices wrap around
@stationarykernel(derivable=False, maxdim=1, input='abs')
def MA(delta, w=None, norm=False):
    """
    Discrete moving average kernel.
    
    .. math::
        k(\\Delta) = \\sum_{k=|\\Delta|}^{n-1} w_k w_{k-|\\Delta|},
        \\quad \\mathbf w = (w_0, \\ldots, w_{n-1}).
    
    The inputs must be integers. It is the autocovariance function of a moving
    average with weights :math:`\\mathbf w` applied to white noise:
    
    .. math::
        k(i, j) &= \\operatorname{Cov}[y_i, y_j], \\\\
        y_i &= \\sum_{k=0}^{n-1} w_k \\epsilon_{i-k}, \\\\
        \\operatorname{Cov}[\\epsilon_i,\\epsilon_j] &= \\delta_{ij}.
    
    If ``norm=True``, the variance is normalized to 1, which amounts to
    normalizing :math:`\\mathbf w` to unit length.
    
    """
    
    # TODO reference? must find some standard book with a treatment which is
    # not too formal yet writes clearly about the covariance function
    
    # TODO nd version with w.ndim == n, it's a nd convolution. use
    # jax.scipy.signal.correlate.
        
    w = jnp.asarray(w)
    assert w.ndim == 1
    if len(w):
        cov = jnp.convolve(w, w[::-1])
        if norm:
            cov /= cov[len(w) - 1]
        return cov.at[delta + len(w) - 1].get(mode='fill', fill_value=0)
    else:
        return jnp.zeros(delta.shape)

@stationarykernel(derivable=False, maxdim=1, input='abs')
def _ARBase(delta, phi=None, gamma=None, maxlag=None, slnr=None, lnc=None, norm=False):
    """
    Discrete autoregressive kernel.
        
    You have to specify one and only one of the sets of parameters
    ``phi+maxlag``, ``gamma+maxlag``, ``slnr+lnc``.

    Parameters
    ----------
    phi : (p,) real
        The autoregressive coefficients at lag 1...p.
    gamma : (p + 1,) real
        The autocovariance at lag 0...p.
    maxlag : int
        The maximum lag that the kernel will be evaluated on. If the actual
        inputs produce higher lags, the missing values are filled with ``nan``.
    slnr : (nr,) real
        The real roots of the characteristic polynomial, expressed in the
        following way: ``sign(slnr)`` is the sign of the root, and
        ``abs(slnr)`` is the natural logarithm of the absolute value.
    lnc : (nc,) complex
        The natural logarithm of the complex roots of the characteristic
        polynomial (:math:`\\log z = \\log|z| + i\\arg z`), where each root
        also stands for its paired conjugate.
    
        In `slnr` and `lnc`, the multiplicity of a root is expressed by
        repeating the root in the array (not necessarily next to each other).
        Only exact repetition counts; very close yet distinct roots are treated
        as separate and lead to numerical instability, in particular complex
        roots very close to the real line. An exactly real complex root behaves
        like a pair of identical real roots. Two complex roots also count as
        equal if conjugate, and the argument is standardized to :math:`[0,
        2\\pi)`.
    norm : bool, default False
        If True, normalize the autocovariance to be 1 at lag 0. If False,
        normalize such that the variance of the generating noise is 1, or use
        the user-provided normalization if `gamma` is specified.
    
    Notes
    -----
    This is the covariance function of a stationary autoregressive process,
    which is defined recursively as
    
    .. math::
        y_i = \\sum_{k=1}^p \\phi_k y_{i-k} + \\epsilon_i,
    
    where :math:`\\epsilon_i` is white noise, i.e.,
    :math:`\\operatorname{Cov}[\\epsilon_i, \\epsilon_j] = \\delta_{ij}`. The
    length :math:`p` of the vector of coefficients :math:`\\boldsymbol\\phi`
    is the "order" of the process.
    
    The covariance function can be expressed in two ways. First as the same
    recursion defining the process:
    
    .. math::
        \\gamma_m = \\sum_{k=1}^p \\phi_k \\gamma_{m-k} + \\delta_{m0},
    
    where :math:`\\gamma_m \\equiv \\operatorname{Cov}[y_i, y_{i+m}]`. This is
    called "Yule-Walker equation." Second, as a linear combination of mixed
    power-exponentials:
    
    .. math::
        \\gamma_m = \\sum_{j=1}^n
                    \\sum_{l=1}^{\\mu_j}
                    a_{jl} |m|^{l-1} x_j^{-|m|},
    
    where :math:`x_j` and :math:`\\mu_j` are the (complex) roots and
    corresponding multiplicities of the "characteristic polynomial"
    
    .. math::
        P(x) = 1 - \\sum_{k=1}^p \\phi_k x^k,
    
    and the :math:`a_{jl}` are uniquely determined complex coefficients. The
    :math:`\\boldsymbol\\phi` vector is valid iff :math:`|x_j|>1, \\forall j`.
    
    There are three alternative parametrization for this kernel.
    
    If you specify `phi`, the first terms of the covariance are computed
    solving the Yule-Walker equation, and then evolved up to `maxlag`. It
    is necessary to specify `maxlag` instead of letting the code figure it out
    from the actual inputs for technical reasons.
    
    Likewise, if you specify `gamma`, the coefficients are obtained with
    Yule-Walker and then used to evolve the covariance. The only difference is
    that the normalization can be different: starting from `phi`, the variance
    of the generating noise :math:`\\epsilon` is fixed to 1, while giving
    `gamma` directly implies an arbitrary value.
    
    Instead, if you specify the roots with `slnr` and `lnc`, the coefficients
    are obtained from the polynomial defined in terms of the roots, and then
    the amplitudes :math:`a_{jl}` are computed by solving a linear system with
    the covariance (from YW) as RHS. Finally, the full covariance function is
    evaluated with the analytical expression.
    
    The reasons for using the logarithm are that 1) in practice the roots are
    tipically close to 1, so the logarithm is numerically more accurate, and 2)
    the logarithm is readily interpretable as the inverse of the correlation
    length.
    
    """
    cond = (
        (phi is not None and maxlag is not None and gamma is None and slnr is None and lnc is None) or
        (phi is None and maxlag is not None and gamma is not None and slnr is None and lnc is None) or
        (phi is None and maxlag is None and gamma is None and slnr is not None and lnc is not None)
    )
    if not cond:
        raise ValueError('invalid set of specified parameters')
    
    # TODO maybe I could allow redundantly specifying gamma and phi, e.g., for
    # numerical accuracy reasons if they are determined from an analytical
    # expression.
    
    if phi is None and gamma is None:
        return _ar_with_roots(delta, slnr, lnc, norm)
    else:
        return _ar_with_phigamma(delta, phi, gamma, maxlag, norm)

def _ar_with_phigamma(delta, phi, gamma, maxlag, norm):
    if phi is None:
        phi = AR.phi_from_gamma(gamma)
    if gamma is None:
        gamma = AR.gamma_from_phi(phi)
    if norm:
        gamma = gamma / gamma[0]
    acf = AR.extend_gamma(gamma, phi, maxlag + 1 - len(gamma))
    return acf.at[delta].get(mode='fill', fill_value=jnp.nan)

def _yule_walker(gamma):
    """
    gamma = autocovariance at lag 0...p
    output: autoregressive coefficients at lag 1...p
    """
    gamma = jnp.asarray(gamma)
    assert gamma.ndim == 1
    t = gamma[:-1]
    b = gamma[1:]
    if t.size:
        return _toeplitz.solve(t, b)
    else:
        return jnp.empty(0)

def _yule_walker_inv_mat(phi):
    phi = jnp.asarray(phi)
    assert phi.ndim == 1
    p = len(phi)
    m = jnp.arange(p + 1)[:, None] # rows
    n = m.T # columns
    phi = jnp.pad(phi, (1, 1))
    kp = jnp.clip(m + n, 0, p + 1)
    km = jnp.clip(m - n, 0, p + 1)
    return jnp.eye(p + 1) - (phi[kp] + phi[km]) / jnp.where(n, 1, 2)
    
def _yule_walker_inv(phi):
    """
    phi = autoregressive coefficients at lag 1...p
    output: autocovariance at lag 0...p, assuming driving noise has sdev 1
    """
    a = _yule_walker_inv_mat(phi)
    b = jnp.zeros(len(a)).at[0].set(1)
    # gamma = _pseudo_solve(a, b)
    gamma = jnp.linalg.solve(a, b)
    return gamma

def _ar_evolve(phi, start, noise):
    """
    phi = autoregressive coefficients at lag 1...p
    start = first p values of the process (increasing time)
    noise = n noise values added at each step
    output: n new process values
    """
    phi = jnp.asarray(phi)
    start = jnp.asarray(start)
    noise = jnp.asarray(noise)
    assert phi.ndim == 1 and phi.shape == start.shape and noise.ndim == 1
    return _ar_evolve_jit(phi, start, noise)

@jax.jit
def _ar_evolve_jit(phi, start, noise):
    
    def f(carry, eps):
        vals, cc, roll = carry
        phi = lax.dynamic_slice(cc, [vals.size - roll], [vals.size])
        nextval = phi @ vals + eps
        if vals.size:
            vals = vals.at[roll].set(nextval)
        # maybe for some weird reason like alignment, actual rolling would
        # be faster. whatever
        roll = (roll + 1) % vals.size
        return (vals, cc, roll), nextval
    
    cc = jnp.concatenate([phi, phi])[::-1]
    _, ev = lax.scan(f, (start, cc, 0), noise, unroll=16)
    return ev

def _ar_with_roots(delta, slnr, lnc, norm):
    phi = AR.phi_from_roots(slnr, lnc) # <---- weak
    gamma = AR.gamma_from_phi(phi) # <---- point
    if norm:
        gamma /= gamma[0]
    ampl = AR.ampl_from_roots(slnr, lnc, gamma)
    acf = AR.cov_from_ampl(slnr, lnc, ampl, delta)
    return acf

    # TODO Currently gamma is not even pos def for high multiplicity/roots close
    # to 1. Raw patch: the badness condition is gamma[0] < 0 or any(abs(gamma) >
    # gamma[0]) or gamma inf/nan. Take the smallest log|root| and assume it
    # alone determines gamma. This is best implemented as an option in
    # _gamma_from_ampl_matmul.
    
    # Is numerical integration of the spectrum a feasible way to get the
    # covariance? The roots correspond to peaks, and they get very high as the
    # roots get close to 1. But I know where the peaks are in advance => nope
    # because the e^iwx oscillates arbitrarily fast. However maybe I can compute
    # the first p terms, which solves my current problem with gamma. I guess I
    # just have to use a multiple of p of quadrature points. The spectrum
    # oscillates too but only up to mode p. The total calculation cost is then
    # O(p^2), better than current O(p^3). See Hamilton (1994, p. 155).
    
    # Other solution (Hamilton p. 319): the covariance should be equal to the
    # impulse response, so I can get gamma from phi by an evolution starting
    # from zeros. => Nope, it's equal only for AR(1).
    
    # condition for phi: in the region phi >= 0, it must be sum(phi) <= 1
    # (Hamilton p. 659).
    
    # p = phi.size
    # yw = _yule_walker_inv_mat(phi)
    # b = jnp.zeros(p + 1).at[0].set(1)
    # ampl = jnp.linalg.solve(yw @ mat, b)
    # lag = delta if delta.ndim else delta[None]
    # acf = _gamma_from_ampl_matmul(slnr, lnc, lag, ampl)
    # if norm:
    #     acf0 = _gamma_from_ampl_matmul(slnr, lnc, jnp.array([0]), ampl)
    #     acf /= acf0
    # return acf if delta.ndim else acf.squeeze(0)

def _pseudo_solve(a, b):
    # this is less accurate than jnp.linalg.solve
    u, s, vh = jnp.linalg.svd(a)
    eps = jnp.finfo(a.dtype).eps
    s0 = s[0] if s.size else 0
    invs = jnp.where(s < s0 * eps * len(a), 0, 1 / s)
    return jnp.einsum('ij,j,jk,k', vh.conj().T, invs, u.conj().T, b)

@jax.jit
def _gamma_from_ampl_matmul(slnr, lnc, lag, ampl, lagnorm=None):

    vec = ampl.ndim == 1
    if vec:
        ampl = ampl[:, None]
    p = slnr.size + 2 * lnc.size
    assert ampl.shape[-2] == p + 1
    if lagnorm is None:
        lagnorm = p
    
    def logcol(root, lag, llag, repeat):
        return -root * lag + jnp.where(repeat, repeat * llag, 0)
    
    def lognorm(root, repeat, lagnorm):
        maxnorm = jnp.where(repeat, repeat * (-1 + jnp.log(repeat / root)), 0)
        defnorm = logcol(root, lagnorm, jnp.log(lagnorm), repeat)
        maxloc = repeat / root
        return jnp.where(maxloc <= lagnorm, maxnorm, defnorm)
    
    # roots at infinity
    # TODO remove this because it's degenerate with large roots, handle the
    # p=0 case outside of this function
    col = jnp.where(lag, 0, 1)
    out = col[..., :, None] * ampl[..., 0, :]
    
    # real roots
    llag = jnp.log(lag)
    val = (jnp.nan, 0, out, slnr, lag, llag, lagnorm)
    def loop(i, val):
        prevroot, repeat, out, slnr, lag, llag, lagnorm = val
        root = slnr[i]
        repeat = jnp.where(root == prevroot, repeat + 1, 0)
        prevroot = root
        sign = jnp.sign(root) ** lag
        aroot = jnp.abs(root)
        lcol = logcol(aroot, lag, llag, repeat)
        norm = lognorm(aroot, repeat, lagnorm)
        col = sign * jnp.exp(lcol - norm)
        out += col[..., :, None] * ampl[..., 1 + i, :]
        return prevroot, repeat, out, slnr, lag, llag, lagnorm
    if slnr.size:
        _, _, out, _, _, _, _ = lax.fori_loop(0, slnr.size, loop, val)
    
    # complex roots
    val = (jnp.nan, 0, out, lnc, lag, llag, lagnorm)
    def loop(i, val):
        prevroot, repeat, out, lnc, lag, llag, lagnorm = val
        root = lnc[i]
        repeat = jnp.where(root == prevroot, repeat + 1, 0)
        prevroot = root
        lcol = logcol(root, lag, llag, repeat)
        norm = lognorm(root.real, repeat, lagnorm)
        col = jnp.exp(lcol - norm)
        idx = 1 + slnr.size + 2 * i
        out += col.real[..., :, None] * ampl[..., idx, :]
        
        # real complex root = a pair of identical real roots
        repeat = jnp.where(root.imag, repeat, repeat + 1)
        col1 = jnp.where(root.imag, -col.imag, col.real * lag)
        out += col1[..., :, None] * ampl[..., idx + 1, :]
        
        return prevroot, repeat, out, lnc, lag, llag, lagnorm
    if lnc.size:
        _, _, out, _, _, _, _ = lax.fori_loop(0, lnc.size, loop, val)
    
    if vec:
        out = out.squeeze(-1)
    
    return out

class AR(_ARBase):
    
    __doc__ = _ARBase.__doc__
    
    @classmethod
    def phi_from_gamma(cls, gamma):
        """
        Determine the autoregressive coefficients from the covariance.
        
        Parameters
        ----------
        gamma : (p + 1,) array
            The autocovariance at lag 0...p.
        
        Returns
        -------
        phi : (p,) array
            The autoregressive coefficients at lag 1...p.
        """
        gamma = cls._process_gamma(gamma)
        return _yule_walker(gamma)
    
    @classmethod
    def gamma_from_phi(cls, phi):
        """
        Determine the covariance from the autoregressive coefficients.

        Parameters
        ----------
        phi : (p,) array
            The autoregressive coefficients at lag 1...p.
        
        Returns
        -------
        gamma : (p + 1,) array
            The autocovariance at lag 0...p. The normalization is
            with noise variance 1.
        
        Notes
        -----
        The result is wildly inaccurate for roots with high multiplicity and/or
        close to 1.
        """
        phi = cls._process_phi(phi)
        return _yule_walker_inv(phi)
        
        # TODO fails (nan) for very small roots. In that case the answer is that
        # gamma is a constant vector. However I can't get the constant out of
        # a degenerate phi, I need the roots, and I don't know the formula.
    
    @classmethod
    def extend_gamma(cls, gamma, phi, n):
        """
        Extends values of the covariance function to higher lags.
        
        Parameters
        ----------
        gamma : (m,) array
            The autocovariance at lag q-m+1...q, with q >= 0 and m >= p + 1.
        phi : (p,) array
            The autoregressive coefficients at lag 1...p.
        n : int
            The number of new values to generate.
        
        Returns
        -------
        ext : (m + n,) array
            The autocovariance at lag q-m+1...q+n.
        """
        gamma = cls._process_gamma(gamma)
        phi = cls._process_phi(phi)
        assert gamma.size > phi.size
        ext = _ar_evolve(phi, gamma[len(gamma) - len(phi):], jnp.broadcast_to(0., (n,)))
        return jnp.concatenate([gamma, ext])
    
    @classmethod
    def phi_from_roots(cls, slnr, lnc):
        """
        Determine the autoregressive coefficients from the roots of the
        characteristic polynomial.
        
        Parameters
        ----------
        slnr : (nr,) real
            The real roots of the characteristic polynomial, expressed in the
            following way: ``sign(slnr)`` is the sign of the root, and
            ``abs(slnr)`` is the natural logarithm of the absolute value.
        lnc : (nc,) complex
            The natural logarithm of the complex roots of the characteristic
            polynomial (:math:`\\log z = \\log|z| + i\\arg z`), where each root
            also stands for its paired conjugate.
        
        Returns
        -------
        phi : (p,) real
            The autoregressive coefficients at lag 1...p, with p = nr + 2 nc.
        """
        slnr, lnc = cls._process_roots(slnr, lnc)
        r = jnp.copysign(jnp.exp(-jnp.abs(slnr)), slnr) # works with +/-0
        c = jnp.exp(-lnc)
        
        # minus sign in the exponentials to do 1/z, the poly output is already
        # reversed
                
        roots = jnp.concatenate([r, c, c.conj()]).sort() # <-- polyroots sorts
        coef = jnp.atleast_1d(jnp.poly(roots))
        
        # TODO the implementation of jnp.poly (and np.poly) is inferior to the
        # one of np.polynomial.polynomial.polyfromroots, which cares about
        # numerical accuracy and would reduce compilation time if ported to jax
        # (current one is O(p), that would be O(log p)).
        
        if coef.size:
            with _jaxext.skipifabstract():
                numpy.testing.assert_equal(coef[0].item(), 1)
                numpy.testing.assert_allclose(jnp.imag(coef), 0, rtol=0, atol=1e-4)
        return -coef.real[1:]
        
        # TODO possibly not accurate for large p. Do a test with an
        # implementation of poly which uses integer roots and non-fft convolve
        # (maybe add it as an option to my to-be-written implementation of poly)

    @classmethod
    def ampl_from_roots(cls, slnr, lnc, gamma):
        # TODO docs
        slnr, lnc = cls._process_roots(slnr, lnc)
        gamma = cls._process_gamma(gamma)
        assert gamma.size == 1 + slnr.size + 2 * lnc.size
        lag = jnp.arange(gamma.size)
        mat = _gamma_from_ampl_matmul(slnr, lnc, lag, jnp.eye(gamma.size))
        # return jnp.linalg.solve(mat, gamma)
        return _pseudo_solve(mat, gamma)
        
        # TODO I'm using pseudo-solve only because of large roots degeneracy
        # in _gamma_from_ampl_matmul, remove it after solving that
        
        # TODO maybe I can increase the precision of the solve with some
        # ordering of the columns of mat, I guess (reversed) global sort of the
        # roots
    
    @classmethod
    def cov_from_ampl(cls, slnr, lnc, ampl, lag):
        # TODO docs
        slnr, lnc = cls._process_roots(slnr, lnc)
        ampl = cls._process_ampl(ampl)
        assert ampl.size == 1 + slnr.size + 2 * lnc.size
        lag = cls._process_lag(lag)
        scalar = lag.ndim == 0
        if scalar:
            lag = lag[None]
        acf = _gamma_from_ampl_matmul(slnr, lnc, lag, ampl)
        return acf.squeeze(0) if scalar else acf
        
    @classmethod
    def inverse_roots_from_phi(cls, phi):
        phi = cls._process_phi(phi)
        poly = jnp.concatenate([jnp.ones(1), -phi])
        return jnp.roots(poly, strip_zeros=False)
    
    # TODO methods:
    # - gamma_from_roots which uses quadrature fourier transf of spectrum
    
    @staticmethod
    def _process_roots(slnr, lnc):
        slnr = jnp.asarray(slnr, float).sort()
        lnc = jnp.asarray(lnc, complex)
        assert slnr.ndim == lnc.ndim == 1
        imag = jnp.abs(lnc.imag) % (2 * jnp.pi)
        imag = jnp.where(imag > jnp.pi, 2 * jnp.pi - imag, imag)
        lnc = lnc.real + 1j * imag
        lnc = lnc.sort()
        return slnr, lnc
    
    @staticmethod
    def _process_gamma(gamma):
        gamma = jnp.asarray(gamma, float)
        assert gamma.ndim == 1 and gamma.size >= 1
        return gamma
    
    @staticmethod
    def _process_phi(phi):
        phi = jnp.asarray(phi, float)
        assert phi.ndim == 1
        return phi
    
    @staticmethod
    def _process_ampl(ampl):
        ampl = jnp.asarray(ampl, float)
        assert ampl.ndim == 1 and ampl.size >= 1
        return ampl
    
    @staticmethod
    def _process_lag(lag):
        lag = jnp.asarray(lag)
        assert jnp.issubdtype(lag, jnp.integer)
        return lag.astype(int)
