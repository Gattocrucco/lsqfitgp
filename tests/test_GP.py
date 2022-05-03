# lsqfitgp/tests/test_GP.py
#
# Copyright (c) 2020, 2022, Giacomo Petrillo
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

import sys

import numpy as np
import gvar

sys.path.insert(0, '.')
import lsqfitgp as lgp

def test_prior_raw_shape():
    gp = lgp.GP(lgp.ExpQuad())
    gp.addx(np.arange(20).reshape(2, 10), 'x')
    cov = gp.prior(raw=True)
    assert cov['x', 'x'].shape == (2, 10, 2, 10)
    
    cov = gp.prior('x', raw=True)
    assert cov.shape == (2, 10, 2, 10)

def test_no_checksym():
    covs = []
    for checksym in [False, True]:
        gp = lgp.GP(lgp.ExpQuad(), checksym=checksym)
        gp.addx({'x': np.arange(20)})
        covs.append(gp.prior('x', raw=True))
    np.testing.assert_equal(*covs)

def test_transf_scalar():
    gp = lgp.GP(lgp.ExpQuad() + lgp.RatQuad())
    x = np.arange(20)
    gp.addx(x, 'x')
    cov1 = gp.prior('x', raw=True)
    
    gp = lgp.GP(lgp.where(lambda f: f, lgp.ExpQuad(dim='f1'), lgp.RatQuad(dim='f1'), dim='f0'))
    y = np.empty((2, len(x)), '?,f8')
    y['f0'] = np.reshape([0, 1], (2, 1))
    y['f1'] = x
    gp.addx(y[0], 'y0')
    gp.addx(y[1], 'y1')
    gp.addtransf({'y0': 1, 'y1': 1}, 'x')
    cov2 = gp.prior('x', raw=True)
    
    np.testing.assert_equal(cov1, cov2)

def test_transf_vector():
    gp = lgp.GP(lgp.ExpQuad())
    gp.addx([0, 1], 'x')
    gp.addtransf({'x': [-1, 1]}, 'y')
    prior = gp.prior()
    y1 = prior['y']
    y2 = prior['x'][1] - prior['x'][0]
    np.testing.assert_allclose(gvar.mean(y1), gvar.mean(y2))
    np.testing.assert_allclose(gvar.sdev(y1), gvar.sdev(y2))
    z = y1 - y2
    np.testing.assert_allclose(gvar.mean(z), 0)
    np.testing.assert_allclose(gvar.sdev(z), 0)

def test_cov():
    A = np.random.randn(20, 20)
    M1 = A.T @ A
    gp = lgp.GP()
    gp.addcov(M1, 'M')
    M2 = gp.prior('M', raw=True)
    np.testing.assert_equal(M1, M2)

def test_proctransf():
    x = np.arange(20)
    gp = lgp.GP()
    gp.addproc(lgp.ExpQuad(), 'a')
    gp.addproc(lgp.RatQuad(), 'b')
    gp.addx(x, 'ax', proc='a')
    gp.addx(x, 'bx', proc='b')
    gp.addtransf({'ax': 2, 'bx': 3}, 'abx1')
    gp.addproctransf({'a': 2, 'b': 3}, 'ab')
    gp.addx(x, 'abx2', proc='ab')
    prior = gp.prior(['abx1', 'abx2'], raw=True)
    np.testing.assert_equal(prior['abx1', 'abx1'], prior['abx2', 'abx2'])
    np.testing.assert_equal(prior['abx1', 'abx2'], prior['abx2', 'abx1'].T)
    np.testing.assert_equal(prior['abx1', 'abx2'], prior['abx1', 'abx1'])

def test_kernelop():
    gp = lgp.GP()
    gp.addproc(lgp.ExpQuad(), 'a')
    f = lambda x: x
    gp.addproc(lgp.ExpQuad() * lgp.Rescaling(stdfun=f), 'b1')
    gp.addkernelop('rescale', f, 'b2', 'a')
    x = np.arange(20)
    gp.addx(x, 'x1', proc='b1')
    gp.addx(x, 'x2', proc='b2')
    prior = gp.prior(['x1', 'x2'], raw=True)
    np.testing.assert_equal(prior['x1', 'x1'], prior['x2', 'x2'])
    np.testing.assert_equal(prior['x2', 'x1'], prior['x1', 'x2'].T)
    np.testing.assert_equal(prior['x1', 'x2'], np.zeros(2 * x.shape))
