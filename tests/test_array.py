import sys

sys.path.insert(0, '.')
import lsqfitgp as lgp

import numpy as np

def test_ellipsis():
    x = np.empty((2, 3), dtype=[('a', float, 4)])
    y = lgp.StructuredArray(x)
    z = y[..., 0]
    assert z.shape == y.shape[:1]
    assert z['a'].shape == y.shape[:1] + x.dtype.fields['a'][0].shape
