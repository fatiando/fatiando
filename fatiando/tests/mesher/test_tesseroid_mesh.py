from ...mesher import TesseroidMesh

import numpy as np


def test_tesseroid_mesh_copy():
    orig = TesseroidMesh((0, 1, 0, 2, 3, 0), (1, 2, 2))
    cp = orig.copy()
    assert cp is not orig
    assert orig.celltype == cp.celltype
    assert orig.bounds == cp.bounds
    assert orig.dump == cp.dump
    orig.addprop('density', 3300 + np.zeros(orig.size))
    cp = orig.copy()
    assert np.array_equal(orig.props['density'], cp.props['density'])
