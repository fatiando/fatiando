from __future__ import absolute_import
from fatiando.mesher import PrismMesh, Prism

import numpy as np


def test_prism_mesh_copy():
    p1 = PrismMesh((0, 1, 0, 2, 0, 3), (1, 2, 2))
    p1.addprop('density', 3200 + np.zeros(p1.size))
    p2 = p1.copy()
    assert p1 is not p2
    assert np.array_equal(p1.props['density'], p2.props['density'])


def test_carvetopo():
    bounds = (0, 1, 0, 1, 0, 2)
    shape = (2, 1, 1)
    topox = [0, 0, 1, 1]
    topoy = [0, 1, 0, 1]
    topoz = [-1, -1, -1, -1]
    # Create reference prism meshs
    p0r = []
    p0r.append(None)
    p0r.append(Prism(0, 1, 0, 1, 1, 2))
    p2r = []
    p2r.append(Prism(0, 1, 0, 1, 0, 1))
    p2r.append(None)
    # Create test prism meshes
    p0 = PrismMesh(bounds, shape)
    p0.carvetopo(topox, topoy, topoz)
    p1 = PrismMesh(bounds, shape)
    p1.carvetopo(topox, topoy, topoz, below=False)
    p2 = PrismMesh(bounds, shape)
    p2.carvetopo(topox, topoy, topoz, below=True)
    # Test p0 and p1 which should be the same
    for pi in [p0, p1]:
        for i, p in enumerate(pi):
            if i == 0:
                assert p is None
            else:
                assert p is not None
                assert np.any(p0r[i].center() == p.center())
    # Test p2
    for i, p in enumerate(p2):
        if i == 1:
            assert p is None
        else:
            assert p is not None
            assert np.any(p2r[i].center() == p.center())
