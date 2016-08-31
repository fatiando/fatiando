from ...mesher import PrismMesh

import numpy as np


def test_prism_mesh_copy():
    p1 = PrismMesh((0, 1, 0, 2, 0, 3), (1, 2, 2))
    p1.addprop('density', 3200 + np.zeros(p1.size))
    p2 = p1.copy()
    assert p1 is not p2
    assert np.array_equal(p1.props['density'], p2.props['density'])
