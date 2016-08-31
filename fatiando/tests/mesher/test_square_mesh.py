from ...mesher import SquareMesh

import numpy as np


def test_square_mesh_copy():
    mesh = SquareMesh((0, 4, 0, 6), (2, 2))
    mesh.addprop('slowness', 234 + np.zeros(mesh.size))
    cp = mesh.copy()
    assert np.array_equal(mesh.props['slowness'], cp.props['slowness'])
    assert mesh is not cp
    assert mesh.bounds == cp.bounds
    assert mesh.dims == cp.dims
    assert np.array_equal(mesh.get_xs(), cp.get_xs())
    assert np.array_equal(mesh.get_ys(), cp.get_ys())
    cp.addprop('density', 3000 + np.zeros(cp.size))
    assert mesh.props != cp.props
