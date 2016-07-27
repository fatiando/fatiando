from ...mesher import SquareMesh


def test_square_mesh_copy():
    mesh = SquareMesh((0, 4, 0, 6), (2, 2))
    mesh.addprop('slowness', [3.4, 8.6])
    cp = mesh.copy()
    assert mesh.props == cp.props
    assert mesh is not cp
    assert mesh.bounds == cp.bounds
    assert mesh.dims == cp.dims
    assert mesh.get_xs().all() == cp.get_xs().all()
    assert mesh.get_ys().all() == cp.get_ys().all()
    cp.addprop('density', 3000)
    assert mesh.props != cp.props
