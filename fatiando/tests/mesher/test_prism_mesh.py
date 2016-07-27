from ...mesher import PrismMesh


def test_prism_mesh_copy():
    p1 = PrismMesh((0, 1, 0, 2, 0, 3), (1, 2, 2))
    p1.addprop('density', 3200)
    p2 = p1.copy()
    assert p1 is not p2
    assert p1.props['density'] == p2.props['density']
    p2.props['density'] = 2000
    assert p1.props['density'] != p2.props['density']
