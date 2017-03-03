from ...mesher import TriaxialEllipsoid, ProlateEllipsoid, OblateEllipsoid
import numpy as np


def test_triaxial_ellipsoid_copy():
    'Check the elements of a duplicated ellipsoid'
    orig = TriaxialEllipsoid(12, 42, 53, 61, 35, 14, 10, 20, 30, {
                             'remanent magnetization': [10, 25, 40],
                             'susceptibility tensor': [0.562, 0.485, 0.25,
                                                       90, 0, 0]})
    cp = orig.copy()
    assert orig is not cp
    assert orig.x == cp.x
    assert orig.y == cp.y
    assert orig.z == cp.z
    assert orig.large_axis == cp.large_axis
    assert orig.intermediate_axis == cp.intermediate_axis
    assert orig.small_axis == cp.small_axis
    assert orig.props == cp.props

    cp.x = 4
    cp.y = 6
    cp.z = 7
    cp.props['susceptibility tensor'] = [0.7, 0.9, 10, 9, 28, -10]
    assert orig.x != cp.x
    assert orig.y != cp.y
    assert orig.z != cp.z
    assert orig.props['susceptibility tensor'] != \
        cp.props['susceptibility tensor']


def test_prolate_ellipsoid_copy():
    'Check the elements of a duplicated ellipsoid'
    orig = ProlateEllipsoid(31, 2, 83, 56, 54, 1, 29, 70, {
                            'remanent magnetization': [8, 25, 40],
                            'susceptibility tensor': [0.562, 0.485, 0.25,
                                                      0, 87, -10]})
    cp = orig.copy()
    assert orig is not cp
    assert orig.x == cp.x
    assert orig.y == cp.y
    assert orig.z == cp.z
    assert orig.large_axis == cp.large_axis
    assert orig.small_axis == cp.small_axis
    assert orig.props == cp.props

    cp.x = 4
    cp.y = 6
    cp.z = 7
    cp.props['susceptibility tensor'] = [0.7, 0.9, 10, 90, -10]
    assert orig.x != cp.x
    assert orig.y != cp.y
    assert orig.z != cp.z
    assert orig.props['susceptibility tensor'] != \
        cp.props['susceptibility tensor']


def test_oblate_ellipsoid_copy():
    'Check the elements of a duplicated ellipsoid'
    orig = OblateEllipsoid(1, 2, 3, 4, 6, 10, 20, 30, {
        'remanent magnetization': [3, -2, 40],
        'susceptibility tensor': [0.562, 0.485, 0.25,
                                  90, 34, 0]})
    cp = orig.copy()
    assert orig is not cp
    assert orig.x == cp.x
    assert orig.y == cp.y
    assert orig.z == cp.z
    assert orig.small_axis == cp.small_axis
    assert orig.large_axis == cp.large_axis
    assert orig.props == cp.props

    cp.x = 4
    cp.y = 6
    cp.z = 7
    cp.props['remanent magnetization'] = [100, -40, -25]
    assert orig.x != cp.x
    assert orig.y != cp.y
    assert orig.z != cp.z
    assert orig.props['remanent magnetization'] != \
        cp.props['remanent magnetization']
