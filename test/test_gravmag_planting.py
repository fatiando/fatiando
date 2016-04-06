from __future__ import division
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import cPickle as pickle
from fatiando.gravmag.planting import (PlantingGravity, PlantingMagnetic,
                                       _Cell)
from fatiando.gravmag import prism
from fatiando.mesher import PrismMesh
from fatiando import utils, gridder



def make_model_n_data():
    bounds = [0, 10000, 0, 10000, 0, 5000]
    model = PrismMesh(bounds, (5, 10, 10))
    dens = np.zeros(model.shape)
    dens[1:-1, 4:6, 4:6] = 500
    model.addprop('density', dens.ravel())
    area = bounds[0:4]
    shape = (25, 25)
    x, y, z = gridder.regular(area, shape, z=-1)
    data = prism.gz(x, y, z, model)
    return x, y, z, data, model


def test_gravity_inversion():
    "gravmag.planting gravity inversion recovers correct model"
    x, y, z, data, model = make_model_n_data()
    # Setup the inversion by creating a mesh and seeds
    seeds = [[5500, 5500, 2500, {'density': 500}]]
    solver = PlantingGravity(x, y, z, data, model).config(
        seeds=seeds, compactness=1, tol=0.001)
    # Run the inversion
    solver.fit()
    # Check if the estimated
    print solver.p_.reshape(model.shape)
    assert_allclose(solver.p_, model.props['density'])


def test_planting_init():
    "gravmag.planting inversion initialization is working properly"
    x, y, z, data, model = make_model_n_data()
    # Setup the inversion by creating a mesh and seeds
    seeds = [[5500, 5500, 2500, {'density': 500}]]
    solver = PlantingGravity(x, y, z, data, model).config(
        seeds=seeds, compactness=1, tol=0.001)
    print solver.seeds
    p, neighbors, misfit, compactness, goal = solver._init_planting()
    assert False


def test_classes_can_be_pickled():
    "gravmag.planting classes can be pickled."
    x, y, z, data, model = make_model_n_data()
    # Setup the inversion by creating a mesh and seeds
    solver = PlantingGravity(x, y, z, data, model)
    solver = pickle.loads(pickle.dumps(solver))#.fit()
    # Check if the estimated
    # assert_allclose(solver.p_, model.props['density'])


def test_shape_of_anomaly():
    "gravmag.planting shape of anomaly function gives expected results"
    x, y, z, data, model = make_model_n_data()
    # Setup the inversion by creating a mesh and seeds
    seeds = [[5000, 5000, 2000, {'density':500}]]
    solver = PlantingGravity(x, y, z, data, model)
    solver.predicted = lambda p: data*0.00001
    assert_almost_equal(solver.shape_of_anomaly(None), 0)
    solver.predicted = lambda p: x
    assert solver.shape_of_anomaly(None) > 0


def test_neighbor_comparison_to_int():
    "gravmag.planting cell class can be compared (==) to ints"
    n = _Cell(42, None, None)
    assert n == 42, "Failed comparison with integer"


def test_neighbor_set():
    "gravmag.planting cell classes can be used in a set"
    set1 = set([_Cell(i, None, None) for i in range(5)])
    set2 = set([_Cell(i, None, None) for i in range(3, 8)])
    union = set1.union(set2)
    diff = set1.difference(set2)
    assert union == {0, 1, 2, 3, 4, 5, 6, 7}, 'Wrong union'
    assert diff == {0, 1, 2}, 'Wrong difference'


def test_neighbor_set_difference_with_ints():
    "gravmag.planting cell can be used in set and compared to set of ints"
    set1 = set([_Cell(i, None, None) for i in range(5)])
    set2 = set(range(3, 8))
    union = set1.union(set2)
    diff = set1.difference(set2)
    assert union == {0, 1, 2, 3, 4, 5, 6, 7}, 'Wrong union'
    assert diff == {0, 1, 2}, 'Wrong difference'


def test_finding_neighbors():
    "gravmag.planting cell class can find its neighbors in the mesh"
    mesh = PrismMesh([0, 10, 0, 10, 0, 10], (3, 4, 5))
    neighbors = [
        # Top corners
        [0,  {1, 5, 20}],
        [4,  {3, 9, 24}],
        [15, {10, 16, 35}],
        [19, {18, 14, 39}],
        # Bottom corners
        [40, {45, 41, 20}],
        [44, {43, 49, 24}],
        [55, {50, 56, 35}],
        [59, {58, 54, 39}],
        # In the middle
        [32, {31, 37, 33, 27, 52, 12}],
        [26, {25, 31, 27, 21, 6, 46}],
    ]
    for i, true_neighbors in neighbors:
        n = _Cell(i, 200, mesh)
        msg = 'Failed neighbor {}: true = {} calculated = {}'.format(
            i, true_neighbors, n.neighbors)
        assert n.neighbors == true_neighbors, msg
