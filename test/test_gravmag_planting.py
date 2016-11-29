from __future__ import division, print_function, unicode_literals
import cPickle as pickle
import copy
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from nose.tools import assert_raises
from fatiando.gravmag.planting import (PlantingGravity, PlantingMagnetic,
                                       _Cell)
from fatiando.gravmag import prism
from fatiando.mesher import PrismMesh
from fatiando import utils, gridder



def make_model_n_data():
    "Make synthetic gravity data from a simple model"
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


def make_2prism_model_n_data():
    "Make synthetic data from a 2 prism model"
    bounds = [0, 4000, 0, 3000, 0, 3000]
    model = PrismMesh(bounds, (3, 3, 4))
    dens = np.zeros(model.shape)
    dens[1, 1, 1:3] = 500
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
    # Check if the estimated densities match the model
    assert_allclose(solver.p_, model.props['density'])
    assert_almost_equal(solver.value(solver.p_), 0)
    assert_almost_equal(solver.shape_of_anomaly(solver.p_), 0)


def test_gravity_inversion_2seeds():
    "gravmag.planting gravity inversion recovers correct model using 2 seeds"
    x, y, z, data, model = make_model_n_data()
    # Setup the inversion by creating a mesh and seeds
    seeds = [[5500, 5500, 2500, {'density': 500}],
             [5500, 4500, 2500, {'density': 500}]]
    solver = PlantingGravity(x, y, z, data, model).config(
        seeds=seeds, compactness=100, tol=0.01)
    # Run the inversion
    solver.fit()
    # Check if the estimated densities match the model
    print(solver.p_.reshape(model.shape))
    assert_allclose(solver.p_, model.props['density'])
    assert_almost_equal(solver.value(solver.p_), 0)
    assert_almost_equal(solver.shape_of_anomaly(solver.p_), 0)


def test_planting_init():
    "gravmag.planting inversion initialization is working properly"
    x, y, z, data, model = make_model_n_data()
    # Setup the inversion by creating a mesh and seeds
    seeds = [[5500, 5500, 2500, {'density': 500}]]
    solver = PlantingGravity(x, y, z, data, model).config(
        seeds=seeds, compactness=1, tol=0.001)
    assert len(solver.seeds) == 1, "Too many seeds"
    assert solver.seeds[0] == 255, 'Wrong seed index'
    p, neighbors, misfit, compactness, goal = solver._init_planting()
    assert misfit > 0, 'Misfit is positive'
    assert compactness == 0, 'Compactness is zero'
    nonzero = np.nonzero(p)[0]
    assert len(nonzero) == 1, 'More than one element in initialization.'
    assert nonzero[0] == solver.seeds[0].index, "Seed prop placed in wrong i."
    assert p[nonzero[0]] == solver.seeds[0].prop, "Wrong seed prop."
    assert len(neighbors) == 1, 'Too many neighbor groups'
    assert len(neighbors[0]) == 6, 'Too many neighbors'
    for n in neighbors[0]:
        assert n.prop == 500, 'Wrong density passed to neighbors'
    # The seed is element 255. Check if the right neighbors were added.
    assert neighbors[0] == {256, 254, 265, 245, 155, 355}, 'Wrong neighbors'
    assert len(solver.effects) == 1, 'Calculated too many effects'
    assert solver.effects.keys() == solver.seeds, 'Only the effect of the seed'


def test_grow():
    "gravmag.planting grows a single seed correctly."
    x, y, z, data, model = make_2prism_model_n_data()
    # Setup the inversion by creating a mesh and seeds
    seeds = [[1500, 1500, 1500, {'density': 500}]]
    solver = PlantingGravity(x, y, z, data, model).config(
        seeds=seeds, compactness=1, tol=0.001)
    assert len(solver.seeds) == 1, "Too many seeds"
    assert solver.seeds[0] == 17, 'Wrong seed index'
    p, neighbors, misfit, compactness, goal = solver._init_planting()
    best = solver._grow(0, neighbors, p, misfit, compactness, goal)
    assert len(np.nonzero(p)[0]) == 1, 'grow added element to p'
    assert best['neighbor'] == 18, 'Add wrong neighbor'
    p[best['neighbor'].index] = best['neighbor'].prop
    assert_allclose(p, model.props['density'])


def test_update_neighbors():
    "gravmag.planting updates the neighbor list correctly"
    x, y, z, data, model = make_2prism_model_n_data()
    # Setup the inversion by creating a mesh and seeds
    seeds = [[1500, 1500, 1500, {'density': 500}]]
    solver = PlantingGravity(x, y, z, data, model).config(
        seeds=seeds, compactness=1, tol=0.001)
    p, neighbors, misfit, compactness, goal = solver._init_planting()
    original = copy.copy(neighbors[0])
    for n in neighbors[0]:
        if n == 18:
            break
    solver._update_neighbors(n, 0, neighbors, p)
    assert len(neighbors[0]) == 10, 'Added wrong number of neighbors'
    new = neighbors[0].difference(original)
    assert len(new) == 5, 'Wrong number of new neighbors'
    assert new == {19, 22, 14, 6, 30}, 'Wrong new neighbors'


def test_classes_can_be_pickled():
    "gravmag.planting classes can be pickled."
    x, y, z, data, model = make_model_n_data()
    # Setup the inversion by creating a mesh and seeds
    seeds = [[5500, 5500, 2500, {'density': 500}]]
    solver = PlantingGravity(x, y, z, data, model).config(
        seeds=seeds, compactness=1, tol=0.001)
    # Pickle back and forth
    solver = pickle.loads(pickle.dumps(solver)).fit()
    # Check if the estimated
    assert_allclose(solver.p_, model.props['density'])


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
    assert n == 42, "Failed equality with integer"
    assert n != 41, 'Failed difference with integer'
    with assert_raises(ValueError):
        n == 3.14
    with assert_raises(ValueError):
        n == '42'

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
