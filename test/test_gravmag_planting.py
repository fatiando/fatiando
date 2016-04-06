from __future__ import division
import numpy as np
from numpy.testing import assert_allclose
from fatiando.gravmag.planting import (PlantingGravity, PlantingMagnetic,
                                       _Neighbor)
from fatiando.gravmag import prism
from fatiando.mesher import PrismMesh
from fatiando import utils, gridder


def test_gravity_inversion():
    "gravmag.planting gravity inversion recovers correct model"
    # Make synthetic data
    bounds = [0, 10000, 0, 10000, 0, 5000]
    model = PrismMesh(bounds, (5, 10, 10))
    dens = np.zeros(model.shape)
    dens[1:-1, 4:6, 4:6] = 500
    model.addprop('density', dens)
    area = bounds[0:4]
    shape = (25, 25)
    x, y, z = gridder.regular(area, shape, z=-1)
    data = prism.gz(x, y, z, model)
    # Setup the inversion by creating a mesh and seeds
    mesh = model.copy()
    seeds = [[5000, 5000, 2000, {'density':500}]]
    solver = PlantingGravity(x, y, z, data, mesh).config(
        seeds=seeds, compactness=1, tol=0.001)
    # Run the inversion
    solver.fit()
    # Check if the estimated
    assert_allclose(solver.p_, dens.ravel())

def test_neighbor_comparison_to_int():
    "gravmag.planting Neighbor class can be compared (==) to ints"
    n = _Neighbor(index=42, prop=1244)
    assert n == 42, "Failed comparison with integer"


def test_neighbor_set():
    "gravmag.planting Neighbor classes can be used in a set"
    set1 = set([_Neighbor(index=i, prop=i) for i in range(5)])
    set2 = set([_Neighbor(index=i, prop=i) for i in range(3, 8)])
    union = set1.union(set2)
    diff = set1.difference(set2)
    assert union == {0, 1, 2, 3, 4, 5, 6, 7}, 'Wrong union'
    assert diff == {0, 1, 2}, 'Wrong difference'


def test_neighbor_set_difference_with_ints():
    "gravmag.planting Neighbor can be used in set and compared to set of ints"
    set1 = set([_Neighbor(index=i, prop=i) for i in range(5)])
    set2 = set(range(3, 8))
    union = set1.union(set2)
    diff = set1.difference(set2)
    assert union == {0, 1, 2, 3, 4, 5, 6, 7}, 'Wrong union'
    assert diff == {0, 1, 2}, 'Wrong difference'


