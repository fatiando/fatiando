import numpy
from fatiando import utils, gridder

def test_gridder_scatter():
    "gridder.scatters returns diff sequence"
    area = [-1000, 1200, -40, 200]
    size = 1300
    for i in xrange(20):
        x1, y1 = gridder.scatter(area, size)
        x2, y2 = gridder.scatter(area, size)
        assert numpy.all(x1 != x2) and numpy.all(y1 != y2)

def test_gridder_scatter_seed():
    "gridder.scatters returns same sequence using same random seed"
    area = [0, 1000, 0, 1000]
    size = 1000
    for seed in numpy.random.randint(low=0, high=10000, size=20):
        x1, y1 = gridder.scatter(area, size, seed=seed)
        x2, y2 = gridder.scatter(area, size, seed=seed)
        assert numpy.all(x1 == x2) and numpy.all(y1 == y2)

def test_gridder_scatter_seed_noseed():
    "gridder.scatters returns diff sequence after using random seed"
    area = [0, 1000, 0, 1000]
    size = 1000
    seed = 1242
    x1, y1 = gridder.scatter(area, size, seed=seed)
    x2, y2 = gridder.scatter(area, size, seed=seed)
    assert numpy.all(x1 == x2) and numpy.all(y1 == y2)
    x3, y3 = gridder.scatter(area, size)
    assert numpy.all(x1 != x3) and numpy.all(y1 != y3)

def test_utils_contaminate():
    "utils.contaminate uses diff noise"
    size = 1235
    data = numpy.linspace(-100., 12255., size)
    noise = 244.4
    for i in xrange(20):
        d1 = utils.contaminate(data, noise)
        d2 = utils.contaminate(data, noise)
        assert numpy.all(d1 != d2)

def test_utils_contaminate_seed():
    "utils.contaminate uses same noise using same random seed"
    size = 1000
    data = numpy.linspace(-1000, 1000, size)
    noise = 10
    for seed in numpy.random.randint(low=0, high=10000, size=20):
        d1 = utils.contaminate(data, noise, seed=seed)
        d2 = utils.contaminate(data, noise, seed=seed)
        assert numpy.all(d1 == d2)

def test_utils_contaminate_seed_noseed():
    "utils.contaminate uses diff noise after using random seed"
    size = 1000
    data = numpy.linspace(-1000, 1000, size)
    noise = 10
    seed = 45212
    d1 = utils.contaminate(data, noise, seed=seed)
    d2 = utils.contaminate(data, noise, seed=seed)
    assert numpy.all(d1 == d2)
    d3 = utils.contaminate(data, noise)
    assert numpy.all(d1 != d3)
