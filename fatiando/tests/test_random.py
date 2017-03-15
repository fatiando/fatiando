from __future__ import absolute_import, division
from future.builtins import range
import numpy
from numpy.testing import assert_allclose
from fatiando import utils


def test_utils_contaminate():
    "utils.contaminate generates noise with 0 mean and right stddev"
    size = 10 ** 6
    data = numpy.zeros(size)
    std = 4.213
    for i in range(20):
        noise = utils.contaminate(data, std)
        assert abs(noise.mean()) < 10 ** -10, 'mean:%g' % (noise.mean())
        assert abs(noise.std() - std) / std < 0.01, 'std:%g' % (noise.std())


def test_utils_contaminate_seed():
    "utils.contaminate noise with 0 mean and right stddev using random seed"
    size = 10 ** 6
    data = numpy.zeros(size)
    std = 4400.213
    for i in range(20):
        noise = utils.contaminate(data, std, seed=i)
        assert abs(noise.mean()) < 10 ** - \
            10, 's:%d mean:%g' % (i, noise.mean())
        assert abs(noise.std() - std) / std < 0.01, \
            's:%d std:%g' % (i, noise.std())


def test_utils_contaminate_diff():
    "utils.contaminate uses diff noise"
    size = 1235
    data = numpy.linspace(-100., 12255., size)
    noise = 244.4
    for i in range(20):
        d1 = utils.contaminate(data, noise)
        d2 = utils.contaminate(data, noise)
        assert numpy.all(d1 != d2)


def test_utils_contaminate_same_seed():
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
