"""
Equivalent layer processing.
"""
import numpy
from scipy.sparse import linalg

from fatiando.gravmag import sphere as kernel
from fatiando import utils


class Data(object):
    """
    Wrap data for use in the equivalent layer inversion.
    """

    def __init__(self, x, y, z, data):
        self.x = x
        self.y = y
        self.z = z
        self.data = data
        self.size = len(data)

class Gz(Data):
    """
    Wrap gravity anomaly data to be used in the equivalent layer processing.
    """

    def __init__(self, x, y, z, data):
        Data.__init__(self, x, y, z, data)

    def sensitivity(self, grid):
        x, y, z = self.x, self.y, self.z
        sens = numpy.transpose([kernel.gz(x, y, z, [s], dens=1.) for s in grid])
        return sens

class TotalField(Data):
    """
    Wrap total field magnetic anomaly data to be used in the equivalent layer
    processing.
    """

    def __init__(self, x, y, z, data, inc, dec):
        Data.__init__(self, x, y, z, data)
        self.inc = inc
        self.dec = dec

    def sensitivity(self, grid):
        x, y, z = self.x, self.y, self.z
        inc, dec = self.inc, self.dec
        mag = utils.dircos(inc, dec)
        sens = numpy.transpose([kernel.tf(x, y, z, [s], inc, dec, pmag=mag)
            for s in grid])
        return sens

def classic(data, grid, damping=0.):
    """
    The classic equivalent layer with damping regularization.
    """
    sensitivity = numpy.concatenate([d.sensitivity(grid) for d in data])
    datavec = numpy.concatenate([d.data for d in data])
    system = numpy.dot(sensitivity, sensitivity.T)
    order = len(system)
    n = len(datavec)
    trace = numpy.trace(system)
    scale = float(n)/trace
    if damping != 0.:
        system[range(order), range(order)] += damping
    tmp = linalg.cg(system*scale, datavec*scale)[0]
    estimate = numpy.dot(sensitivity.T, tmp)
    pred = numpy.dot(sensitivity, estimate)
    predicted = []
    start = 0
    for d in data:
        predicted.append(pred[start:start + d.size])
        start += d.size
    return estimate, predicted
