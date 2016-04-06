"""
"""
from __future__ import division, unicode_literals, print_function
from future.builtins import super, object, range
import numpy as np
import bisect
import math

from ..inversion import Misfit
from . import prism as prism_kernel


class PlantingMagnetic(object):
    pass

class _PlantingAlgorithm(object):

    def predicted(self, p):
        pass

    def shape_of_anomaly(self, p):
        pass

    def config(self):
        pass


    def fit(self):
        pass


class PlantingGravity(Misfit, _PlantingAlgorithm):
    """
    """

    def __init__(self, x, y, z, data, mesh, field='gz'):
        super().__init__(data=data, nparams=mesh.size, islinear=False)
        self.x = x
        self.y = y
        self.z = z
        self.mesh = mesh
        self.field = field
        self.seeds = None
        self.mu = None
        self.tol = None
        self.effects = {}
        self.kernel = prism_kernel.gz
        self.dnorm = np.linalg.norm(data)

    def jacobian(self, p):
        pass

    def predicted(self, p):
        pred = np.zeros(self.ndata)
        for i in np.nonzero(p)[0]:
            if i not in self.effects:
                self.effects[i] = self.kernel(self.x, self.y, self.z,
                                              [self.mesh[i]], dens=p[i])
            pred += self.effects[i]
        return pred

    def shape_of_anomaly(self, p):
        predicted = self.predicted(p)
        alpha = np.sum(self.data*predicted)/self.dnorm**2
        return np.linalg.norm(alpha*self.data - predicted)

    def config(self, seeds, compactness, tol):
        """
        """
        self.seeds = _sow(seeds, self.mesh)
        self.compactness = compactness
        self.mu = compactness/(sum(self.mesh.shape)/3)
        self.tol = tol
        # The prism effects change when seeds change because they include the
        # physical property of each neighbor, which are inherited from the
        # seeds
        self.effects = {}
        return self

    def fit(self):
        """
        """
        p = np.zeros(self.nparams, dtype=np.float)
        added = set()
        for s in self.seeds:
            p[s.index] = s.prop
            added.add(s.index)
        neighbors = {}
        allneighbors = set()
        for s in self.seeds:
            tmp = s.neighbors.difference(allneighbors).difference(added)
            allneighbors.update(tmp)
            neighbors[s] = tmp
        misfit = math.sqrt(self.value(p))
        compactness = 0
        goal = self.shape_of_anomaly(p) + self.mu*compactness
        for iteration in range(self.nparams - len(self.seeds)):
            grew = False
            for s in self.seeds:
                best = None
                for n in neighbors[s]:
                    p[n.index] = n.prop
                    newmisfit = math.sqrt(self.value(p))
                    if (newmisfit >= misfit or
                        abs(newmisfit - misfit)/misfit < self.tol):
                        continue
                    newcompactness = compactness + n.distance_to(s)
                    newgoal = self.shape_of_anomaly(p) + self.mu*compactness
                    p[n.index] = 0
                    if best is None or newgoal < bestgoal:
                        best = n
                        bestgoal = newgoal
                        bestmisfit = newmisfit
                        bestcompactness = newcompactness
                if best is not None:
                    grew = True
                    goal = bestgoal
                    misfit = bestmisfit
                    compactness = bestcompactness
                    p[best.index] = best.prop
                    added.add(best.index)
                    neighbors[s].remove(best)
                    allneighbors.remove(best)
                    tmp = best.neighbors.difference(allneighbors).difference(added)
                    neighbors[s].update(tmp)
                    allneighbors.update(tmp)
            if not grew:
                break
        self.p_ = p
        return self


class _Cell(object):
    """
    A cell in a mesh.
    """

    def __init__(self, index, prop, mesh):
        self.index = index
        self.prop = prop
        self.mesh = mesh

    def __repr__(self):
        return str(self.index)

    def __eq__(self, other):
        "Compare if another neighbor (or an index in the mesh) is this one."
        if isinstance(other, int):
            return self.index == other
        elif hasattr(other, 'index'):
            return self.index == other.index
        else:
            raise ValueError("Can't compare Neighbor object to {} type".format(
                str(type(other))))

    def __hash__(self):
        "A unique value identifying this neighbor (it's index in the mesh)"
        return self.index

    def distance_to(self, other):
        """
        Calculate the distance between this cell and another in the mesh.
        """
        ni, nj, nk = np.unravel_index(self.index, self.mesh.shape)
        mi, mj, mk = np.unravel_index(other.index, self.mesh.shape)
        return math.sqrt((ni - mi)**2 + (nj - mj)**2 + (nk - mk)**2)

    @property
    def neighbors(self):
        "Find the neighboring prisms in the mesh."
        nz, ny, nx = self.mesh.shape
        n = self.index
        indexes = []
        # The guy above
        tmp = n - nx*ny
        if tmp > 0:
            indexes.append(tmp)
        # The guy below
        tmp = n + nx*ny
        if tmp < self.mesh.size:
            indexes.append(tmp)
        # The guy in front
        tmp = n + 1
        if n % nx < nx - 1:
            indexes.append(tmp)
        # The guy in the back
        tmp = n - 1
        if n % nx != 0:
            indexes.append(tmp)
        # The guy to the left
        tmp = n + nx
        if n % (nx*ny) < nx*(ny - 1):
            indexes.append(tmp)
        # The guy to the right
        tmp = n - nx
        if n % (nx*ny) >= nx:
            indexes.append(tmp)
        # Filter out the ones that are masked (topography)
        return set([self.__class__(i, self.prop, self.mesh)
                    for i in indexes if self.mesh[i] is not None])


def _sow(locations, mesh):
    seeds = []
    for x, y, z, props in locations:
        index = _find_index((x, y, z), mesh)
        # Check for duplicates
        if index not in (s for s in seeds):
            seeds.append(_Cell(index, props['density'], mesh))
    return seeds


def _find_index(point, mesh):
    """
    Find the index of the cell that has point inside it.
    """
    x1, x2, y1, y2, z1, z2 = mesh.bounds
    nz, ny, nx = mesh.shape
    xs = mesh.get_xs()
    ys = mesh.get_ys()
    zs = mesh.get_zs()
    x, y, z = point
    if (x <= x2 and x >= x1 and y <= y2 and y >= y1 and
        ((z <= z2 and z >= z1 and mesh.zdown) or
         (z >= z2 and z <= z1 and not mesh.zdown))):
        # -1 because bisect gives the index z would have. I want to know
        # what index z comes after
        k = bisect.bisect_left(zs, z) - 1
        j = bisect.bisect_left(ys, y) - 1
        i = bisect.bisect_left(xs, x) - 1
        seed = i + j*nx + k*nx*ny
        # Check if the cell is not masked (topography)
        if mesh[seed] is not None:
            return seed
    return None
