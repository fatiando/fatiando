"""
3D potential field inversion by planting anomalous densities.

Implements the method of Uieda and Barbosa (2012a) with improvements by
Uieda and Barbosa (2012b).

A "heuristic" inversion for compact 3D geologic bodies. Performs the inversion
by iteratively growing the estimate around user-specified "seeds". Supports
various kinds of data (gravity, gravity tensor).

**References**

Uieda, L., and V. C. F. Barbosa (2012a), Robust 3D gravity gradient inversion
by planting anomalous densities, Geophysics, 77(4), G55-G66,
doi:10.1190/geo2011-0388.1

Uieda, L., and V. C. F. Barbosa (2012b),
Use of the "shape-of-anomaly" data misfit in 3D inversion by planting anomalous
densities, SEG Technical Program Expanded Abstracts, 1-6,
doi:10.1190/segam2012-0383.1

----

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
        return self.regul_param*np.linalg.norm(alpha*self.data - predicted)

    def value(self, p):
        residuals = self.data - self.predicted(p)
        return np.linalg.norm(residuals)/self.dnorm

    def config(self, seeds, compactness, tol):
        """
        """
        assert len(seeds) > 0, "No seeds provided."
        assert compactness >= 0, "Compactness parameter must be positive."
        assert tol >= 0, 'tol parameter must be positive.'
        self.seeds = _sow(seeds, self.mesh)
        self.nseeds = len(seeds)
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
        Run the planting algorithm on the given data.

        The estimated parameter vector can be accessed through the
        ``p_`` attribute.

        The ``estimate_`` attribute is a list the mesh elements with non-zero
        physical properties (according to ``p_``).
        """
        p, neighbors, misfit, compactness, goal = self._init_planting()
        for iteration in range(self.nparams - len(self.seeds)):
            grew = False
            for s in range(self.nseeds):
                new = self._grow(s, neighbors, p, misfit, compactness, goal)
                if new is not None:
                    grew = True
                    goal = new['goal']
                    misfit = new['misfit']
                    compactness = new['compactness']
                    n = new['neighbor']
                    p[n.index] = n.prop
                    self._update_neighbors(n, s, neighbors, p)
            if not grew:
                break
        self.p_ = p
        return self

    def _init_planting(self):
        """
        Initialize the planting inversion.
        """
        p = np.zeros(self.nparams, dtype=np.float)
        for s in self.seeds:
            p[s.index] = s.prop
        neighbors = []
        nonzero = np.nonzero(p)[0]
        for s, seed in enumerate(self.seeds):
            tmp = seed.neighbors.difference(nonzero)
            for i in range(s):
                tmp = tmp.difference(neighbors[i])
            neighbors.append(tmp)
        misfit = self.value(p)
        compactness = 0
        goal = self.shape_of_anomaly(p) + self.mu*compactness
        return p, neighbors, misfit, compactness, goal

    def _update_neighbors(self, n, seed, neighbors, p):
        """
        Remove n from the list of neighbors and add it's neighbors.

        WARNING: Changes 'neighbors' in place.
        """
        neighbors[seed].remove(n)
        tmp = n.neighbors
        # Remove the ones that are already in the estimate
        tmp = tmp.difference(np.nonzero(p)[0])
        for already_neighbors in neighbors:
            tmp = tmp.difference(already_neighbors)
        neighbors[seed].update(tmp)

    def _grow(self, seed, neighbors, p, misfit, compactness, goal):
        """
        Grow the given seed through the accretion of one of it's neighbors.

        Returns the best neighbor in a dictionary. None if no neighbors are
        good enough.
        """
        best = None
        for n in neighbors[seed]:
            p[n.index] = n.prop
            newmisfit = self.value(p)
            if (newmisfit >= misfit or
                abs(newmisfit - misfit)/misfit < self.tol):
                p[n.index] = 0
                continue
            newcompactness = compactness + n.distance_to(self.seeds[seed])
            newgoal = self.shape_of_anomaly(p) + self.mu*compactness
            p[n.index] = 0
            if best is None or newgoal < bestgoal:
                best = n
                bestgoal = newgoal
                bestmisfit = newmisfit
                bestcompactness = newcompactness
        if best is not None:
            best = dict(neighbor=best, goal=bestgoal, misfit=bestmisfit,
                        compactness=bestcompactness)
        return best


class _Cell(object):
    """
    A cell in a mesh.

    Knows its index in the mesh and holds a physical property value. Can
    calculate who it's neighbors are in the mesh and the distance to a given
    cell.

    Can be compared with equality (==) to an integer, assuming that the integer
    is an index of the mesh, or another _Cell instance.
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
        elif isinstance(other, _Cell):
            return self.index == other.index
        else:
            raise ValueError("Can't compare Neighbor object to {} type".format(
                str(type(other))))

    def __hash__(self):
        "A unique value identifying this neighbor (it's index in the mesh)"
        # This is needed to put _Cell objects in a set or dict
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
