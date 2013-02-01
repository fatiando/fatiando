"""
3D potential field inversion by planting anomalous densities.

Implements the method of Uieda and Barbosa (2012).

A "heuristic" inversion for compact 3D geologic bodies. Performs the inversion
by iteratively growing the estimate around user-specified "seeds". Supports
various kinds of data (gravity, gravity tensor).

The inversion is performed by function
:func:`~fatiando.gravmag.harvester.harvest`. The required information, such as
observed data, seeds, and regularization, are passed to the function though
seed classes and data modules.

**Functions**

* :func:`~fatiando.gravmag.harvester.harvest`: Performs the inversion
* :func:`~fatiando.gravmag.harvester.wrapdata`: Creates the data modules
  required by ``harvest``
* :func:`~fatiando.gravmag.harvester.loadseeds`: Loads a set of points and
  physical properties that specify the seeds from a file
* :func:`~fatiando.gravmag.harvester.sow`: Creates the seeds from a set of
  points that specify their locations

**Usage**

The recommened way of generating the required seeds and data modules is to use
the helper functions :func:`~fatiando.gravmag.harvester.wrapdata`,
:func:`~fatiando.gravmag.harvester.loadseeds`, and
:func:`~fatiando.gravmag.harvester.sow`.

A typical script to run the inversion on a data set looks like::

    import numpy
    import fatiando as ft
    # Load the data from a file
    xp, yp, zp, gz = numpy.loadtxt('mydata.xyz', unpack=True)
    # Create a mesh assuming that 'bounds' are the limits of the mesh and
    # 'shape' is the number of prisms in each dimension
    bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
    shape = (nz, ny, nx)
    mesh = ft.mesher.PrismMesh(bounds, shape)
    # Make the data modules
    dms = ft.gravmag.harvester.wrapdata(mesh, xp, yp, zp, gz=gz)
    # Read the seed locations and physical properties from a file
    seeds = ft.gravmag.harvester.sow(ft.gravmag.harvester.loadseeds('myseedfile.txt'),
                                 mesh, mu=0.1)
    # Run the inversion
    estimate, goals, misfits = ft.gravmag.harvester.harvest(dms, seeds)
    # fill the mesh with the density values
    mesh.addprop('density', estimate['density'])
    # Save the mesh in UBC-GIF format
    mesh.dump('result.msh', 'result.den', 'density')


**Seeds**

A seed class determines what kind of geometric element is used to parametrize
the anomalous density distribution. For example, if you use a SeedPrism, the
output of :func:`~fatiando.gravmag.harvester.harvest` will be a list of prisms
that make up the estimated density distribution.

* :class:`~fatiando.gravmag.harvester.SeedPrism`

**Data Modules**

Data modules wrap the observed data and calculate the predicted data for a given
parametrization.

* :class:`~fatiando.gravmag.harvester.DMPrismGz`
* :class:`~fatiando.gravmag.harvester.DMPrismGxx`
* :class:`~fatiando.gravmag.harvester.DMPrismGxy`
* :class:`~fatiando.gravmag.harvester.DMPrismGxz`
* :class:`~fatiando.gravmag.harvester.DMPrismGyy`
* :class:`~fatiando.gravmag.harvester.DMPrismGyz`
* :class:`~fatiando.gravmag.harvester.DMPrismGzz`

**References**

Uieda, L., and V. C. F. Barbosa (2012), Robust 3D gravity gradient inversion by
planting anomalous densities, Geophysics, 77(4), G55-G66,
doi:10.1190/geo2011-0388.1

----

"""
import json
import time
from math import sqrt

import numpy

from fatiando import utils
import fatiando.logger

log = fatiando.logger.dummy('fatiando.gravmag.harvester')


def loadseeds(fname):
    """
    Load a set of seed locations and physical properties from a file.

    The output can then be used with the :func:`~fatiando.gravmag.harvester.sow`
    function.

    The seed file should be formatted as::

        [
            [x1, y1, z1, {"density":dens1}],
            [x2, y2, z2, {"density":dens2, "magnetization":mag2}],
            [x3, y3, z3, {"magnetization":mag3, "inclination":inc3,
                          "declination":dec3}],
            ...
        ]

    x, y, z are the coordinates of the seed and the dict (``{'density':2670}``)
    are its physical properties.

    .. warning::

        Must use ``"``, not ``'``, in the physical property names!

    Each seed can have different kinds of physical properties. If inclination
    and declination are not given, will use the inc and dec of the inducing
    field (i.e., no remanent magnetization).

    The techie among you will recognize that the seed file is in JSON format.

    Remember: the coordinate system is x->North, y->East, and z->Down

    Parameters:

    * fname : str or file
        Open file object or filename string

    Returns:

    * [[x1, y1, z1, props1], [x2, y2, z2, props2],  ...]
        (x, y, z) are the points where the seeds will be placed
        and *props* is dict with the values of the physical properties of each,
        seed.

    Example:

        >>> from StringIO import StringIO
        >>> file = StringIO(
        ...     '[[1, 2, 3, {"density":4, "magnetization":5}],' +
        ...     ' [6, 7, 8, {"magnetization":-1}]]')
        >>> seeds = loadseeds(file)
        >>> for s in seeds:
        ...     print s
        [1, 2, 3, {u'magnetization': 5, u'density': 4}]
        [6, 7, 8, {u'magnetization': -1}]

    """
    openned = False
    if isinstance(fname, str):
        fname = open(fname)
        openned = True
    seeds = json.load(fname)
    if openned:
        fname.close()
    return seeds

def sow(locations, mesh):
    """
    Create the seeds given a list of (x,y,z) coordinates and physical 
    properties.
    """
    seeds = []
    for point, props in locations:
        index = _find_index(point, mesh)
        if index is None:
            raise ValueError("Couldn't find seed at location %s" % (str(point)))
        # Check for duplicates
        if index not in (s.i for s in seeds):
            seeds.append(Seed(index, props))
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
    if x <= x2 and x >= x1 and y <= y2 and y >= y1 and z <= z2 and z >= z1:
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

def harvest(data, seeds, mesh, compactness, threshold):
    """
    """
    nseeds = len(seeds)
    # Initialize the estimate with the seeds
    estimate = dict((s.i, s.props) for s in seeds)
    # Initialize the neighbors list
    neighbors = []
    for seed in seeds:
        neighbors.extend(_get_neighbors(seed, neighbors, estimate, mesh, data))
    # Initialize the predicted data
    predicted = _init_predicted(data, seeds, mesh)
    # Start the goal function, data-misfit function and regularizing function
    totalgoal = _shapefunc(data, predicted)
    totalmisfit = _misfitfunc(data, predicted)
    regularizer = 0.
    # Weight the regularizing function by the mean extent of the mesh
    mu = compactness*1./(sum(mesh.shape)/3.) 
    # Begin the growth process
    for iteration in xrange(mesh.size - nseeds):
        grew = False # To check if at least one seed grew (stopping criterion)
        for s in xrange(nseeds):
            best, bestgoal, bestmisfit, bestregularizer = _grow(neighbors[s], 
                data, predicted, totalmisfit, mu, regularizer, threshold)
            # If there was a best, add to estimate, remove it, and add its 
            # neighbors
            if best is not None:
                if best.i not in estimate:
                    estimate[best.i] = {}
                estimate[best.i].update(best.props)
                totalgoal = bestgoal
                totalmisfit = bestmisfit
                regularizer = bestregularizer
                predicted += best.effect
                neighbors[s].pop(best.i)
                neighbors[s].update(
                    _get_neighbors(best, neighbors, estimate, mesh, data))
                del best
                grew = True
        if not grew:
            break
    return _fmt_estimate(estimate), predicted

def _init_predicted(data, seeds, mesh):
    """
    Make a list with the initial predicted data vectors (effect of seeds)
    """
    predicted = []
    for d in data:
        p = numpy.zeros(len(d.observed), dtype='f')
        for seed in seeds:
            p += d.effect(mesh[seed.i], seed.props)
        predicted.append(p)
    return predicted 

def _fmt_estimate(estimate):
    """
    Make a nice dict with the estimated physical properties in separate array
    """
    output = {}
    for i, props in estimate:
        for p in props:
            if p not in output:
                output[p] = utils.SparseList(mesh.size)
            output[p][i] = props[p]
    return output

def _grow(neighbors, data, predicted, totalmisfit, mu, regularizer, threshold):
    """
    Find the neighbor with smallest goal function that also decreases the 
    misfit
    """
    best = None
    bestgoal = None
    bestmisfit = None
    bestregularizer = None
    for n in neighbors:
        pred = [p + e for p, e in zip(predicted, n.effect)]
        misfit = _misfitfunc(data, pred)
        if (misfit < totalmisfit and 
            abs(misfit - totalmisfit)/totalmisfit >= threshold):
            reg = regularizer + n.distance
            goal = _shapefunc(data, pred) + mu*reg
            if bestgoal is None or goal < bestgoal:
                bestgoal = goal
                best = n
                bestmisfit = misfit
                bestregularizer = reg
    return best, bestgoal, bestmisfit, bestregularizer

def _shapefunc(data, predicted):
    """
    Calculate the total shape of anomaly function between the observed and 
    predicted data.
    """
    result = 0.
    for d, p in zip(data, predicted):
        alpha = numpy.sum(d.observed*p)/d.norm**2
        result += numpy.norm(alpha*d.observed - p)
    return result

def _misfitfunc(data, predicted):
    """
    Calculate the total data misfit function between the observed and predicted
    data.
    """
    return sum(numpy.norm(d.observed - p)/d.norm
               for d, p in zip(data, predicted))

def _get_neighbors(cell, neighborhood, estimate, mesh, data):
    """
    Return a dict with the new neighbors of cell. 
    keys are the index of the neighbors in the mesh. values are the Neighbor 
    objects.
    """
    indexes = [n for n in _neighbor_indexes(cell.i, mesh)
               if not _is_neighbor(n, props, neighborhood) 
                  and not _in_estimate(n, props, estimate)]
    neighbors = dict(
        (i, Neighbor(
            i, cell.props, cell.seed, _distance(i, cell.seed, mesh), 
            _calc_effect(i, cell.props, mesh, data))) 
        for i in indexes)
    return neighbors

def _calc_effect(index, props, mesh, data):
    """
    Calculate the effect of cell mesh[index] with physical properties prop for
    each data set.
    """
    cell = mesh[index]
    return [d.effect(cell, props) for d in data]

def _distance(n, m, mesh):
    """
    Calculate the distance (in number of cells) between cells n and m in mesh.
    """
    ni, nj, nk = _index2ijk(n, mesh)
    mi, mj, mk = _index2ijk(m, mesh)
    return sqrt((ni - mi)**2 + (nj - mj)**2 + (nk - mk)**2)

def _index2ijk(index, mesh):
    """
    Transform the index of a cell in mesh to a 3-dimensional (i,j,k) index.
    """
    nz, ny, nx = mesh.shape
    k = index/(nx*ny)
    j = (index - k*(nx*ny))/nx
    i = (index - k*(nx*ny) - j*nx)
    return i, j, k

def _in_estimate(index, props, estimate):
    """
    Check is index is in estimate with props
    """
    if index in estimate:
        for p in props:
            if p in estimate[index]:
                return True
    return False
    
def _is_neighbor(index, props, neighborhood):
    """
    Check if index is already in the neighborhood with props
    """    
    for neighbors in neighborhood:
        for n in neighbors:
            if index == n.i:
                for p in props:
                    if p in n.props:
                        return True
    return False

def _neighbor_indexes(n, mesh):
    """Find the indexes of the neighbors of n"""
    nz, ny, nx = mesh.shape
    indexes = []
    # The guy above
    tmp = n - nx*ny
    if tmp > 0:
        indexes.append(tmp)
    # The guy bellow
    tmp = n + nx*ny
    if tmp < mesh.size:
        indexes.append(tmp)
    # The guy in front
    tmp = n + 1
    if n%nx < nx - 1:
        indexes.append(tmp)
    # The guy in the back
    tmp = n - 1
    if n%nx != 0:
        indexes.append(tmp)
    # The guy to the left
    tmp = n + nx
    if n%(nx*ny) < nx*(ny - 1):
        indexes.append(tmp)
    # The guy to the right
    tmp = n - nx
    if n%(nx*ny) >= nx:
        indexes.append(tmp)
    # Filter out the ones that do not exist or are masked (topography)
    return [i for i in indexes if i is not None and mesh[i] is not None]

class Seed(object):
    """
    A seed.
    """
    
    def __init__(self, i, props):
        self.i = i
        self.props = props
        self.seed = i
    
class Neighbor(object):
    """
    A neighbor.
    """
    
    def __init__(self, i, props, seed, distance, effect):
        self.i = i
        self.props = props
        self.seed = seed
        self.distance = distance
        self.effect = effect

class Data(object):
    """
    A container for some potential field data.
    
    Know about its data, observation positions, nature of the mesh, and how
    to calculate the effect of a single cell.
    """
    
    def __init__(self, x, y, z, data, meshtype):
        self.x = x
        self.y = y
        self.z = z
        self.observed = data
        self.size = len(data)
        self.norm = numpy.norm(data)
        if self.meshtype not in ['prism']:
            raise AttributeError("Invalid mesh type '%s'" % (meshtype))
        if self.meshtype == 'prism':
            import fatiando.gravmag.prism
            self.effectmodule = fatiando.gravmag.prism

class Potential(Data):
    """
    A container for data of the gravitational potential.
    """

    def __init__(self, x, y, z, data, meshtype='prism'):
        Data.__init__(self, x, y, z, data, meshtype)
        self.prop = 'density'
        self.effectfunc = self.effectmodule.potential

    def effect(self, prism, props):
        """Calculate the effect of a prism with the given physical props"""
        if self.prop not in props:
            return numpy.zeros(self.size, dtype='f')
        return self.effectfunc(self.x, self.y, self.z, [prism], 
            props[self.prop])

class Gz(Potential):
    """
    A container for data of the gravity anomaly.
    """
    
    def __init__(self, x, y, z, data, meshtype='prism'):
        Potential.__init__(self, x, y, z, data, meshtype)
        self.effectfunc = self.effectmodule.gz

class Gxx(Potential):
    """
    A container for data of the xx (north-north) component of the gravity 
    gradient tensor.
    """
    
    def __init__(self, x, y, z, data, meshtype='prism'):
        Potential.__init__(self, x, y, z, data, meshtype)
        self.effectfunc = self.effectmodule.gxx

class Gxy(Potential):
    """
    A container for data of the xy (north-east) component of the gravity 
    gradient tensor.
    """
    
    def __init__(self, x, y, z, data, meshtype='prism'):
        Potential.__init__(self, x, y, z, data, meshtype)
        self.effectfunc = self.effectmodule.gxy

class Gxz(Potential):
    """
    A container for data of the xz (north-vertical) component of the gravity 
    gradient tensor.
    """
    
    def __init__(self, x, y, z, data, meshtype='prism'):
        Potential.__init__(self, x, y, z, data, meshtype)
        self.effectfunc = self.effectmodule.gxz

class Gyy(Potential):
    """
    A container for data of the yy (east-east) component of the gravity 
    gradient tensor.
    """
    
    def __init__(self, x, y, z, data, meshtype='prism'):
        Potential.__init__(self, x, y, z, data, meshtype)
        self.effectfunc = self.effectmodule.gyy

class Gyz(Potential):
    """
    A container for data of the yz (east-vertical) component of the gravity 
    gradient tensor.
    """
    
    def __init__(self, x, y, z, data, meshtype='prism'):
        Potential.__init__(self, x, y, z, data, meshtype)
        self.effectfunc = self.effectmodule.gyz

class Gzz(Potential):
    """
    A container for data of the zz (vertical-vertical) component of the gravity 
    gradient tensor.
    """
    
    def __init__(self, x, y, z, data, meshtype='prism'):
        Potential.__init__(self, x, y, z, data, meshtype)
        self.effectfunc = self.effectmodule.gzz

class TotalField(Potential):
    """
    A container for data of the total field magnetic anomaly.
    """
    
    def __init__(self, x, y, z, data, inc, dec, meshtype='prism'):
        if meshtype != 'prism':
            raise AttributeError(
                "Unsupported mesh type '%s' for total field anomaly." 
                % (meshtype))
        Potential.__init__(self, x, y, z, data, meshtype)
        self.effectfunc = self.effectmodule.tf
        self.prop = 'magnetization'
        self.inc = inc
        self.dec = dec

    def effect(self, prism, props):
        """Calculate the effect of a prism with the given physical props"""
        if self.prop not in props:
            return numpy.zeros(self.size, dtype='f')
        pinc, pdec = None, None
        if 'inclination' in props:
            pinc = props['inclinaton']
        if 'declination' in props:
            pdec = props['declination']
        return self.effectfunc(self.x, self.y, self.z, [prism], self.inc, 
            self.dec, pmag=props[self.prop], pinc=pinc, pdec=pdec)
