"""
3D potential field inversion by planting anomalous densities.

Implements the method of Uieda and Barbosa (2012a) with improvements by
Uieda and Barbosa (2012b).

A "heuristic" inversion for compact 3D geologic bodies. Performs the inversion
by iteratively growing the estimate around user-specified "seeds". Supports
various kinds of data (gravity, gravity tensor).

The inversion is performed by function
:func:`~fatiando.gravmag.harvester.harvest`. The required information, such as
observed data, seeds, and regularization, are passed to the function through
classes :class:`~fatiando.gravmag.harvester.Seed` and
:class:`~fatiando.gravmag.harvester.Potential`,
:class:`~fatiando.gravmag.harvester.Gz`,
:class:`~fatiando.gravmag.harvester.Gxx`, etc.

See the :ref:`Cookbook <cookbook>` for some example applications to synthetic
data.

**Functions**

* :func:`~fatiando.gravmag.harvester.harvest`: Performs the inversion
* :func:`~fatiando.gravmag.harvester.iharvest`: Iterator to step through the
  inversion one accretion at a time
* :func:`~fatiando.gravmag.harvester.sow`: Creates the seeds from a set of
  (x, y, z) points and physical properties
* :func:`~fatiando.gravmag.harvester.loadseeds`: Loads from a JSON file a set
  of (x, y, z) points and physical properties that specify the seeds. Pass
  output to :func:`~fatiando.gravmag.harvester.sow`
* :func:`~fatiando.gravmag.harvester.weights`: Computes data weights based on
  the distance to the seeds

**Data types**

* :class:`~fatiando.gravmag.harvester.Potential`: gravitational potential
* :class:`~fatiando.gravmag.harvester.Gz`: vertical component of gravitational
  acceleration (i.e., gravity anomaly)
* :class:`~fatiando.gravmag.harvester.Gxx`: North-North component of the
  gravity gradient tensor
* :class:`~fatiando.gravmag.harvester.Gxy`: North-East component of the gravity
  gradient tensor
* :class:`~fatiando.gravmag.harvester.Gxz`: North-vertical component of the
  gravity gradient tensor
* :class:`~fatiando.gravmag.harvester.Gyy`: East-East component of the gravity
  gradient tensor
* :class:`~fatiando.gravmag.harvester.Gyz`: East-vertical component of the
  gravity gradient tensor
* :class:`~fatiando.gravmag.harvester.Gzz`: vertical-vertical component of the
  gravity gradient tensor

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
import json
import bisect
from math import sqrt

import numpy

from fatiando.gravmag import prism as prism_engine
from fatiando.gravmag import tesseroid as tesseroid_engine
from fatiando import utils
from fatiando.mesher import Prism, Tesseroid


def loadseeds(fname):
    """
    Load a set of seed locations and physical properties from a file.

    The output can then be used with the
    :func:`~fatiando.gravmag.harvester.sow` function.

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

    Removes seeds that would fall on the same location with overlapping
    physical properties.

    Parameters:

    * locations : list
        The locations and physical properties of the seeds. Should be a list
        like::

            [
                [x1, y1, z1, {"density":dens1}],
                [x2, y2, z2, {"density":dens2, "magnetization":mag2}],
                [x3, y3, z3, {"magnetization":mag3, "inclination":inc3,
                              "declination":dec3}],
                ...
            ]

    * mesh : :class:`fatiando.mesher.PrismMesh`
        The mesh that will be used in the inversion.

    Returns:

    * seeds : list of seeds
        The seeds that can be passed to
        :func:`~fatiando.gravmag.harvester.harvest`

    """
    seeds = []
    if mesh.celltype == Tesseroid:
        seedtype = TesseroidSeed
    elif mesh.celltype == Prism:
        seedtype = PrismSeed
    for x, y, z, props in locations:
        index = _find_index((x, y, z), mesh)
        if index is None:
            raise ValueError(
                "Couldn't find seed at location (%g,%g,%g)" % (x, y, z))
        # Check for duplicates
        if index not in (s.i for s in seeds):
            seeds.append(seedtype(index, (x, y, z), mesh[index], props))
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
        if mesh.zdown:
            # -1 because bisect gives the index z would have. I want to know
            # what index z comes after
            k = bisect.bisect_left(zs, z) - 1
        else:
            # If z is not positive downward, zs will not be sorted
            k = len(zs) - bisect.bisect_left(zs[::-1], z)
        j = bisect.bisect_left(ys, y) - 1
        i = bisect.bisect_left(xs, x) - 1
        seed = i + j * nx + k * nx * ny
        # Check if the cell is not masked (topography)
        if mesh[seed] is not None:
            return seed
    return None


def harvest(data, seeds, mesh, compactness, threshold, report=False):
    """
    Run the inversion algorithm and produce an estimate physical property
    distribution (density and/or magnetization).

    Parameters:

    * data : list of data (e.g., :class:`~fatiando.gravmag.harvester.Gz`)
        The data that will be inverted. Data used must match the physical
        properties given to the seeds (e.g., gravity data requires seeds to
        have ``'density'`` prop)

    * seeds : list of :class:`~fatiando.gravmag.harvester.Seed`
        Lits of seeds used to start the growth process of the inversion. Use
        :func:`~fatiando.gravmag.harvester.sow` to generate seeds.

    * mesh : :class:`fatiando.mesher.PrismMesh`
        The mesh used in the inversion. Will estimate the physical property
        distribution on this mesh

    * compactness : float
        The compactness regularing parameter (i.e., how much should the
        estimate be consentrated around the seeds). Must be positive. To find a
        good value for this, start with a small value (like 0.001), run the
        inversion and increase the value until desired compactness is achieved.

    * threshold : float
        Control how much the solution can grow (usually downward). In order for
        estimate to grow by the accretion of 1 prism, this prism must decrease
        the data misfit measure by *threshold* decimal percent. Depends on the
        size of the cells in the *mesh* and the distance from a cell to the
        observations. Use values between 0.001 and 0.000001.
        If cells are small and *threshold* is large (0.001), the seeds won't
        grow. If cells are large and *threshold* is small (0.000001), the seeds
        will grow too much.

    * report : True or False
        If ``True``, also will return a dict as::

            report = {'goal': goal_function_value,
                      'shape-of-anomaly': SOA_function_value,
                      'misfit': data_misfit_value,
                      'regularizer': regularizing_function_value,
                      'accretions': number_of_accretions}


    Returns:

    * estimate, predicted_data : a dict and a list
        *estimate* is a dict like::

            {'physical_property':array, ...}

        *estimate* contains the estimates physical properties. The properties
        present in *estimate* are the ones given to the seeds. Include the
        properties in the *mesh* using::

            mesh.addprop('density', estimate['density'])

        This way you can plot the estimate using :mod:`fatiando.vis.myv`.

        *predicted_data* is a list of numpy arrays with the predicted (model)
        data. The list is in the same order as *data*. To plot a map of the fit
        for visual inspection and a histogram of the residuals::

            from fatiando.vis import mpl
            mpl.figure()
            # Plot the observed and predicted data as contours for visual
            # inspection
            mpl.subplot(1, 2, 1)
            mpl.axis('scaled')
            mpl.title('Observed and predicted data')
            levels = mpl.contourf(x, y, gz, (ny, nx), 10)
            mpl.colorbar()
            # Assuming gz is the only data used
            mpl.contour(x, y, predicted[0], (ny, nx), levels)
            # Plot a histogram of the residuals
            residuals = gz - predicted[0]
            mpl.subplot(1, 2, 2)
            mpl.title('Residuals')
            mpl.hist(residuals, bins=10)
            mpl.show()
            # It's also good to see the mean and standard deviation of the
            # residuals
            print "Residuals mean:", residuals.mean()
            print "Residuals stddev:", residuals.std()


    """
    for accretions, update in enumerate(iharvest(data, seeds, mesh,
                                                 compactness, threshold)):
        continue
    estimate, predicted = update[:2]
    output = [fmt_estimate(estimate, mesh.size), predicted]
    if report:
        goal, misfit, regul = update[4:]
        soa = goal - compactness * 1. / (sum(mesh.shape) / 3.) * regul
        output.append({'goal': goal, 'misfit': misfit, 'regularizer': regul,
                       'accretions': accretions, 'shape-of-anomaly': soa})
    return output


def iharvest(data, seeds, mesh, compactness, threshold):
    """
    Same as the :func:`fatiando.gravmag.harvester.harvest` function but this
    one returns an iterator that yields the information of each accretion.

    Yields:

    * [estimate, predicted, new, neighbors, goal, misfit, regularizer]
        The unformated estimate, predicted data vectors, the new element added
        during this iteration, list of neighbors, goal function value, misfit,
        regularizing function value.

    The first yield contains the seeds. Thus ``new`` will be ``None``.

    To format the estimate in a way that can be added to a mesh, use
    function fmt_estimate of this module.

    """
    nseeds = len(seeds)
    estimate = dict((s.i, s.props) for s in seeds)
    neighbors = []
    for seed in seeds:
        neighbors.append(_get_neighbors(seed, neighbors, estimate, mesh, data))
    predicted = _init_predicted(data, seeds, mesh)
    totalgoal = _shapefunc(data, predicted)
    totalmisfit = _misfitfunc(data, predicted)
    regularizer = 0.
    # Weight the regularizing function by the mean extent of the mesh
    mu = compactness * 1. / (sum(mesh.shape) / 3.)
    yield [estimate, predicted, None, neighbors, totalgoal, totalmisfit,
           regularizer]
    accretions = 0
    for iteration in xrange(mesh.size - nseeds):
        grew = False  # To check if at least one seed grew (stopping criterion)
        for s in xrange(nseeds):
            best, bestgoal, bestmisfit, bestregularizer = _grow(
                neighbors[s], data, predicted, totalmisfit, mu, regularizer,
                threshold)
            if best is not None:
                if best.i not in estimate:
                    estimate[best.i] = {}
                estimate[best.i].update(best.props)
                totalgoal = bestgoal
                totalmisfit = bestmisfit
                regularizer = bestregularizer
                for p, e in zip(predicted, best.effect):
                    p += e
                neighbors[s].pop(best.i)
                neighbors[s].update(
                    _get_neighbors(best, neighbors, estimate, mesh, data))
                grew = True
                accretions += 1
                yield [estimate, predicted, best, neighbors, totalgoal,
                       totalmisfit, regularizer]
                del best
        if not grew:
            break


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


def fmt_estimate(estimate, size):
    """
    Make a nice dict with the estimated physical properties in separate arrays
    """
    output = {}
    for i in estimate:
        props = estimate[i]
        for p in props:
            if p not in output:
                output[p] = utils.SparseList(size)
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
        pred = [p + e for p, e in zip(predicted, neighbors[n].effect)]
        misfit = _misfitfunc(data, pred)
        if (misfit < totalmisfit and
                float(abs(misfit - totalmisfit)) / totalmisfit >= threshold):
            reg = regularizer + neighbors[n].distance
            goal = _shapefunc(data, pred) + mu * reg
            if bestgoal is None or goal < bestgoal:
                bestgoal = goal
                best = neighbors[n]
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
        alpha = numpy.sum(d.observed * p) / d.norm ** 2
        result += numpy.linalg.norm(alpha * d.observed - p)
    return result


def _misfitfunc(data, predicted):
    """
    Calculate the total data misfit function between the observed and predicted
    data.
    """
    result = 0.
    for d, p, in zip(data, predicted):
        residuals = d.observed - p
        result += sqrt(numpy.dot(d.weights * residuals, residuals)) / d.norm
    return result


def _get_neighbors(cell, neighborhood, estimate, mesh, data):
    """
    Return a dict with the new neighbors of cell.
    keys are the index of the neighbors in the mesh. values are the Neighbor
    objects.
    """
    indexes = [n for n in _neighbor_indexes(cell.i, mesh)
               if not _is_neighbor(n, cell.props, neighborhood)
               and not _in_estimate(n, cell.props, estimate)]
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
    return sqrt((ni - mi) ** 2 + (nj - mj) ** 2 + (nk - mk) ** 2)


def _index2ijk(index, mesh):
    """
    Transform the index of a cell in mesh to a 3-dimensional (i,j,k) index.
    """
    nz, ny, nx = mesh.shape
    k = index / (nx * ny)
    j = (index - k * (nx * ny)) / nx
    i = (index - k * (nx * ny) - j * nx)
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
            if index == neighbors[n].i:
                for p in props:
                    if p in neighbors[n].props:
                        return True
    return False


def _neighbor_indexes(n, mesh):
    """Find the indexes of the neighbors of n"""
    nz, ny, nx = mesh.shape
    indexes = []
    # The guy above
    tmp = n - nx * ny
    if tmp > 0:
        indexes.append(tmp)
    # The guy below
    tmp = n + nx * ny
    if tmp < mesh.size:
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
    if n % (nx * ny) < nx * (ny - 1):
        indexes.append(tmp)
    # The guy to the right
    tmp = n - nx
    if n % (nx * ny) >= nx:
        indexes.append(tmp)
    # Filter out the ones that do not exist or are masked (topography)
    return [i for i in indexes if i is not None and mesh[i] is not None]


class PrismSeed(Prism):

    """
    A seed that is a right rectangular prism.
    """

    def __init__(self, i, location, prism, props):
        Prism.__init__(self, prism.x1, prism.x2, prism.y1, prism.y2, prism.z1,
                       prism.z2, props=props)
        self.i = i
        self.seed = i
        self.x, self.y, self.z = location


class TesseroidSeed(Tesseroid):

    """
    A seed that is a tesseroid (spherical prism).
    """

    def __init__(self, i, location, tess, props):
        Tesseroid.__init__(self, tess.w, tess.e, tess.s, tess.n, tess.top,
                           tess.bottom, props=props)
        self.i = i
        self.seed = i
        self.x, self.y, self.z = location


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


def weights(x, y, seeds, influences, decay=2):
    """
    Calculate weights for the data based on the distance to the seeds.
    Use weights to ignore regions of data outside of the target anomaly.

    Parameters:

    * x, y : 1d arrays
        The x and y coordinates of the observations
    * seeds : list
        List of seeds, as returned by :func:`~fatiando.gravmag.harvester.sow`
    * influences : list of floats
        The respective diameter of influence for each seed. Observations
        outside the influence will have very small weights.
        A recommended value is aproximately the diameter of the anomaly
    * decay : float
        The decay factor for the weights. Low decay factor makes the weights
        spread out more. High decay factor makes the transition from large
        weights to low weights more abrupt.

    Returns:

    * weights : 1d array
        The calculated weights

    """
    distances = numpy.array([((x - s.x) ** 2 + (y - s.y) ** 2) / influence ** 2
                             for s, influence in zip(seeds, influences)])
    # min along axis=0 gets the smallest value from each column
    weights = numpy.exp(-(distances.min(axis=0) ** decay))
    return weights


class Data(object):

    """
    A container for some potential field data.

    Know about its data, observation positions, nature of the mesh, and how
    to calculate the effect of a single cell.
    """

    def __init__(self, x, y, z, data, weights, meshtype):
        self.x = x
        self.y = y
        self.z = z
        self.observed = data
        self.size = len(data)
        self.norm = numpy.linalg.norm(data)
        self.meshtype = meshtype
        if self.meshtype not in ['prism', 'tesseroid']:
            raise AttributeError("Invalid mesh type '%s'" % (meshtype))
        if self.meshtype == 'prism':
            self.engine = prism_engine
        if self.meshtype == 'tesseroid':
            self.engine = tesseroid_engine
        self.weights = weights


class Potential(Data):

    """
    A container for data of the gravitational potential.

    Coordinate system used: x->North y->East z->Down

    Parameters:

    * x, y, z : 1D arrays
        Arrays with the x, y, z coordinates of the data points

    * data : 1D array
        The values of the data at the observation points

    * weight : float or array
        The weight of this data set in the misfit function. Pass an array to
        give weights to each data points or a float to weight the entire misfit
        function. See function :func:`~fatiando.gravmag.harvester.weights`

    """

    def __init__(self, x, y, z, data, weights=1., meshtype='prism'):
        Data.__init__(self, x, y, z, data, weights, meshtype)
        self.prop = 'density'
        self.effectfunc = self.engine.potential

    def effect(self, prism, props):
        if self.prop not in props:
            return numpy.zeros(self.size, dtype='f')
        return self.effectfunc(self.x, self.y, self.z, [prism],
                               props[self.prop])


class Gz(Potential):

    """
    A container for data of the gravity anomaly.

    Coordinate system used: x->North y->East z->Down

    Parameters:

    * x, y, z : 1D arrays
        Arrays with the x, y, z coordinates of the data points

    * data : 1D array
        The values of the data at the observation points

    * weight : float or array
        The weight of this data set in the misfit function. Pass an array to
        give weights to each data points or a float to weight the entire misfit
        function. See function :func:`~fatiando.gravmag.harvester.weights`

    """

    def __init__(self, x, y, z, data, weights=1., meshtype='prism'):
        Potential.__init__(self, x, y, z, data, weights, meshtype)
        self.effectfunc = self.engine.gz


class Gxx(Potential):

    """
    A container for data of the xx (north-north) component of the gravity
    gradient tensor.

    Coordinate system used: x->North y->East z->Down

    Parameters:

    * x, y, z : 1D arrays
        Arrays with the x, y, z coordinates of the data points

    * data : 1D array
        The values of the data at the observation points

    * weight : float or array
        The weight of this data set in the misfit function. Pass an array to
        give weights to each data points or a float to weight the entire misfit
        function. See function :func:`~fatiando.gravmag.harvester.weights`

    """

    def __init__(self, x, y, z, data, weights=1., meshtype='prism'):
        Potential.__init__(self, x, y, z, data, weights, meshtype)
        self.effectfunc = self.engine.gxx


class Gxy(Potential):

    """
    A container for data of the xy (north-east) component of the gravity
    gradient tensor.

    Coordinate system used: x->North y->East z->Down

    Parameters:

    * x, y, z : 1D arrays
        Arrays with the x, y, z coordinates of the data points

    * data : 1D array
        The values of the data at the observation points

    * weight : float or array
        The weight of this data set in the misfit function. Pass an array to
        give weights to each data points or a float to weight the entire misfit
        function. See function :func:`~fatiando.gravmag.harvester.weights`

    """

    def __init__(self, x, y, z, data, weights=1., meshtype='prism'):
        Potential.__init__(self, x, y, z, data, weights, meshtype)
        self.effectfunc = self.engine.gxy


class Gxz(Potential):

    """
    A container for data of the xz (north-vertical) component of the gravity
    gradient tensor.

    Coordinate system used: x->North y->East z->Down

    Parameters:

    * x, y, z : 1D arrays
        Arrays with the x, y, z coordinates of the data points

    * data : 1D array
        The values of the data at the observation points

    * weight : float or array
        The weight of this data set in the misfit function. Pass an array to
        give weights to each data points or a float to weight the entire misfit
        function. See function :func:`~fatiando.gravmag.harvester.weights`

    """

    def __init__(self, x, y, z, data, weights=1., meshtype='prism'):
        Potential.__init__(self, x, y, z, data, weights, meshtype)
        self.effectfunc = self.engine.gxz


class Gyy(Potential):

    """
    A container for data of the yy (east-east) component of the gravity
    gradient tensor.

    Coordinate system used: x->North y->East z->Down

    Parameters:

    * x, y, z : 1D arrays
        Arrays with the x, y, z coordinates of the data points

    * data : 1D array
        The values of the data at the observation points

    * weight : float or array
        The weight of this data set in the misfit function. Pass an array to
        give weights to each data points or a float to weight the entire misfit
        function. See function :func:`~fatiando.gravmag.harvester.weights`

    """

    def __init__(self, x, y, z, data, weights=1., meshtype='prism'):
        Potential.__init__(self, x, y, z, data, weights, meshtype)
        self.effectfunc = self.engine.gyy


class Gyz(Potential):

    """
    A container for data of the yz (east-vertical) component of the gravity
    gradient tensor.

    Coordinate system used: x->North y->East z->Down

    Parameters:

    * x, y, z : 1D arrays
        Arrays with the x, y, z coordinates of the data points

    * data : 1D array
        The values of the data at the observation points

    * weight : float or array
        The weight of this data set in the misfit function. Pass an array to
        give weights to each data points or a float to weight the entire misfit
        function. See function :func:`~fatiando.gravmag.harvester.weights`

    """

    def __init__(self, x, y, z, data, weights=1., meshtype='prism'):
        Potential.__init__(self, x, y, z, data, weights, meshtype)
        self.effectfunc = self.engine.gyz


class Gzz(Potential):

    """
    A container for data of the zz (vertical-vertical) component of the gravity
    gradient tensor.

    Coordinate system used: x->North y->East z->Down

    Parameters:

    * x, y, z : 1D arrays
        Arrays with the x, y, z coordinates of the data points

    * data : 1D array
        The values of the data at the observation points

    * weight : float or array
        The weight of this data set in the misfit function. Pass an array to
        give weights to each data points or a float to weight the entire misfit
        function. See function :func:`~fatiando.gravmag.harvester.weights`

    """

    def __init__(self, x, y, z, data, weights=1., meshtype='prism'):
        Potential.__init__(self, x, y, z, data, weights, meshtype)
        self.effectfunc = self.engine.gzz


class TotalField(Potential):

    """
    A container for data of the total field magnetic anomaly.

    Coordinate system used: x->North y->East z->Down

    Parameters:

    * x, y, z : 1D arrays
        Arrays with the x, y, z coordinates of the data points

    * data : 1D array
        The values of the data at the observation points

    * inc, dec : floats
        The inclination and declination of the inducing field

    * weight : float or array
        The weight of this data set in the misfit function. Pass an array to
        give weights to each data points or a float to weight the entire misfit
        function. See function :func:`~fatiando.gravmag.harvester.weights`

    """

    def __init__(self, x, y, z, data, inc, dec, weights=1., meshtype='prism'):
        if meshtype != 'prism':
            raise AttributeError(
                "Unsupported mesh type '%s' for total field anomaly."
                % (meshtype))
        Potential.__init__(self, x, y, z, data, weights, meshtype)
        self.effectfunc = self.engine.tf
        self.prop = 'magnetization'
        self.inc = inc
        self.dec = dec

    def effect(self, prism, props):
        if self.prop not in props:
            return numpy.zeros(self.size, dtype='f')
        return self.effectfunc(self.x, self.y, self.z, [prism], self.inc,
                               self.dec, pmag=props[self.prop])
