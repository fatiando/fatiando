# Copyright 2010 The Fatiando a Terra Development Team
#
# This file is part of Fatiando a Terra.
#
# Fatiando a Terra is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Fatiando a Terra is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Fatiando a Terra.  If not, see <http://www.gnu.org/licenses/>.
"""
Robust 3D potential field inversion by planting anomalous densities.

Performs the inversion by iteratively growing the estimate around user-specified
"seeds". Supports various kinds of data (gravity, gravity tensor, magnetic) and
there are plans to support different kinds of seeds as well.

The inversion is performed by function
:func:`fatiando.potential.harvester.harvest`. The required information, such as
observed data, seeds, and regularization, are passed to the function though
seed classes and data modules.

**SEEDS**

A seed class determines what kind of geometric element is used to parametrize
the anomalous density distribution. For example, if you use a SeedPrism, the
output of :func:`fatiando.potential.harvester.harvest` will be a list of prisms
that make up the estimated density distribution.

* :class:`fatiando.potential.harvester.SeedPrism`

Coming soon:

* :class:`fatiando.potential.harvester.SeedTesseroid`
* :class:`fatiando.potential.harvester.SeedPolyhedron`

**DATA MODULES**

Data modules wrap the observed data and calculate the predicted data for a given
parametrization. Data modules should match the type of seed used! Trying to
invert using SeedTesseroid and  DMPrismGz will cause errors.

* :class:`fatiando.potential.harvester.DMPrismGz`
* :class:`fatiando.potential.harvester.DMPrismGxx`
* :class:`fatiando.potential.harvester.DMPrismGxy`
* :class:`fatiando.potential.harvester.DMPrismGxz`
* :class:`fatiando.potential.harvester.DMPrismGyy`
* :class:`fatiando.potential.harvester.DMPrismGyz`
* :class:`fatiando.potential.harvester.DMPrismGzz`

**USAGE**

The standard usage of :mod:`fatiando.potential.harvester` is::

    import fatiando.mesher.ddd
    # Create a mesh
    # Assuming that bounds has the limits of the mesh and shape has the number
    # of prisms in each dimension
    bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
    shape = (nz, ny, nx)
    mesh = fatiando.mesher.ddd.PrismMesh(bounds, shape)
    # Make the data modules
    # Assuming the gravity data has been loaded into arrays xp, yp, zp and gz
    dms = wrapdata(mesh, xp, yp, zp, gz=gz)
    # Make the seed and set the compactness regularizing parameter mu
    seeds = sow(points=[[x, y, z]], props=[800], name='density', mu=0.1)
    # Run the inversion
    estimate, goals, misfits = harvest(dms, seeds)
    # The estimate is a list with the density of each prism in mesh
    # goals is a list of goal function value per iteration.
    # misfits is a list of the data misfit value per iteration.

----

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 17-Nov-2010'

import time
import math
import bisect

from fatiando.potential import _prism
from fatiando import utils, logger

log = logger.dummy()


class SeedPrism(object):
    """
    A 3D right rectangular prism seed.

    One of the types of seed required by
    :func:`fatiando.potential.harvester.harvest`.

    Wraps the information about a seed. Also knows how to grow a seed and the
    estimate it produced.

    Parameters:

    * point
        ``(x, y, z)``: x, y, z coordinates of where you want to place the seed.
        The seed will be a prism of the mesh that has this point inside it.
    * props
        Dictionary with the physical properties assigned to the seed.
        Ex: ``props={'density':10, 'susceptibility':10000}``
    * mesh
        The model space mesh (or interpretative model). A
        :class:`fatiando.mesher.ddd.PrismMesh` or a list of
        :func:`fatiando.mesher.ddd.Prism`.
    * mu
        Compactness regularizing parameters. Positive scalar that measures the
        trade-off between fit and regularization. This applies only to this
        seeds contribution to the total regularizing function. This way you can
        assign different mus to different seeds.
    * delta
        Minimum percentage of change required in the goal function to perform
        an accretion. The smaller this is, the less the solution is able to grow
    * compact
        Wether or not to impose compactness algorithmically on the solution.
        If False, the compactness of the solution will depend on the value of
        *mu*.
    
    """

    def __init__(self, point, props, mesh, mu=0., delta=0.0001 compact=False):
        self.props = props
        self.mesh = mesh
        self.delta = delta
        if compact:
            self._judge = self._compact_judge
        else:
            self._judge = self._standard_judge
        index = self._get_index(point, mesh)
        self.seed = [index, props]
        self.estimate = {}
        for prop in props:
            estimate[prop] = [[index, props[prop]]]
        nz, ny, nx = mesh.shape
        dx, dy, dz = mesh.dims
        self.weight = 1./((sum([nx*dx, ny*dy, nz*dz])/3.))
        self.mu = mu*self.weight
        self.neighbors = []
        self.reg = 0
        self.distance = {}

    def initialize(self, seeds):
        """
        Initialize the neighbor list of this seed.

        Leaves out elements that are already neighbors of other seeds or that
        are the seeds.
        """
        pass

    def set_mu(self, mu):
        """
        Set the value of the regularizing parameter mu.
        """
        self.mu = self.weight*mu
        
    def set_delta(self, delta):
        """
        Set the value of the delta threshold.
        """
        self.delta = delta

    def _get_index(self, point, mesh):
        """
        Get the index of the prism in mesh that has point inside it.
        """
        pass

    def _update_neighbors(self, n, seeds):
        """
        Remove neighbor n from the list of neighbors and include its neighbors
        """
        pass

    def _standard_judge(self, goals, misfits, goal, misfit):
        """
        Choose the best neighbor using the following criteria:

        1. Must decrease the misfit
        2. Must produce the smallest goal function out of all that pass 1.        
        """
        pass

    def _compact_judge(self, goals, misfits, goal, misfit):
        """
        Choose the best neighbor using the following criteria:

        1. Must satisfy the compactness criterion
        2. Must decrease the goal function  
        """
        pass

    def grow(self, dms, seeds, goal, misfit):
        """
        Try to grow this seed by adding a prism to it's periphery.
        """
        misfits = [sum(dm.testdrive(n) for dm in dms) for n in self.neighbors]
        goals = [m + sum((s.reg for s in seeds), self.mu*self.distance[n[0]])
                 for n, m in zip(self.neighbors, misfits)]
        best = self._judge(goals, misfits, goal, misfit)
        if best is None:
            return None
        index = self.neighbors[best][0]
        for prop in self.props:
            estimate[prop].append([index, self.props[prop]])
        self._update_neighbors(best, seeds)
        return self.neighbors[best]
            


class DataModule(object):
    """
    Mother class for data modules

    Remember: the coordinate system is x->North, y->East, and z->Down

    Parameters:
    * x, y, z
        Arrays with the x, y, and z coordinates of the data points.
    * obs
        Array with the observed values of the component of the potential field.
    * norm
        Order of the norm of the residual vector to use (Default: 2). Can be:
        * 1 -> l1 norm
        * 2 -> l2 norm
        * etc
    * weight
        Wether of not to use a weighing factor for this data type. The weight
        if the norm of the observed data.
    * use_shape
		Wether to use the Shape-of-Anomaly criterion instead of regular
		residuals
    
    """

    def __init__(self, x, y, z, obs, norm, weight):
        self.norm = norm
        self.obs = obs
        self.x = x
        self.y = y
        self.z = z
        self.effect = {}
        self.predicted = numpy.zeros_like(self.obs)
        self.l2obs = numpy.linalg.norm(obs, 2)**2
        self.absobs = numpy.abs(obs)
        self.obsmax = self.absobs.max()
        if weight is None:
            self.weight = 1./numpy.linalg.norm(obs, norm)
        else:
            self.weight = weight

    def calc_effect(self, prop, x1, x2, y1, y2, z1, z2, x, y, z):
        """
        Calculate the effect of the cell with a physical property prop.
        """
        raise NotImplementedError("Method 'calc_col' was not implemented")
		
    def new_predicted(self, neighbor, mesh):
        """
        Calculate the predicted data vector due to adding the neighbor to the
        estimate.
        
        Parameters:
        * neighbor
            [n, props]
            n is the index of the neighbor in the mesh.
            props is a dictionary with the physical properties of the neighbor
        * mesh
            A 3D mesh. See :mod:`fatiando.mesher.volume`
        
        Returns:
        * array
            Array with the predicted data
                    
        """
        n, props = neighbor['index'], neighbor['props']
        if self.prop not in props:
            return self.predicted
        if n not in self.effect:
            c = mesh[n]
            self.effect[n] = self.calc_effect(float(props[self.prop]),
                c['x1'], c['x2'], c['y1'], c['y2'], c['z1'], c['z2'],
                self.x, self.y, self.z)
        return self.predicted + self.effect[n]

    def residuals(self, predicted):
        """
        Calculate the residuals vector.

        Parameters:
        * predicted
			Array with the predicted data

        Returns:
        * array
            Array with the residuals
           
        """
        return self.obs - predicted

    def misfit(self, predicted):
        """
        Return the data misfit given a predicted data vector.
        
        Uses the prespecified norm.

        Parameters:
        * predicted
            Array with the predicted data calculated on each data point

        Returns:
        * float
            The data misfit
            
        """
        return self.weight*numpy.linalg.norm(self.obs - predicted, self.norm)
        
    def shape_of_anomaly(self, predicted):
        """
        Calculate the shape-of-anomaly criterion.

        Parameters:
        * predicted
			Array with the predicted data

        Returns:
        * float
            
        """
        scale = numpy.sum(self.obs*predicted)/self.l2obs
        return numpy.linalg.norm(scale*self.obs - predicted, 2)
        
    def update(self, neighbor, mesh):
        """
        Update the predicted data vector and delete unnecessary things after
        neighbor has been added to the estimate.
    
        Parameters:
        * neighbor
            [n, props]
            n is the index of the neighbor in the mesh.
            props is a dictionary with the physical properties of the neighbor
        * mesh
            A 3D mesh. See :mod:`fatiando.mesher.volume`
            
        """
        n, props = neighbor['index'], neighbor['props']
        if self.prop not in props:
            pass
        if n not in self.effect:
            c = mesh[n]
            self.effect[n] = self.calc_effect(float(props[self.prop]),
                c['x1'], c['x2'], c['y1'], c['y2'], c['z1'], c['z2'],
                self.x, self.y, self.z)
        self.predicted += self.effect[n]
        del self.effect[n]
        
class PrismGzModule(DataModule):
    """
    Data module for the vertical component of the gravitational attraction (gz).

    Remember: the coordinate system is x->North, y->East, and z->Down

    Parameters:
    * x, y, z
        Arrays with the x, y, and z coordinates of the data points.
    * obs
        Array with the observed values of the gz component of the gravity field.
    * norm
        Order of the norm of the residual vector to use (Default: 2). Can be:
        * 1 -> l1 norm
        * 2 -> l2 norm
        * etc
    * weight
        The relative weight of this data module. Should be a positive number
        from 1 to 0.
    
    """

    def __init__(self, x, y, z, obs, norm=2, weight=None):
        DataModule.__init__(self, x, y, z, obs, norm, weight)
        self.prop = 'density'
        self.calc_effect = _prism.prism_gz
        
class PrismGxxModule(DataModule):
    """
    Data module for the gxx component of the gravity gradient tensor.

    Remember: the coordinate system is x->North, y->East, and z->Down

    Parameters:
    * x, y, z
        Arrays with the x, y, and z coordinates of the data points.
    * obs
        Array with the observed values of the component.
    * norm
        Order of the norm of the residual vector to use (Default: 2). Can be:
        * 1 -> l1 norm
        * 2 -> l2 norm
        * etc
    * weight
        The relative weight of this data module. Should be a positive number
        from 1 to 0.
    
    """

    def __init__(self, x, y, z, obs, norm=2, weight=None):
        DataModule.__init__(self, x, y, z, obs, norm, weight)
        self.prop = 'density'
        self.calc_effect = _prism.prism_gxx
        
class PrismGxyModule(DataModule):
    """
    Data module for the gxy component of the gravity gradient tensor.

    Remember: the coordinate system is x->North, y->East, and z->Down

    Parameters:
    * x, y, z
        Arrays with the x, y, and z coordinates of the data points.
    * obs
        Array with the observed values of the component.
    * norm
        Order of the norm of the residual vector to use (Default: 2). Can be:
        * 1 -> l1 norm
        * 2 -> l2 norm
        * etc
    * weight
        The relative weight of this data module. Should be a positive number
        from 1 to 0.
    
    """

    def __init__(self, x, y, z, obs, norm=2, weight=None):
        DataModule.__init__(self, x, y, z, obs, norm, weight)
        self.prop = 'density'
        self.calc_effect = _prism.prism_gxy
        
class PrismGxzModule(DataModule):
    """
    Data module for the gxz component of the gravity gradient tensor.

    Remember: the coordinate system is x->North, y->East, and z->Down

    Parameters:
    * x, y, z
        Arrays with the x, y, and z coordinates of the data points.
    * obs
        Array with the observed values of the component.
    * norm
        Order of the norm of the residual vector to use (Default: 2). Can be:
        * 1 -> l1 norm
        * 2 -> l2 norm
        * etc
    * weight
        The relative weight of this data module. Should be a positive number
        from 1 to 0.
    
    """

    def __init__(self, x, y, z, obs, norm=2, weight=None):
        DataModule.__init__(self, x, y, z, obs, norm, weight)
        self.prop = 'density'
        self.calc_effect = _prism.prism_gxz
        
class PrismGyyModule(DataModule):
    """
    Data module for the gyy component of the gravity gradient tensor.

    Remember: the coordinate system is x->North, y->East, and z->Down

    Parameters:
    * x, y, z
        Arrays with the x, y, and z coordinates of the data points.
    * obs
        Array with the observed values of the component.
    * norm
        Order of the norm of the residual vector to use (Default: 2). Can be:
        * 1 -> l1 norm
        * 2 -> l2 norm
        * etc
    * weight
        The relative weight of this data module. Should be a positive number
        from 1 to 0.
    
    """

    def __init__(self, x, y, z, obs, norm=2, weight=None):
        DataModule.__init__(self, x, y, z, obs, norm, weight)
        self.prop = 'density'
        self.calc_effect = _prism.prism_gyy
        
class PrismGyzModule(DataModule):
    """
    Data module for the gyz component of the gravity gradient tensor.

    Remember: the coordinate system is x->North, y->East, and z->Down

    Parameters:
    * x, y, z
        Arrays with the x, y, and z coordinates of the data points.
    * obs
        Array with the observed values of the component.
    * norm
        Order of the norm of the residual vector to use (Default: 2). Can be:
        * 1 -> l1 norm
        * 2 -> l2 norm
        * etc
    * weight
        The relative weight of this data module. Should be a positive number
        from 1 to 0.
    
    """

    def __init__(self, x, y, z, obs, norm=2, weight=None):
        DataModule.__init__(self, x, y, z, obs, norm, weight)
        self.prop = 'density'
        self.calc_effect = _prism.prism_gyz
        
class PrismGzzModule(DataModule):
    """
    Data module for the gzz component of the gravity gradient tensor.

    Remember: the coordinate system is x->North, y->East, and z->Down

    Parameters:
    * x, y, z
        Arrays with the x, y, and z coordinates of the data points.
    * obs
        Array with the observed values of the component.
    * norm
        Order of the norm of the residual vector to use (Default: 2). Can be:
        * 1 -> l1 norm
        * 2 -> l2 norm
        * etc
    * weight
        The relative weight of this data module. Should be a positive number
        from 1 to 0.
    
    """

    def __init__(self, x, y, z, obs, norm=2, weight=None):
        DataModule.__init__(self, x, y, z, obs, norm, weight)
        self.prop = 'density'
        self.calc_effect = _prism.prism_gzz

class ConcentrationRegularizer(object):
    """
    The mass concentration regularizer.
    Use it to force the estimated bodies to concentrate around the seeds.

    Parameters:
    * seeds
        List of seeds as output by :func:`fatiando.inversion.harvester.sow`
    * mesh
        A 3D mesh. See :mod:`fatiando.mesher.volume`   
    * mu
        The regularing parameter. Controls the tradeoff between fitting the data
        and regularization.
    * power
        Power to which the distances are raised. Usually between 3 and 7.
        
    """

    def __init__(self, seeds, mesh, mu=10**(-4), power=3, weight=True):
        self.mu = mu
        self.power = power
        self.seeds = seeds
        self.mesh = mesh
        self.reg = 0
        self.timeline = [0.]
        self.record = self.timeline.append
        self.dists = {}
        self.weight = 1.
        if weight:
            nz, ny, nx = mesh.shape
            dx, dy, dz = mesh.dims
            #self.weight = 1./((sum([nx*dx, ny*dy, nz*dz])/3.)**power)
            self.weight = 1./((sum([nx*dx, ny*dy, nz*dz])/3.))

    def calc_dist(self, cell1, cell2):
        """
        Calculate the distance between 2 cells
        """
        dx = abs(cell1['x1'] - cell2['x1'])
        dy = abs(cell1['y1'] - cell2['y1'])
        dz = abs(cell1['z1'] - cell2['z1'])        
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def __call__(self, neighbor, seed):
        """
        Evaluate the regularizer with the neighbor included in the estimate.

        Parameters:
        * neighbor
            [n, props]
            n is the index of the neighbor in the mesh.
            props is a dictionary with the physical properties of the neighbor
        * seed
            [s, props]
            s is the index of the seed in the mesh.
            props is a dictionary with the physical properties of the seed

        Returns:
        * float
            The value of the regularing function already multiplied by the
            regularizing parameter mu
            
        """
        n = neighbor['index']
        if n not in self.dists:
            s = seed['index']
            self.dists[n] = self.calc_dist(self.mesh[n], self.mesh[s])
        return self.reg + self.weight*self.mu*(self.dists[n]**self.power)

    def update(self, neighbor):
        """
        Clean up things after adding the neighbor to the estimate.
        """
        n = neighbor['index']
        self.reg += self.weight*self.mu*(self.dists[n]**self.power)
        self.record(self.reg)
        del self.dists[n]
                
def loadseeds(fname, prop):
    """
    Load a set of seed locations and physical properties from a file.

    The file should have 4 columns: x, y, z, value
    x, y, and z are the coordinates where the seed should be put. value is value
    of the physical property associated with the seed.

    Remember: the coordinate system is x->North, y->East, and z->Down

    Parameters:
    * fname
        Open file object or filename string
    * prop
        String with the name of the physical property. Ex: density

    Returns:
    * list
        A list with the position and physical property of the seeds, as required
        by :func:`fatiando.inversion.harvester.sow`::        
            [((x1,y1,z1), {prop:value1}), ((x2,y2,z2), {prop:value2}), ...] 
    
    """
    return [((x, y, z), {prop:v}) for x, y, z, v  in numpy.loadtxt(fname)]

def sow(mesh, rawseeds):
    """ 
    Find the index of the seeds in the mesh given their (x,y,z) location.

    The output of this function should be passed to
    :func:`fatiando.inversion.harvester.harvest`

    Parameters:
    * mesh
        A 3D mesh. See :mod:`fatiando.mesher.volume`
    * rawseeds
        A list with the position and physical property of each seed::        
            [((x1,y1,z1), {'density':v1}), ((x2,y2,z2), {'density':v2}), ...] 

    Returns:
    * list
        A list seeds as required by :func:`fatiando.inversion.harvester.harvest`
        
    """
    log.info("Sowing seeds in the mesh:")
    tstart = time.clock()
    seeds = []
    append = seeds.append
    # This is a quick hack to get the xs, ys, and zs.
    # TODO: make PrismMesh have get_xs, etc, methods
    x1, x2, y1, y2, z1, z2 = mesh.bounds
    dx, dy, dz = mesh.dims
    nz, ny, nx = mesh.shape
    xs = numpy.arange(x1, x2, dx)
    ys = numpy.arange(y1, y2, dy)
    zs = numpy.arange(z1, z2, dz)
    for point, props in rawseeds:
        x, y, z = point
        found = False
        if x <= x2 and x >= x1 and y <= y2 and y >= y1 and z <= z2 and z >= z1:
            # -1 because bisect gives the index z would have. I want to know
            # what index z comes after
            k = bisect.bisect_left(zs, z) - 1
            j = bisect.bisect_left(ys, y) - 1
            i = bisect.bisect_left(xs, x) - 1
            s = i + j*nx + k*nx*ny
            if mesh[s] is not None:
                found = True
                append({'index':s, 'props':props})
        if not found:
            raise ValueError, "Couldn't find seed at location %s" % (str(point))
    # Search for duplicates
    duplicates = []
    for i in xrange(len(seeds)):
        si, pi = seeds[i]['index'], seeds[i]['props']
        for j in xrange(i + 1, len(seeds)):
            sj, pj = seeds[j]['index'], seeds[j]['props']
            if si == sj and True in [p in pi for p in pj]:
                duplicates.append((i, j))
    if duplicates:
        guilty = ', '.join(['%d and %d' % (i, j) for i, j in duplicates])
        msg1 = "Can't have seeds with same location and physical properties!"
        msg2 = "Guilty seeds: %s" % (guilty)
        raise ValueError, ' '.join([msg1, msg2])
    log.info("  found %d seeds" % (len(seeds)))
    tfinish = time.clock() - tstart
    log.info("  time: %s" % (utils.sec2hms(tfinish)))
    return seeds
    
def find_neighbors(neighbor, mesh, full=False, up=True, down=True):
    """
    Return neighboring prisms of neighbor (that share a face).

    Parameters:
    * neighbor
        Dictionary with keys:
        'index': the index of the neighbor in the mesh.
        'props': a dictionary with the physical properties of the neighbor
    * mesh
        A 3D mesh. See :mod:`fatiando.mesher.volume`
    * full
        If True, return also the prisms on the diagonal

    Returns:
    * list
        List with the neighbors (in the same format as parameter *neighbor*)
    
    """
    nz, ny, nx = mesh.shape 
    n, props = neighbor['index'], neighbor['props']
    above, bellow, front, back, left, right = [None]*6
    # The guy above
    tmp = n - nx*ny    
    if up and tmp > 0:        
        above = tmp
    # The guy bellow
    tmp = n + nx*ny
    if down and tmp < mesh.size:
        bellow = tmp    
    # The guy in front
    tmp = n + 1
    if n%nx < nx - 1:
        front = tmp
    # The guy in the back
    tmp = n - 1
    if n%nx != 0:
        back = tmp
    # The guy to the left
    tmp = n + nx
    if n%(nx*ny) < nx*(ny - 1):
        left = tmp
    # The guy to the right
    tmp = n - nx
    if n%(nx*ny) >= nx:
        right = tmp
    indexes = [above, bellow, front, back, left, right]
    # The diagonal neighbors
    if full:
        append = indexes.append
        if front is not None and left is not None:        
            append(left + 1)    
        if front is not None and right is not None:        
            append(right + 1)
        if back is not None and left is not None:
            append(left - 1)
        if back is not None and right is not None:
            append(right - 1)
        if above is not None and left is not None:
            append(above + nx)
        if above is not None and right is not None:
            append(above - nx)
        if above is not None and front is not None:
            append(above + 1)
        if above is not None and back is not None:
            append(above - 1)
        if above is not None and front is not None and left is not None:
            append(above + nx + 1)
        if above is not None and front is not None and right is not None:
            append(above - nx + 1)
        if above is not None and back is not None and left is not None:
            append(above + nx - 1)
        if above is not None and back is not None and right is not None:
            append(above - nx - 1)
        if bellow is not None and left is not None:
            append(bellow + nx)
        if bellow is not None and right is not None:
            append(bellow - nx)
        if bellow is not None and front is not None:
            append(bellow + 1)
        if bellow is not None and back is not None:
            append(bellow - 1)
        if bellow is not None and front is not None and left is not None:
            append(bellow + nx + 1)
        if bellow is not None and front is not None and right is not None:
            append(bellow - nx + 1)
        if bellow is not None and back is not None and left is not None:
            append(bellow + nx - 1)
        if bellow is not None and back is not None and right is not None:
            append(bellow - nx - 1)
    # Filter out the ones that do not exist or are masked
    neighbors = [{'index':i, 'props':props} for i in indexes if i is not None
                 and mesh[i] is not None]
    return neighbors

def in_estimate(estimate, neighbor):
    """
    Check if the neighbor is already set (not 0) in any of the physical
    properties of the estimate.
    """
    n = neighbor['index']
    for p in neighbor['props']:
        if estimate[p][n] != 0:
            return True
    return False    
   
def free_neighbors(estimate, neighbors):
    """
    Remove neighbors that have their physical properties already set on the
    estimate.
    """    
    return [n for n in neighbors if not in_estimate(estimate, n)]

def in_tha_hood(neighborhood, neighbor):
    """
    Check if a neighbor is already in the neighborhood with the same physical
    properties.
    """
    n, props = neighbor['index'], neighbor['props']
    for neighbors in neighborhood:
        for tmp in neighbors:
            if n == tmp['index']:
                for p in props:
                    if p in tmp['props']:
                        return True
    return False

def not_neighbors(neighborhood, neighbors):
    """
    Remove the neighbors that are already in the neighborhood.
    """
    return [n for n in neighbors if not in_tha_hood(neighborhood, n)]

def is_compact(estimate, mesh, neighbor, compact):
    """
    Check if this neighbor satifies the compactness criterion.
    """
    around = neighbor['neighbors']
    free = free_neighbors(estimate, around)
    return len(around) - len(free) >= compact
    
def is_eligible(predicted, tol, dmods):
    """
    Check is a neighbor is eligible for accretion based on the residuals it
    produces.
    The criterion is that the predicted data must not be larger than the
    observed data in absolute value.
    """
    for dm, pred in zip(dmods, predicted):
        if True in (d < -tol for d in (dm.absobs - abs(pred))/dm.obsmax):
            return False
    return True

def standard_jury(regularizer=None, thresh=0.0001, tol=0.01):
    """
    Creates a standard jury function (neighbor chooser) based on regular data
    misfit and regularization.
    """
    def jury(seed, neighbors, estimate, datamods, misfit, mesh, it, nseeds,
             thresh=thresh, tol=tol, regularizer=regularizer):
        left = [(i, n) for i, n in enumerate(neighbors)]
        # Calculate the predicted data of the ones that are left
        pred = [[dm.new_predicted(n, mesh) for dm in datamods]
                for n in neighbors]
        # Filter the eligible for accretion based on their predicted data
        #left = [(i, n) for i, n in enumerate(neighbors)
                #if is_eligible(pred[i], tol, datamods)]
        misfits = ((i, sum(dm.misfit(p) for dm, p in zip(datamods, pred[i])))
                   for i, n in left)
        # Keep only the ones that decrease the data misfit function
        decreased = [(i, m) for i, m in misfits
                     if m < misfit and abs(m - misfit)/misfit >= thresh]
        if not decreased:
            return None
        # Calculate the goal functions
        if regularizer is not None:
            goals = [m + regularizer(neighbors[i], seed) for i, m in decreased]
        else:
            goals = [m for i, m in decreased]
        #TODO: what if there is a tie?
        # Choose the best neighbor (decreases the goal function most)
        best = decreased[numpy.argmin(goals)]
        if regularizer is not None:
            regularizer.update(neighbors[best[0]])
        return best
    return jury

def shape_jury(regularizer=None, thresh=0.0001, maxcmp=4, tol=0.01):
    """
    Creates a jury function (neighbor chooser) based on shape-of-anomaly data
    misfit, algorithmic compactness, and regularization.
    """    
    def jury(seed, neighbors, estimate, datamods, goal, mesh, it, nseeds,
             maxcmp=maxcmp, thresh=thresh, tol=tol, regularizer=regularizer):
        # Make the compactness criterion vary with iteration so that the first
        # neighbors are eligible (they only have the seed as neighbor)
        compact = 1 + it/nseeds
        if compact > maxcmp:
            compact = maxcmp
        # Filter the ones that don't satisfy the compactness criterion
        left = [(i, n) for i, n in enumerate(neighbors)
                if is_compact(estimate, mesh, n, compact)]
        # Calculate the predicted data of the ones that are left
        pred = dict((i, [dm.new_predicted(n, mesh) for dm in datamods])
                    for i, n in left)
        # Filter the eligible for accretion based on their predicted data
        #left = [(i, n) for i, n in left if is_eligible(pred[i], tol, datamods)]
        misfits = [(i, sum(dm.misfit(p) for dm, p in zip(datamods, pred[i])))
                   for i, n in left]
        # Calculate the goal function
        if regularizer is not None:
            reg = (regularizer(n, seed) for i, n in left)
            goals = [(m[0], m[1] + r) for m, r in zip(misfits, reg)]
        else:
            goals = misfits
        # Keep only the ones that decrease the goal function
        decreased = [(i, g) for i, g in goals
                     if g < goal and abs(g - goal)/goal >= thresh]
        if not decreased:
            return None
        # Find any holes
        #hole = find_holes(
        # Choose based on the shape-of-anomaly criterion
        soa = [sum(dm.shape_of_anomaly(p) for dm, p in zip(datamods, pred[i]))
               for i, g in decreased]
        #TODO: what if there is a tie?
        # Choose the best neighbor (decreases the goal function most)
        best = decreased[numpy.argmin(soa)]
        if regularizer is not None:
            regularizer.update(neighbors[best[0]])
        return best
    return jury

def grow(seeds, mesh, datamods, jury):
    """
    Yield one accretion at a time
    """
    # Initialize the estimate with SparseLists
    estimate = {}
    for seed in seeds:
        for p in seed['props']:
            if p not in estimate:
                estimate[p] = utils.SparseList(mesh.size)
    # Include the seeds in the estimate
    for seed in seeds:
        for p in seed['props']:
            estimate[p][seed['index']] = seed['props'][p]
        for dm in datamods:
			dm.update(seed, mesh)
    # Find the neighbors of the seeds
    neighborhood = []
    for seed in seeds:
        neighbors = not_neighbors(neighborhood,
                        free_neighbors(estimate, 
                            find_neighbors(seed, mesh)))
        for n in neighbors:
            n['neighbors'] = find_neighbors(n, mesh, full=True)
        neighborhood.append(neighbors)
	# Calculate the initial goal function
    goal = sum(dm.misfit(dm.predicted) for dm in datamods)
    # Spit out a changeset
    yield {'estimate':estimate, 'neighborhood':neighborhood, 'goal':goal,
           'datamods':datamods}
    # Perform the accretions. The maximum number of accretions is the whole mesh
    # minus seeds. The goal function starts with the total misfit of the seeds.
    nseeds = len(seeds)
    for iteration in xrange(mesh.size - nseeds):
        onegrew = False
        for seed, neighbors in zip(seeds, neighborhood):
            chosen = jury(seed, neighbors, estimate, datamods, goal, mesh,
                          iteration, nseeds)
            if chosen is not None:                
                onegrew = True
                j, goal = chosen
                best = neighbors[j]
                # Add it to the estimate
                for p in best['props']:
                    estimate[p][best['index']] = best['props'][p]
                for dm in datamods:
                    dm.update(best, mesh)
                # Update the neighbors of this neighborhood
                neighbors.pop(j)
                newneighbors = not_neighbors(neighborhood,
									free_neighbors(estimate,
                                        find_neighbors(best, mesh)))
                for n in newneighbors:
                    n['neighbors'] = find_neighbors(n, mesh, full=True)
                neighbors.extend(newneighbors)
                # Spit out a changeset
                yield {'estimate':estimate, 'neighborhood':neighborhood,
                       'goal':goal, 'datamods':datamods}
        if not onegrew:
            break 

def _cat_estimate(seeds):
    """
    Concatenate the estimate of all seeds to produce the final estimate.
    What estimate is depends on the kind of seed.
    """
    estimate = None
    kind = seeds[0].kind
    if kind == 'prism':
        estimate = {}
        for seed in seeds:
            for prop in seed.estimate:
                if prop not in estimate:
                    estimate[prop] = seed.estimate[prop]
                else:
                    estimate[prop].extend(seed.estimate[prop])
        size = seeds[0].mesh.size
        for prop in estimate:
            estimate[prop] = utils.SparseList(size, dict(estimate[prop]))        
    return estimate

def _harvest_iterator(dms, seeds, first_goal):
    """
    Iterator that yields the growth iterations of a 3D potential field inversion
    by planting anomalous densities.
    For more details on the parameters, see
    :func:`fatiando.potential.harvester.harvest`.

    Yields:

    * changeset
        A dictionary with keys:

        * 'estimate'
            The estimate at this growth iteration
        * 'goal'
            Goal function value at this growth iteration
        * 'misfit'
            Data misfit value at this growth iteration
            
    """
    pass
    
def _harvest_solver(dms, seeds, first_goal):
    """
    Solve a 3D potential field inversion by planting anomalous densities.
    For more details on the parameters and return values, see
    :func:`fatiando.potential.harvester.harvest`.           
    """
    goals = [first_goal]
    upgoal = goals.append
    # Since the compactness regularizing function is zero in the begining
    misfits = [first_goal]
    upmisfit = misfits.append
    goal = first_goal
    misfit = first_goal
    while True:
        grew = False
        for seed in seeds:
            change = seed.grow(dms, seeds, goal, misfit)
            if change is not None:
                grew = True
                params, goal, misfit = change
                upgoal(goal)
                upmisfit(misfit)
                for dm in dms:
                    dm.update(params)
        if not grew:
            break
    estimate = _cat_estimate(seeds)
    return [estimate, goals, misfits]

def harvest(dms, seeds, mu=None, shape=None, delta=None, iterate=False):
    """
    Robust 3D potential field inversion by planting anomalous densities.

    Performs the inversion on a data set by iteratively growing the given seeds
    until the observed data fit the predicted data (according to the misfit
    measure specified).

    Parameters:

    * dms
        List of data modules (see the docs of
        :mod:`fatiando.potential.harvester` for information on data modules)
    * seeds
        List of seeds (see the docs of :mod:`fatiando.potential.harvester` for
        information on seeds)
    * mu
        Compactness regularizing parameters. Positive scalar that measures the
        trade-off between fit and regularization. If None, will use the values
        passed to each seed. If not None, will overwrite the values passed to
        the seeds.
    * delta
        Minimum percentage of change required in the goal function to perform
        an accretion. The smaller this is, the less the solution is able to
        grow. If None, will use the values passed to each seed. If not None,
        will overwrite the values passed to the seeds.
    * shape
        Wether or not to use the shape-of-anomaly data misfit function of
        Rene (1986) instead of the conventional l1-norm data misfit. If None,
        will use the values passed to each seed. If not None, will overwrite the
        values passed to the seeds.
    * iterate
        If True, will return an iterator object that yields one growth iteration
        at a time.

    Returns:

    * [estimate, goals, misfits]
        goals is a list with the goal function value per iteration.
        misfits is a list with the data misfit value per iteration.
        The contents of *estimate* depend on the type of seed used:

        * Prisms
            A dictionary of physical properties. Each key is a physical property
            name, like ``'density'``, and each value is a list of values of
            that physical property for each element in the given model space
            mesh (interpretative model). Example::

                estimate = {'density':[1, 0, 6, 9, 7, 8, ...],
                            'susceptibility':[0, 4, 8, 3, 4, 5.4, ...]}

    References:

    Rene, R. M. (1986). Gravity inversion using open, reject, and
        "shape-of-anomaly" fill criteria. Geophysics, 51, 988.
        doi:10.1190/1.1442157
                
    """
    log.info("Harvesting inversion results from planting anomalous densities:")
    log.info("  regularizing parameter (mu): %g" % (mu))
    log.info("  delta (threshold): %g" % (delta))
    log.info("  shape-of-anomaly: %s" % (str(shape)))
    log.info("  iterate: %s" % (str(iterate)))
    # Make sure the seeds are all of the same kind. The .cound hack is from
    # stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-
    # identical
    kinds = [seed.kind for seed in seeds]
    if kinds.count(kinds[0]) != len(kinds):
        raise ValueError, "Seeds must all be of the same kind!"
    # Initialize the seeds and data modules before starting
    for i, seed in enumerate(seeds):
        if mu is not None:
            seed.set_mu(mu)
        if delta is not None:
            seed.set_delta(delta)
        seed.initialize(seeds)
    for dm in dms:
        if shape is not None and shape:
            dm.use_shape()
        for seed in seeds:
            dm.update(seed.seed)
    # Calculate the initial goal function
    goal = sum(dm.misfit(dm.predicted) for dm in dms)
    # Now run the actual inversion
    if iterate:
        return _harvest_iterator(dms, seeds, goal)
    else:
        tstart = time.clock()
        results = _harvest_solver(dms, seeds, goal)
        tfinish = time.clock() - tstart
        its = len(results[1])
        log.info("  Final goal function value: %g" % (results[1][-1]))   
        log.info("  Total number of accretions: %d" % (its))
        log.info("  Average time per accretion: %s" %
            (utils.sec2hms(float(tfinish)/its)))
        log.info("  Total time for inversion: %s" % (utils.sec2hms(tfinish)))
        return results
