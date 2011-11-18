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
3D iterative inversion using a planting algorithm.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 17-Nov-2010'

import time

import numpy

from fatiando import potential, utils, logger

log = logger.dummy()

class DataModule(object):
    """
    Mother of all data modules.

    Parameters:
    * norm
        Order of the norm of the residual vector to use (Default: 2). Can be:
        * 1 -> l1 norm
        * 2 -> l2 norm
        * etc
            
    """

    def __init__(self, norm=2):
        self.norm = norm

    def misfit(self, residuals):
        """
        Return the data misfit given a residual vector.
        
        Uses the prespecified norm.

        Parameters:
        * residuals
            Array with the residuals calculated on each data point

        Returns:
        * float
            The data misfit
            
        """
        return numpy.linalg.norm(residuals, self.norm)

    def residuals(self, mesh, estimate, neighbor):
        """
        Calculate the residuals vector due to adding a neighbor to the estimate.

        Parameters:
        * mesh
            A 3D mesh. See :mod:`fatiando.mesher.volume`
        * estimate
            Dictionary with the physical properties of the current estimate
            without the neighbor
        * neighbor
            [n, props]
            n is the index of the neighbor in the mesh.
            props is a dictionary with the physical properties of the neighbor

        Returns:
        * array
            Array with the residuals
            
        """
        raise NotImplementedError("'residuals' was not implemented")

    def cleanup(self, neighbor):
        """
        Delete unnecessary things after neighbor has been added to the estimate.

        Parameters:
        * neighbor
            [n, props]
            n is the index of the neighbor in the mesh.
            props is a dictionary with the physical properties of the neighbor
        
        """
        pass
                
class PotentialModule(DataModule):
    """
    Mother class for potential field data modules

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
    
    """

    def __init__(self, x, y, z, obs, norm=2):
        DataModule.__init__(self, norm)
        self.x = x
        self.y = y
        self.z = z
        self.res = numpy.array(obs)
        self.obs = obs
        self.effect = {}

    def calc_effect(self, prop, x1, x2, y1, y2, z1, z2, x, y, z):
        """
        Calculate the effect of the cell with a physical property prop.
        """
        raise NotImplementedError("Method 'calc_col' was not implemented")        

    def residuals(self, mesh, estimate, neighbor):
        """
        Calculate the residuals vector due to adding a neighbor to the estimate.

        Parameters:
        * mesh
            A 3D mesh. See :mod:`fatiando.mesher.volume`
        * estimate
            Dictionary with the physical properties of the current estimate
            without the neighbor. *IGNORED*
        * neighbor
            [n, props]
            n is the index of the neighbor in the mesh.
            props is a dictionary with the physical properties of the neighbor

        Returns:
        * array
            Array with the residuals
            
        """
        n, props = neighbor
        if self.prop not in props:
            return self.res
        prop = float(props[self.prop])
        if n not in self.effect:
            c = mesh[n]
            self.effect[n] = self.calc_effect(prop, c['x1'], c['x2'], c['y1'],
                c['y2'], c['z1'], c['z2'], self.x, self.y, self.z)  
        return self.res - self.effect[n]    
        
    def cleanup(self, neighbor):
        """
        Delete unnecessary things after neighbor has been added to the estimate.

        Parameters:
        * neighbor
            [n, props]
            n is the index of the neighbor in the mesh.
            props is a dictionary with the physical properties of the neighbor
            
        """
        self.res = self.res - self.effect[neighbor[0]]
        del self.effect[neighbor[0]]

    def predicted(self):
        """
        The predicted data at the end of an inversion process.
        Calculated by:
            pred = obs - residuals

        Returns:
        * array
            Array of predicted data.
            
        """
        return self.obs - self.res
        
class GzModule(PotentialModule):
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
    
    """

    def __init__(self, x, y, z, obs, norm=2):
        PotentialModule.__init__(self, x, y, z, obs, norm)
        self.prop = 'density'
        self.calc_effect = potential._prism.prism_gz
        
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
        A list the index of each seed in the mesh and their physical properties,
        as required by :func:`fatiando.inversion.harvester.harvest`
        
    """
    seeds = []
    append = seeds.append
    for point, props in rawseeds:
        x, y, z = point
        for s, cell in enumerate(mesh):
            if (x >= cell['x1'] and x <= cell['x2'] and y >= cell['y1'] and  
                y <= cell['y2'] and z >= cell['z1'] and z <= cell['z2']):
                append((s, props))
                break
    # Search for duplicates
    duplicates = []
    for i in xrange(len(seeds)):
        si, pi = seeds[i]
        for j in xrange(i+1, len(seeds)):
            sj, pj = seeds[j]
            if si == sj and True in [p in pi for p in pj]:
                duplicates.append((i,j))
    if duplicates:
        guilty = ', '.join(['%d and %d' % (i, j) for i, j in duplicates])
        msg1 = "Can't have seeds with same location and physical properties!"
        msg2 = "Guilty seeds: %s" % (guilty)
        raise ValueError, ' '.join([msg1, msg2])    
    return seeds

def find_neighbors(neighbor, mesh, full=False):
    """
    Return all neighbors of neighbor, even if marked already.
    """
    return []

def free(estimate, neighbors):
    """
    Remove neighbors that are marked from the list.
    """
    return []

def compact(estimate, mesh, neighbors):
    """
    Remove neighbors that don't satisfy the compactness criterion.
    """
    return []

    
    
def grow(seeds, mesh, dmods, compact=0, thresh=0.0001):
    """
    Yield one accretion at a time
    """
    # Define the goal and regularizing functions    
    def regularizer(est, mesh, n, mu=compact, seeds=seeds):
        pass
    def goalfunc(est, mesh=mesh, dmods=dmods, n):
        """
        Calculate the goal function for the estimate including the neighbor.
        """
        mft = sum(dm.misfit(dm.residuals(mesh, estimate, neighbor)) for dm in dmods)
    
    # Initialize the estimate with SparseLists
    estimate = {}
    for s, props in seeds:
        for p in props:
            if p not in estimate:
                estimate[p] = utils.SparseList(mesh.size)
    # Include the seeds in the estimate
    goal = 0
    for seed in seeds:
        goal = 0
        for dm in dmods:
            goal += dm.misfit(dm.residuals(mesh, estimate, seed))
            dm.cleanup(seed)
        s, props = seed
        for p in props:
            estimate[p][s] = props[p]
    # Find the neighboring prisms of the seeds
    neighbors = []
    for seed in seeds:
        neighbors.extend(find_neighbors(seed, mesh, full=True))
    neighbors = free(estimate, neighbors)
    # Spit out a changeset
    yield {'estimate':estimate, 'neighbors':neighbors, 'goal':goal,
           'dmods':dmods}
    # Perform the accretions until goal changes are bellow thresh. The maximum
    # number of accretions is the whole mesh minus seeds.
    for iteration in xrange(mesh.size - len(seeds)):
        
        goals = [goalfunc(estimate, n) for n in neighbors]
        bestgoal = 
                
        

def harvest(seeds, mesh, dmods, compact=0, thresh=0.0001):
    """
    Perform all accretions. Return mesh with special props in it.
    special props don't store 0s, return 0 if try to access element not in dict
    """
    tstart = time.clock()
    for i, changeset in enumerate(grow(seeds, mesh, dmods, compact, thresh)):
        continue
    tfinish = time.clock() - tstart
    #log.info("Total time for inversion: %s" % (utils.sec2hms(tfinish)))
    log.info("Total time for inversion: %s" % (str(tfinish)))
    log.info("Total number of accretions: %d" % (i + 1))
    #log.info("Average time per accretion: %s" % (utils.sec2hms(tfinish/i + 1)))
    log.info("Final goal function value: %g" % (changeset['goal']))
    return changeset['estimate']  


