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
import math

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
    * weight
        Wether of not to use a weighing factor for this data type. The weight
        if the norm of the observed data.
            
    """

    def __init__(self, obs, norm=2, weight=True):
        self.norm = norm
        self.res = numpy.array(obs)
        self.obs = obs
        self.weight = 1.
        if weight:
            self.weight = 1./self.misfit(self.res)

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
        return self.weight*numpy.linalg.norm(residuals, self.norm)

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
    * weight
        Wether of not to use a weighing factor for this data type. The weight
        if the norm of the observed data.
    
    """

    def __init__(self, x, y, z, obs, norm=2, weight=True):
        DataModule.__init__(self, obs, norm, weight)
        self.x = x
        self.y = y
        self.z = z
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
    * weight
        Wether of not to use a weighing factor for this data type. The weight
        if the norm of the observed data.
    
    """

    def __init__(self, x, y, z, obs, norm=2, weight=True):
        PotentialModule.__init__(self, x, y, z, obs, norm, weight)
        self.prop = 'density'
        self.calc_effect = potential._prism.prism_gz

class ConcentrationRegularizer(object):
    """
    The mass concentration regularizer.
    Use it to force the estimated bodies to concentrate around the seeds.

    Parameters:
    * mu
        The regularing parameter. Controls the tradeoff between fitting the data
        and regularization.
    * seeds
        List of seeds as output by :func:`fatiando.inversion.harvester.sow`
    * mesh
        A 3D mesh. See :mod:`fatiando.mesher.volume`       
        
    """

    def __init__(self, seeds, mesh, mu=10**(-4), power=3, weight=True):
        self.mu = mu
        self.power = power
        self.seeds = seeds
        self.mesh = mesh
        self.reg = 0
        self.dists = {}
        self.weight = 1.
        if weight:
            nz, ny, nx = mesh.shape
            dx, dy, dz = mesh.dims
            self.weight = 1./(max([nx*dx, ny*dy, nz*dz])**power)

    def calc_dist(self, cell1, cell2):
        """
        Calculate the distance between 2 cells
        """
        dx = abs(cell1['x1'] - cell2['x1'])
        dy = abs(cell1['y1'] - cell2['y1'])
        dz = abs(cell1['z1'] - cell2['z1'])        
        return math.sqrt(dx**2 + dy**2 + dx**2)

    def __call__(self, neighbor, s):
        """
        Evaluate the regularizer with the neighbor included in the estimate.

        Parameters:
        * neighbor
            [n, props]
            n is the index of the neighbor in the mesh.
            props is a dictionary with the physical properties of the neighbor
        * s
            Index of the seed which the neighbor is neighboring prism of.
            Ex: first seed: s=0; second: s=1; etc

        Returns:
        * float
            The value of the regularing function already multiplied by the
            regularizing parameter mu
            
        """
        n = neighbor[0]
        if n not in self.dists:
            self.dists[n] = self.calc_dist(self.mesh[n],
                                           self.mesh[self.seeds[s][0]])
        return self.reg + self.weight*self.mu*(self.dists[n]**self.power)

    def cleanup(self, neighbor):
        """
        Clean up things after adding the neighbor to the estimate.
        """
        n = neighbor[0]
        self.reg += self.weight*self.mu*(self.dists[n]**self.power)
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

def _full_neighbors(allpossible, mesh, props):
    """
    Return the diagonal neighbors.
    """
    above, bellow, front, back, left, right = allpossible
    nz, ny, nx = mesh.shape
    neighbors = []
    append = neighbors.append
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
    return [(i, props) for i in neighbors]
    
def find_neighbors(neighbor, mesh, full=False, up=True, down=True):
    """
    Return neighboring prisms of neighbor (that share a face).

    Parameters:
    * neighbor
        [n, props]
        n is the index of the neighbor in the mesh.
        props is a dictionary with the physical properties of the neighbor
    * mesh
        A 3D mesh. See :mod:`fatiando.mesher.volume`
    * full
        If True, return also the prisms on the diagonal

    Returns:
    * list
        List with the neighbors (in the same format as parameter *neighbor*)
    
    """
    nz, ny, nx = mesh.shape 
    n, props = neighbor
    above, bellow, front, back, left, right = [None]*6
    # The guy above
    tmp = n - nx*ny    
    if tmp > 0 and up:        
        above = tmp
    # The guy bellow
    tmp = n + nx*ny
    if tmp < mesh.size and down:
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
    allpossible = [above, bellow, front, back, left, right]
    neighbors = [(i, props) for i in allpossible if i is not None]
    if full:
        neighbors.extend(_full_neighbors(allpossible, mesh, props))
    return neighbors

def in_estimate(estimate, neighbor):
    """
    Check if the neighbor is already set (not 0) in any of the physical
    properties of the estimate.
    """
    n, props = neighbor
    for p in props:
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
    n, props = neighbor
    for neighbors in neighborhood:
        for tmp in neighbors:
            if n == tmp[0]:
                for p in props:
                    if p in tmp[1]:
                        return True
    return False

def not_neighbors(neighborhood, neighbors):
    """
    Remove the neighbors that are already in the neighborhood.
    """
    return [n for n in neighbors if not in_tha_hood(neighborhood, n)]

def is_compact(estimate, mesh, neighbor):
    """
    Check if this neighbor satifies the compactness criterion.
    """
    around = find_neighbors(neighbor, mesh, full=True)
    free = free_neighbors(estimate, around)
    return len(around) - len(free) >= 3
    
def compact_neighbors(estimate, mesh, neighbors):
    """
    Remove neighbors that don't satisfy the compactness criterion.
    """
    return [n for n in neighbors if is_compact(estimate, mesh, n)]

def is_eligible(residuals, tol, dmods):
    """
    Check is a neighbor is eligible for accretion based on the residuals it
    produces.
    The criterion is that the predicted data must not be larger than the
    observed data in absolute value.
    """
    for dm, res in zip(dmods, residuals):
        diff = abs(dm.obs) - abs(dm.obs - res)
        if True in [abs(d) >= tol for d in diff if d < 0]:
            return False
    return True

    
    
def choose_best(s, neighbors, goalfunc, goal, mesh, estimate, dmods, thresh,
                tol):
    """
    Find which of the neighbors is the best one for the accretion.

    Returns:
    * [j, goal]
        j is the index in the neighbor list of the best. goal is the goal
        function value resulting from adding the best to the estimate.
        
    """
    # Calculate the residuals of all data modules for each neighbor
    res = [[dm.residuals(mesh, estimate, n) for dm in dmods] for n in neighbors]
    # Choose the ones that are eligible for accretion based on their residuals
    # Need the index i to know what neighbor we're talking about
    eligible = [(i, r) for i, r in enumerate(res) if is_eligible(r, tol, dmods)]
    # Calculate the goal functions
    goals = [(i, goalfunc(r, neighbors[i], s)) for i, r in eligible]
    # Keep only the ones that decrease the goal function
    decreased = [(i, g) for i, g in goals
                 if g < goal and abs(g - goal)/goal >= thresh]
    if decreased:
        #TODO: what if there is a tie?
        # Choose the best neighbor (decreases the goal function most)
        j, goal = decreased[numpy.argmin([g for i, g in decreased])]
        return j, goal
    return None

def grow(seeds, mesh, dmods, regularizer=None, thresh=0.0001, tol=0.01):
    """
    Yield one accretion at a time
    """
    # Define the goal function
    def misfit(residuals, dmods=dmods):
        return sum(dm.misfit(res) for dm, res in zip(dmods, residuals))
    if regularizer is None:  
        def goalfunc(residuals, neighbor, s, mesh=mesh):
            return misfit(residuals)
    else:
        def goalfunc(residuals, neighbor, s, mesh=mesh):
            return misfit(residuals) + regularizer(neighbor, s)    
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
    neighborhood = []
    for s in seeds:
        neighborhood.append(not_neighbors(neighborhood,
                                free_neighbors(estimate,
                                    find_neighbors(s, mesh, full=False, up=False, down=False))))
    # Spit out a changeset
    yield {'estimate':estimate, 'neighborhood':neighborhood, 'goal':goal,
           'dmods':dmods}
    # Perform the accretions. The maximum number of accretions is the whole mesh
    # minus seeds. The goal function starts with the total misfit of the seeds.
    for iteration in xrange(mesh.size - len(seeds)):
        onegrew = False
        for s, neighbors in enumerate(neighborhood):
            chosen = choose_best(s, neighbors, goalfunc, goal, mesh, estimate,
                                 dmods, thresh, tol)
            if chosen is not None:                
                onegrew = True
                j, goal = chosen
                best = neighbors[j]
                nbest, pbest = best
                # Add it to the estimate
                for p in pbest:
                    estimate[p][nbest] = pbest[p]                
                # Find its neighbors and append them
                neighbors.pop(j)
                neighbors.extend(compact_neighbors(estimate, mesh,
                                    not_neighbors(neighborhood,
                                        free_neighbors(estimate, 
                                            find_neighbors(best, mesh, up=False, down=False)))))
                # Clean up after adding the neighbor
                for dm in dmods:
                    dm.cleanup(best)
                if regularizer is not None:
                    regularizer.cleanup(best)
                # Spit out a changeset
                yield {'estimate':estimate, 'neighborhood':neighborhood,
                       'goal':goal, 'dmods':dmods}
        if not onegrew:
            break
    log.info("Final goal function value: %g" % (goal))    

def harvest(seeds, mesh, dmods, compactness=0, thresh=0.0001, tol=0.01):
    """
    Perform all accretions. Return mesh with special props in it.
    special props don't store 0s, return 0 if try to access element not in dict
    """
    tstart = time.clock()
    grower = grow(seeds, mesh, dmods, compactness, thresh, tol)
    goals = [chset['goal'] for i, chset in enumerate(grower)]
    tfinish = time.clock() - tstart
    log.info("Total time for inversion: %s" % (utils.sec2hms(tfinish)))
    log.info("Total number of accretions: %d" % (i))
    if i > 0:
        log.info("Average time per accretion: %s" % (utils.sec2hms(tfinish/i)))
    return chset, goals
