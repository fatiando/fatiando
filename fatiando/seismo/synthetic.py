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
Generate synthetic seismological data, such as travel times and seismograms.

Functions:

* :func:`fatiando.seismo.synthetic.vel_from_image`
    Create a 2D velocity model from an image file (uses PIL)

* :func:`fatiando.seismo.synthetic.shoot_cartesian_straight`
    Shoot straight rays along a 2D velocity model

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 11-Sep-2010'


import logging

from PIL import Image

import numpy
import scipy.misc

import fatiando
import fatiando.seismo.traveltime


# Add the default handler (a null handler) to the logger to ensure that
# it won't print verbose if the program calling them doesn't want it
log = logging.getLogger('fatiando.seismo.synthetic')
log.addHandler(fatiando.default_log_handler)


def vel_from_image(fname, vmax, vmin):
    """
    Create a 2D velocity model from an image file. 
    
    The image is first converted to gray scale and then set to the desired 
    velocity interval.
    
    Parameters:
    
    * fname
        Name of the image file
    
    * vmax, vmin
        Value range of the velocities (used to convert the gray scale to 
        velocity values)
                  
    Returns:
    
    * 2D array-like velocity model
    
    """
            
    log.info("Creating velocity model from image file '%s'" % (fname))
        
    image = Image.open(fname)
    
    imagearray = scipy.misc.fromimage(image, flatten=True)
    
    # Invert the color scale
    model = numpy.max(imagearray) - imagearray
    
    # Normalize
    model = model/numpy.max(imagearray)
    
    # Put it in the interval [vmin,vmax]
    model = model*(vmax - vmin) + vmin
    
    # Convert the model to a list so that I can reverse it (otherwise the
    # image will be upside down)
    model = model.tolist()
    model.reverse()    
    model = numpy.array(model)
    
    nx, ny = model.shape
    
    log.info("  model size: nx=%d ny=%d" % (nx, ny))
    
    return model
    
    
def shoot_cartesian_straight(model, src_n, rec_n, type='circle', rec_span=45.):
    """
    Shoot straight rays through a 2D Cartesian velocity model.
    
    Parameters:
        
    * model
        2D array-like (matrix) Cartesian velocity model
    
    * src_n
        Number of ray sources
    
    * rec_n
        Number of receivers
    
    * type
        Source and receiver configuration. Can be any one of:
        
        * ``'circle'``
            Random sources and circular array of receivers around center of the 
            model
            
        * ``'xray'``
            X-ray array. One source and some receivers rotating around the 
            center of the figure. In this case, *src_n* is the number different 
            angles the array is placed in and *rec_n* is the number of receivers 
            per source.
            
        * ``'rand'``
            Both receivers and sources are randomly distributed
          
    * rec_span
        Angular spread of the receivers in the xray type configuration in 
        decimal degrees
                            
    Return:
    
    * dictionary with travel time data
        
    The data dictionary must be such as::
         
        {'src':[(x1,y1), (x2,y2), ...], 'rec':[(x1,y1), (x2,y2), ...], 
         'traveltime':[time1, time2, ...], 'error':[error1, error2, ...]}
         
    A source and receiver location is given for every travel time.

    Note: ``'error'`` is the standard deviation of each travel time
    """
    
    types = ['circle', 'xray', 'rand']
    
    assert type in types, "Invalid source and receiver configuration type"
    
    assert src_n > 0 and rec_n > 0, \
        "Source and receiver number must be positive"
    
    log.info("Shooting straight rays through 2D Cartesian velocity model")
    
    data = {'src':[], 'rec':[]}
        
    sizey, sizex = numpy.shape(model)    
    smallest = min(sizey, sizex)
    
    # Create the source and receiver pairs    
    if type == 'circle':
        
        rand_src_x = numpy.random.random(src_n)*sizex
        rand_src_y = numpy.random.random(src_n)*sizey   
        radius = numpy.random.normal(0.48*smallest, 0.02*smallest, rec_n)        
        angle = numpy.random.random(rec_n)*2*numpy.pi                
        rand_rec_x = 0.5*sizex + radius*numpy.cos(angle)       
        rand_rec_y = 0.5*sizey + radius*numpy.sin(angle)
        
        # Connect each source with all the receivers      
        for i in xrange(src_n):
            
            src = (rand_src_x[i], rand_src_y[i])
            
            for j in xrange(rec_n):
                
                data['src'].append(src)        
                data['rec'].append((rand_rec_x[j], rand_rec_y[j]))
                
    elif type == 'xray':
        
        rec_range = rec_span*numpy.pi/180.
        rec_step = rec_range/(rec_n - 1)
        
        radius = 1.5*smallest
        angle_step = 2*numpy.pi/(src_n)
        # Not starting in zero to avoid having horizontal or vertical rays
        # They cause a problem with the direct model (Ray intercepting cell in
        # more than 2 points error)
        src_angles = numpy.arange(10**(-5), 2*numpy.pi, angle_step)
        
        # Connect each source with all the receivers
        for src_angle in src_angles:
            
            src_x = 0.5*sizex + radius*numpy.cos(src_angle)
            src_y = 0.5*sizey + radius*numpy.sin(src_angle)
                
            start = src_angle + numpy.pi - 0.5*rec_range
            end = src_angle + numpy.pi + 0.5*rec_range
                        
            rec_angles = numpy.arange(start, end + rec_step, rec_step)
                        
            for i, rec_angle in enumerate(rec_angles):
            
                # Sometimes the rec_angles have an extra angle due to rounding
                # and floating point representation. This limits the number of 
                # receivers
                if i >= rec_n:
                
                    break
                
                rec_x = 0.5*sizex + radius*numpy.cos(rec_angle)
                rec_y = 0.5*sizey + radius*numpy.sin(rec_angle)                
                
                data['src'].append((src_x, src_y))
                data['rec'].append((rec_x, rec_y))
                
    elif type == 'rand':
                
        rand_src_x = numpy.random.random(src_n)*sizex
        rand_src_y = numpy.random.random(src_n)*sizey
        rand_rec_x = numpy.random.random(rec_n)*sizex
        rand_rec_y = numpy.random.random(rec_n)*sizey
        
        # Connect each source with all the receivers
        for i in range(src_n):
                        
            src = (rand_src_x[i], rand_src_y[i])
            
            for j in range(rec_n):
        
                data['src'].append(src)        
                data['rec'].append((rand_rec_x[j], rand_rec_y[j]))
                
    assert len(data['src']) == len(data['rec']), \
        "Error shooting rays. Different size of data['src'] and  data['rec']"
                
    # Now shoot the rays
    ntraveltimes = len(data['src'])
    data['traveltime'] = numpy.zeros(ntraveltimes)
    
    for l in xrange(ntraveltimes):
        
        src_x, src_y = data['src'][l]
        rec_x, rec_y = data['rec'][l]
        
        # Sum the time spent by the ray inside each cell
        for i in xrange(sizey):
            
            for j in xrange(sizex):
                
                time = fatiando.seismo.traveltime.cartesian_straight(
                                1./model[i][j], \
                                j, i, j + 1, i + 1, \
                                src_x, src_y, rec_x, rec_y)
                
                data['traveltime'][l] += time
                
    data['error'] = numpy.zeros(ntraveltimes)
                
    log.info("  rays shot=%d" % (ntraveltimes))
                
    return data