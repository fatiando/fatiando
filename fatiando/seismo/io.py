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
Input and output of seismological data

Functions:

* :func:`fatiando.seismo.io.dump_traveltime` 
    Save travel time data to a file

* :func:`fatiando.seismo.io.load_traveltime` 
    Load traveltime data from a file

"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 11-Sep-2010'


import logging

import fatiando
import fatiando.seismo.traveltime


# Add the default handler (a null handler) to the logger to ensure that
# it won't print verbose if the program calling them doesn't want it
log = logging.getLogger('fatiando.seismo.io')       
log.setLevel(logging.DEBUG)
log.addHandler(fatiando.default_log_handler)


def dump_traveltime(fname, data, fmt='ascii'):
    """
    Save travel time data to a file.
        
    File columns format::
    
      src_x  src_y  rec_x  rec_y  time  error
    
    Parameters:
    
    * fname
        Either string with file name or file object.
    
    * data
        Travel time data stored in a dictionary.
                
    * fmt
        File format. For now only supports ``ASCII``.
      
    The data dictionary must be such as::
         
        {'src':[(x1,y1), (x2,y2), ...], 'rec':[(x1,y1), (x2,y2), ...], 
         'traveltime':[time1, time2, ...], 'error':[error1, error2, ...]}
         
    """
    
    if isinstance(fname, file):
        
        output = fname
        
    else:
        
        output = open(fname, 'w')
        
    log.info("Saving travel times to file '%s'" % (output.name))
        
    output.write("# File structure:\n")
    output.write("# src_x  src_y  rec_x  rec_y  time  error\n")
        
    for src, rec, data, error in zip(data['src'], data['rec'], 
                                     data['traveltime'], data['error']):
        
        output.write("%g %g %g %g %g %g\n" % (src[0], src[1], rec[0], rec[1], 
                                              data, error))
        
    output.close()
    
    
def load_traveltime(fname, fmt='ascii'):
    """
    Load travel time data from a file.
    
    Note: lines that start with ``#`` will be considered comments.
    
    File columns format::
    
      src_x  src_y  rec_x  rec_y  time  error
    
    Parameters:
    
    * fname
        Either string with file name or file object.
                
    * fmt
        File format. For now only supports ``ASCII``.
      
    Return:
    
    * travel time data stored in a dictionary
            
    The data dictionary must be such as::
         
        {'src':[(x1,y1), (x2,y2), ...], 'rec':[(x1,y1), (x2,y2), ...], 
         'traveltime':[time1, time2, ...], 'error':[error1, error2, ...]}
         
    """
    
    if isinstance(fname, file):
        
        input = fname
        
    else:
        
        input = open(fname, 'r')
        
    log.info("Loading travel times from file '%s'" % (input.name))
    
    data = {'src':[], 'rec':[], 'traveltime':[], 'error':[]}
    
    for l, line in enumerate(input):
        
        if line[0] == '#':
            
            continue
        
        args = line.strip().split(" ")
        
        if len(args) != 6:
            
            log.warning("  Wrong number of values in line %d." % (l + 1) + 
                        " Ignoring it.")
            
            continue
        
        src_x, src_y, rec_x, rec_y, value, error = args
        
        data['src'].append((float(src_x), float(src_y)))        
        data['rec'].append((float(rec_x), float(rec_y)))
        data['traveltime'].append(float(value))
        data['error'].append(float(error))
        
    log.info("  travel times loaded=%d" % (len(data['traveltime'])))
    
    return data