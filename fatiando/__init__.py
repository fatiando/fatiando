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
Geophysical direct and inverse modeling package. 

Includes various direct models, inversion programs,and various utilities for 
general geophysics tasks.

Subpackages:

* :mod:`fatiando.grav`
    Gravimetry, geodesy and gravity gradiometry
    
* :mod:`fatiando.heat` 
    Geothermology modeling
    
* :mod:`fatiando.inversion`
    A collection of geophysical inverse problem solvers.
    
* :mod:`fatiando.seismo`
    Seismology and seismic methods
        
Modules:

* :mod:`fatiando.mesh`
    Mesh generation and handling of geometric elements
    
* :mod:`fatiando.stats`
    Statistical tests and utilities for inverse problems
    
* :mod:`fatiando.utils`
    Miscellaneous utilities

* :mod:`fatiando.vis`
    Visualization of results in 2D and 3D
        
Functions:

* :func:`fatiando.test`
    Run the unit test suite for this package
        
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 02-Apr-2010'


__version__ = '0.0.1'


# Create a default NullHandler so that logging is only enabled explicitly
################################################################################ 
import logging

class NullHandler(logging.Handler):
    """
    Default null handler so that logging is only done when explicitly asked for.
    """
    
    def emit(self, record):
        
        pass

default_log_handler = NullHandler()
################################################################################


def test(label='fast', verbose=True):
    """
    Runs the unit tests for the fatiando package.

    Parameters:

    * label
        Can be either ``'fast'`` for a smaller and faster test or ``'full'`` for
        the full test suite

    * verbose
        Controls if the whole test information is printed or just the final 
        results
        
    """
    
    if label != 'fast' and label != 'full':
        
        from exceptions import ValueError
        
        raise ValueError("Test label must be either 'fast' or 'full'")

    import unittest

    import fatiando.tests

    suite = unittest.TestSuite()
    
    suite.addTest(fatiando.tests.suite(label))

    if verbose:
        runner = unittest.TextTestRunner(verbosity=2)
    else:
        runner = unittest.TextTestRunner(verbosity=0)

    runner.run(suite)