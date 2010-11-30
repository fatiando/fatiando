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
A collection of gravimetry and gravity gradiometry direct models, 
transformations and utilities.

Modules:
    
* :mod:`fatiando.grav.io`
    Input and output of gravity related data
    
* :mod:`fatiando.grav.prism`
    Calculate the gravitational potential and its first and second derivatives 
    for the right rectangular prism using the formulas by Nagy *et* *al.* (2000)

* :mod:`fatiando.grav.sphere`
    Calculate the gravitational potential and its first and second derivatives
    for a sphere.
    
* :mod:`fatiando.grav.synthetic`
    Create synthetic gravity data from various types of model
    
* :mod:`fatiando.grav.transform`
    Gravity field transformations like upward continuation, derivatives and 
    total mass.
        
Functions:

* :func:`fatiando.grav.test`
    Run the unit test suite for this package
    
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 16-Mar-2010'




def test(label='fast', verbose=True):
    """
    Runs the unit tests for the fatiando.grav package.

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

    import fatiando.grav.tests

    suite = unittest.TestSuite()
    
    suite.addTest(fatiando.grav.tests.suite(label))

    if verbose:
        runner = unittest.TextTestRunner(verbosity=2)
    else:
        runner = unittest.TextTestRunner(verbosity=0)

    runner.run(suite)