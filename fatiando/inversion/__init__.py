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
A collection of geophysical inverse problem solvers.

Modules:

* :mod:`fatiando.inversion.solvers`
    Generic solvers used to implement the specific inversions

* :mod:`fatiando.inversion.interg2d`
    2D gravity inversion for the relief of an interface

* :mod:`fatiando.inversion.pgrav3d`
    3D gravity inversion for density

* :mod:`fatiando.inversion.simpletom`
    Very simplified Cartesian travel time tomography example

Functions:

* :func:`fatiando.inversion.test`
    Run the unit test suite for this package.
    
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 02-Apr-2010'


def test(label='fast', verbose=True):
    """
    Runs the unit tests for the fatiando.inversion package.

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

    import fatiando.inversion.tests

    suite = unittest.TestSuite()
    suite.addTest(fatiando.inversion.tests.suite(label))

    if verbose:
        runner = unittest.TextTestRunner(verbosity=2)
    else:
        runner = unittest.TextTestRunner(verbosity=0)

    runner.run(suite)