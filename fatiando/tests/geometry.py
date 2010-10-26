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
Unit tests for :mod:`fatiando.geometry`
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 26-Oct-2010'

import unittest

import fatiando.geometry as geometry


class PrismTestCase(unittest.TestCase):
    """Test case for :func:`fatiando.geometry.prism`"""
    
    
    label = 'fast'


    def setUp(self):

        self._fixtures = (
        ((1, 2, 3, 4, 5, 6, {'a':42}),
         {'x1':1, 'x2':2, 'y1':3, 'y2':4, 'z1':5, 'z2':6, 'a':42}),
        ((1, 2, 3, 4, 5, 6, {}), 
         {'x1':1, 'x2':2, 'y1':3, 'y2':4, 'z1':5, 'z2':6}),
        ((1, 2, 3, 4, 5, 6), 
         {'x1':1, 'x2':2, 'y1':3, 'y2':4, 'z1':5, 'z2':6}),
        ((1, 2, 3, 4, 5, 6, {'a':42, 'q':'6x9'}), 
         {'x1':1, 'x2':2, 'y1':3, 'y2':4, 'z1':5, 'z2':6, 'a':42, 'q':'6x9'})
                         )
        
        self._fails = (
            (1, 1, 3, 4, 5, 6, {}),
            (1, 1, 3, 4, 5, 6),
            (1, 2, 3, 3, 5, 6),
            (1, 2, 3, 4, 5, 5)
                      )


    def test_known_values(self):
        "geometry.prism returns correct values"
        
        for i, test in enumerate(self._fixtures):
            
            params, correct = test
            
            output = geometry.prism(*params)
            
            failmsg = ("Failed test case %d." % (i + 1) + 
                       "test:%s" % (str(test)) + " output:%s" % (str(output)))
            
            self.assertEquals(output, correct, msg=failmsg)
            
    
    def test_assertions(self):
        "geometry.prism fails when passed bad input"
        
        for test in self._fails:
                        
            self.assertRaises(AssertionError, geometry.prism, *test)           
          


        
# Return the test suit for this module
################################################################################
def suite(label='fast'):

    testsuite = unittest.TestSuite()

    PrismTestCase.label = label
    testsuite.addTest(unittest.makeSuite(PrismTestCase, prefix='test'))
    
    return testsuite

################################################################################


if __name__ == '__main__':

    unittest.main(defaultTest='suite')
