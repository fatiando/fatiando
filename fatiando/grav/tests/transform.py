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
Unit tests for :mod:`fatiando.grav.transform`
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 26-Oct-2010'

import unittest

import fatiando.grav.transform as transform


class UpContinueTestCase(unittest.TestCase):
    """Test case for :func:`fatiando.grav.transform.upcontinue`"""
    
    label = 'fast'

    def setUp(self):
        pass


    def test_known_values(self):
        "grav.transform.upcontinue returns correct results"
        pass

        
# Return the test suit for this module
################################################################################
def suite(label='fast'):

    testsuite = unittest.TestSuite()

    UpContinueTestCase.label = label
    testsuite.addTest(unittest.makeSuite(UpContinueTestCase, prefix='test'))
    
    return testsuite

################################################################################


if __name__ == '__main__':

    unittest.main(defaultTest='suite')
