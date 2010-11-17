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
Test suite for the fatiando package.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 29-Mar-2010'

import unittest

# The package tests
import fatiando.grav.tests
import fatiando.seismo.tests
import fatiando.inv.tests
import fatiando.heat.tests
# The module tests
import fatiando.tests.geometry


def suite(label='fast'):

    testsuite = unittest.TestSuite()

    testsuite.addTest(fatiando.grav.tests.suite(label))
    testsuite.addTest(fatiando.seismo.tests.suite(label))
    testsuite.addTest(fatiando.inv.tests.suite(label))
    testsuite.addTest(fatiando.heat.tests.suite(label))
    
    testsuite.addTest(fatiando.tests.geometry.suite(label))

    return testsuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')