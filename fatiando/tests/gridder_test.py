# Copyright 2011 The Fatiando a Terra Development Team
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
Unit tests for the fatiando.gridder module
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 26-Sep-2011'


import unittest
from fatiando import gridder

class GridderRegularTest(unittest.TestCase):
    def setUp(self):
        self._fixtures = [
        [(1,2,1,3,(3,3)),
         [[1,1.5,2,1,1.5,2,1,1.5,2],[1,1,1,2,2,2,3,3,3]]
        ],
        [(-1,2,-2,2,(5,4)),
         [[-1,0,1,2,-1,0,1,2,-1,0,1,2,-1,0,1,2,-1,0,1,2],
          [-2,-2,-2,-2,-1,-1,-1,-1,0,0,0,0,1,1,1,1,2,2,2,2]]
        ],
        [(2.5,3,-5,-1,(5,2),10.2),
         [[2.5,3,2.5,3,2.5,3,2.5,3,2.5,3],
          [-5,-5,-4,-4,-3,-3,-2,-2,-1,-1],
          [10.2,10.2,10.2,10.2,10.2,10.2,10.2,10.2,10.2,10.2]]
        ]
        ]
        self._fails = ()

    def test_regular(self):
        "gridder.regular returns correct values"
        for test in self._fixtures:
            args, true = test
            output = gridder.regular(*args)
            failmsg = ("\ntest:%s" % (str(test)) + "\noutput:%s" % (str(output)))
            for ctrue, cout in zip(true, output):
                self.assertEqual(cout.tolist(), ctrue, msg=failmsg)

class GridderScatterTest(unittest.TestCase):
    def setUp(self):
        self._fixtures = [
        ]

    def test_scatter(self):
        "gridder.scatter returns correct values"
        # mock out the random numbers
        gridder.numpy.random.uniform = range


if __name__ == '__main__':
    unittest.main()
