# -*- coding: utf-8 -*-
################################################################################
"""
Unit tests for linalg.solve_linsys_lu
(Solve a linear system given it's LU decomposition).
"""
################################################################################
# Created on 04-Mar-2010
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__revision__ = '$Revision: 16 $'
__date__ = '$Date: 2010-03-19 14:30:43 -0300 (Fri, 19 Mar 2010) $'

# Do the startup
import sloot_test_startup

import unittest
import exceptions as ex
import pylab

from sloot.linalg import solve_linsys_lu


# TEST FOR SUCCESS
################################################################################

class SolveLinsysLUKnownValues(unittest.TestCase):
    """
    Test if solve_linsys_lu is returning the right results
    """

    number_testcases = 30

    def test_return_matrix_dim(self):
        "solve_linsys_lu returns a vector of the right dimension"

        for tnum in range(1, self.number_testcases + 1):

            lu = pylab.loadtxt('test/linsys-data/lu%d.txt' % (tnum)).tolist()
            p = pylab.loadtxt('test/linsys-data/permut%d.txt' % (tnum), dtype=int).tolist()
            y = pylab.loadtxt('test/linsys-data/data%d.txt' % (tnum)).tolist()

            x_my = solve_linsys_lu(lu, p, y)

            self.assertEqual(len(x_my), len(lu), \
                msg="Failed test %d. Different number of lines in returned solution x" % (tnum))


    def test_for_known_values(self):
        "solve_linsys_lu returns the correct results given test data"

        for tnum in range(1, self.number_testcases + 1):

            lu = pylab.loadtxt('test/linsys-data/lu%d.txt' % (tnum)).tolist()
            p = pylab.loadtxt('test/linsys-data/permut%d.txt' % (tnum), dtype=int).tolist()
            y = pylab.loadtxt('test/linsys-data/data%d.txt' % (tnum)).tolist()
            x_known = pylab.loadtxt('test/linsys-data/solution%d.txt' % (tnum)).tolist()
        
            x_my = solve_linsys_lu(lu, p, y)

            # Check each element in x_my
            for i in range(len(x_my)):
                self.assertAlmostEqual(x_my[i], x_known[i], \
                places=8, \
                msg="Failed test %d for element %d: my = %.8f / known = %.8f\n"\
                    % (tnum, i, x_my[i], x_known[i]))


################################################################################


# TEST FOR FAILURE
################################################################################

class SolveLinsysLURaises(unittest.TestCase):
    """
    Test if solve_linsys_lu is raising the correct exceptions.
    """

    # A as wrong input
    ####################################################################
    def test_A_string_input(self):
        "solve_linsys_lu raises TypeError when passed string as A"
        self.assertRaises(ex.TypeError, solve_linsys_lu, "meh", [1,2], [1,2])

    def test_A_int_input(self):
        "solve_linsys_lu raises TypeError when passed int as A"
        self.assertRaises(ex.TypeError, solve_linsys_lu, 533, [1,2], [1,2])

    def test_A_float_input(self):
        "solve_linsys_lu raises TypeError when passed float as A"
        self.assertRaises(ex.TypeError, solve_linsys_lu, 34.2564, [1,2], [1,2])

    def test_A_tuple_input(self):
        "solve_linsys_lu raises TypeError when passed tuple as A"
        self.assertRaises(ex.TypeError, solve_linsys_lu, ((1,2),(3,4)), [1,2], [1,2])

    def test_A_dict_input(self):
        "solve_linsys_lu raises TypeError when passed dict as A"
        self.assertRaises(ex.TypeError, solve_linsys_lu, {1:2,3:4}, [1,2], [1,2])

    def test_A_listofstrings_input(self):
        "solve_linsys_lu raises TypeError when passed list of strings as A"
        self.assertRaises(ex.TypeError, solve_linsys_lu, ['m','e'], [1,2], [1,2])

    def test_A_listoffloats_input(self):
        "solve_linsys_lu raises TypeError when passed list of floats as A"
        self.assertRaises(ex.TypeError, solve_linsys_lu, [2.34, 24.677], [1,2], [1,2])

    def test_A_listofints_input(self):
        "solve_linsys_lu raises TypeError when passed list of ints as A"
        self.assertRaises(ex.TypeError, solve_linsys_lu, [5654, 677], [1,2], [1,2])

    def test_A_nonsquare_input(self):
        "solve_linsys_lu raises AttributeError when passed non-square matrix as A"
        self.assertRaises(ex.AttributeError, solve_linsys_lu, [[5654, 677],[2.3]], [2,3], [1,2])
        self.assertRaises(ex.AttributeError, solve_linsys_lu, [[5654],[2.3]], [1,4,5], [1,2])
        self.assertRaises(ex.AttributeError, solve_linsys_lu, [[5654, 677],[2.3,4,5]], [2,3], [1,2])

    def test_A_matrixofstrings_input(self):
        "solve_linsys_lu raises TypeError when passed matrix of strings as A"
        self.assertRaises(ex.TypeError, solve_linsys_lu, [['1','2'],['3','4']], [1,2], [1,2])
    ####################################################################

    # Y AS INPUT
    ####################################################################
    def test_y_string_input(self):
        "solve_linsys_lu raises TypeError when passed string as y"
        self.assertRaises(ex.TypeError, solve_linsys_lu, [[1,2],[3,4]], [1,2], "meh")

    def test_y_int_input(self):
        "solve_linsys_lu raises TypeError when passed int as y"
        self.assertRaises(ex.TypeError, solve_linsys_lu, [[1,2],[3,4]], [1,2], 533)

    def test_y_float_input(self):
        "solve_linsys_lu raises TypeError when passed float as y"
        self.assertRaises(ex.TypeError, solve_linsys_lu, [[1,2],[3,4]], [1,2], 34.2564)

    def test_y_tuple_input(self):
        "solve_linsys_lu raises TypeError when passed tuple as y"
        self.assertRaises(ex.TypeError, solve_linsys_lu, [[1,2],[3,4]], [1,2], (1,2))

    def test_y_dict_input(self):
        "solve_linsys_lu raises TypeError when passed dict as y"
        self.assertRaises(ex.TypeError, solve_linsys_lu, [[1,2],[3,4]], [1,2], {1:2,3:4})

    def test_y_listofstrings_input(self):
        "solve_linsys_lu raises TypeError when passed list of strings as y"
        self.assertRaises(ex.TypeError, solve_linsys_lu, [[1,2],[3,4]], [1,2], ['m','e'])

    def test_y_wrongsize_input(self):
        "solve_linsys_lu raises AttributeError when passed wrong size y"
        self.assertRaises(ex.AttributeError, solve_linsys_lu, [[1,2],[3,4]], [1,2], [1])
        self.assertRaises(ex.AttributeError, solve_linsys_lu, [[1,2],[3,4]], [1,2], [1,2,3,4])

    ####################################################################

################################################################################