# -*- coding: utf-8 -*-
################################################################################
"""
Unit tests for linalg.lu_decomp_nopivot (LU decomposition of a matrix without
pivoting).
"""
################################################################################
# Created on 01-Mar-2010
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__revision__ = '$Revision: 16 $'
__date__ = '$Date: 2010-03-19 14:30:43 -0300 (Fri, 19 Mar 2010) $'

# Do the startup
import sloot_test_startup

import unittest
import exceptions as ex

from sloot.linalg import lu_decomp_nopivot


# TEST FOR SUCCESS
################################################################################

class LUDecompNoPivotKnownValues(unittest.TestCase):
    """
    Test if lu_decomp_nopivot is returning the right results
    """

            #     Matrix                LU  (the 2 matrices are combined in one)
    known = ( ( [[1,2],[3,4]]  ,  [[1,2],[3,-2]] ), \
              ( [[5,3],[1,8]]  ,  [[5,3],[0.2,7.4]] ), \
              ( [[6,9],[7,4]]  ,  [[6,9],[7./6,4-63./6]] ), \
              ( [[1,8,2],[2,6,5],[7,10,1]] ,  [[1,8,2],[2,-10,1],[7,4.6,-17.6]])
            )


    def test_lu_decomp_nopivot_matrix_dim(self):
        "lu_decomp_nopivot returns matrix of the right dimension"

        tnum = 0
        for matrix, lu_known in self.known:

            lu_my = lu_decomp_nopivot(matrix)

            tnum += 1

            nlines = len(lu_my)

            self.assertEqual(nlines, len(matrix), \
                             msg="Failed test %d. Different numb of lines" \
                             % (tnum))

            # Check each line length
            lnum = 0
            for line in lu_my:
                lnum += 1
                self.assertEqual(len(line), nlines, \
                msg="Failed test %d. Different numb of columns in line %d" \
                % (tnum, lnum))


    def test_lu_decomp_nopivot_x_known_values(self):
        "lu_decomp_nopivot returns the correct results given test data"

        tnum = 0
        for system, lu_known in self.known:

            lu_my = lu_decomp_nopivot(system)

            tnum += 1

            # Check each element in lu_my
            for i in range(len(lu_my)):
                for j in range(len(lu_my[i])):

                    self.assertAlmostEqual(lu_my[i][j], lu_known[i][j], \
                    places=8, \
                    msg="Failed test %d for matrix element %d,%d = %g\n"\
                        % (tnum, i, j, lu_my[i][j]))


################################################################################


# TEST FOR FAILURE
################################################################################

class LUDecompNoPivotRaises(unittest.TestCase):
    """
    Test if lu_decomp_nopivot is raising the correct exceptions.
    """

    # Non-decomposable matrices
    known = ( [[1,3,2],[2,6,4],[7,10,1]], \
              [[3,4,6],[9,12,18],[7,4,2]], \
            )

    def test_string_input(self):
        "lu_decomp_nopivot raises TypeError when passed string"
        self.assertRaises(ex.TypeError, lu_decomp_nopivot, "meh")

    def test_int_input(self):
        "lu_decomp_nopivot raises TypeError when passed int"
        self.assertRaises(ex.TypeError, lu_decomp_nopivot, 533)

    def test_float_input(self):
        "lu_decomp_nopivot raises TypeError when passed float"
        self.assertRaises(ex.TypeError, lu_decomp_nopivot, 34.2564)

    def test_tuple_input(self):
        "lu_decomp_nopivot raises TypeError when passed tuple"
        self.assertRaises(ex.TypeError, lu_decomp_nopivot, ((1,2),(3,4)))

    def test_dict_input(self):
        "lu_decomp_nopivot raises TypeError when passed dict"
        self.assertRaises(ex.TypeError, lu_decomp_nopivot, {1:2,3:4})

    def test_listofstrings_input(self):
        "lu_decomp_nopivot raises TypeError when passed list of strings"
        self.assertRaises(ex.TypeError, lu_decomp_nopivot, ['meh','bla'])

    def test_listoffloats_input(self):
        "lu_decomp_nopivot raises TypeError when passed list of floats"
        self.assertRaises(ex.TypeError, lu_decomp_nopivot, [2.34, 24.677])

    def test_listofints_input(self):
        "lu_decomp_nopivot raises TypeError when passed list of ints"
        self.assertRaises(ex.TypeError, lu_decomp_nopivot, [5654, 677])

    def test_nonsquare_input(self):
        "lu_decomp_nopivot raises AttributeError when passed non-square matrix"
        self.assertRaises(ex.AttributeError, lu_decomp_nopivot, [[5654, 677],[2.3]])
        self.assertRaises(ex.AttributeError, lu_decomp_nopivot, [[5654],[2.3]])
        self.assertRaises(ex.AttributeError, lu_decomp_nopivot, [[5654, 677],[2.3,4,5]])

    def test_matrixofstrings_input(self):
        "lu_decomp_nopivot raises TypeError when passed matrix of strings"
        self.assertRaises(ex.TypeError, lu_decomp_nopivot, [['1','2'],['3','4']])

    def test_nondecomposable_input(self):
        "lu_decomp_nopivot raises AttributeError when passed a non-decomposable matrix"
        tnum = 0
        for matrix in self.known:
            tnum += 1
            try:
                self.assertRaises(ex.AttributeError, lu_decomp_nopivot, matrix)
            except Exception as e:
                e.args = ("AttributeError not raised in test case %d" % (tnum),)
                raise e


################################################################################