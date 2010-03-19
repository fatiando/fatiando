# -*- coding: utf-8 -*-
################################################################################
"""
Unit tests for linalg.lu_decomp (LU decomposition of a matrix using pivoting).
"""
################################################################################
# Created on 04-Mar-2010
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__revision__ = '$Revision: $'
__date__ = '$Date: $'

# Do the startup
import sloot_test_startup

import unittest
import exceptions as ex
import pylab

from sloot.linalg import lu_decomp


# TEST FOR SUCCESS
################################################################################

class LUDecompKnownValues(unittest.TestCase):
    """
    Test if lu_decomp is returning the right results
    """

    number_testcases = 30

    def test_lu_decomp_matrix_dim(self):
        "lu_decomp returns matrices of the right dimension"

        for tnum in range(1, self.number_testcases + 1):

            matrix = pylab.loadtxt('test/linsys-data/matrix%d.txt' % (tnum)).tolist()

            lu_my, p_my = lu_decomp(matrix)

            # Check the number of lines in lu
            nlines = len(lu_my)

            self.assertEqual(nlines, len(matrix), \
                msg="Failed test %d. Different number of lines in LU" % (tnum))

            # Check each line length
            lnum = 0
            for line in lu_my:
                lnum += 1
                self.assertEqual(len(line), nlines, \
                msg="Failed test %d. Different number of columns in line %d of LU" \
                % (tnum, lnum))

            # Check the size of the permutation vector
            self.assertEqual(len(p_my), len(matrix), \
                msg="Failed test %d. Different number of lines in permutation vector" % (tnum))


    def test_lu_decomp_x_known_values(self):
        "lu_decomp returns the correct results given test data"

        for tnum in range(1, self.number_testcases + 1):

            lu_known = pylab.loadtxt('test/linsys-data/lu%d.txt' % (tnum)).tolist()
            p_known = pylab.loadtxt('test/linsys-data/permut%d.txt' % (tnum)).tolist()
            matrix = pylab.loadtxt('test/linsys-data/matrix%d.txt' % (tnum)).tolist()

            lu_my, p_my = lu_decomp(matrix)

            # Check each element in lu_my
            for i in range(len(lu_my)):
                for j in range(len(lu_my[i])):

                    self.assertAlmostEqual(lu_my[i][j], lu_known[i][j], \
                    places=6, \
                    msg="Failed test %d for matrix element %d,%d: my = %.6f / known = %.6f\n"\
                        % (tnum, i, j, lu_my[i][j], lu_known[i][j]))

            # Check each element in p_my
            for i in range(len(p_my)):
                self.assertAlmostEqual(p_my[i], p_known[i], \
                places=6, \
                msg="Failed test %d for permutation element %d: my = %.6f / known = %.6f\n"\
                    % (tnum, i, p_my[i], p_known[i]))


################################################################################


# TEST FOR FAILURE
################################################################################

class LUDecompRaises(unittest.TestCase):
    """
    Test if lu_decomp is raising the correct exceptions.
    """

    # Non-decomposable matrices
    known = ( [[1,3,2],[2,6,4],[7,10,1]], \
              [[3,4,6],[9,12,18],[7,4,2]], \
            )

    def test_string_input(self):
        "lu_decomp raises TypeError when passed string"
        self.assertRaises(ex.TypeError, lu_decomp, "meh")

    def test_int_input(self):
        "lu_decomp raises TypeError when passed int"
        self.assertRaises(ex.TypeError, lu_decomp, 533)

    def test_float_input(self):
        "lu_decomp raises TypeError when passed float"
        self.assertRaises(ex.TypeError, lu_decomp, 34.2564)

    def test_tuple_input(self):
        "lu_decomp raises TypeError when passed tuple"
        self.assertRaises(ex.TypeError, lu_decomp, ((1,2),(3,4)))

    def test_dict_input(self):
        "lu_decomp raises TypeError when passed dict"
        self.assertRaises(ex.TypeError, lu_decomp, {1:2,3:4})

    def test_listofstrings_input(self):
        "lu_decomp raises TypeError when passed list of strings"
        self.assertRaises(ex.TypeError, lu_decomp, ['meh','bla'])

    def test_listoffloats_input(self):
        "lu_decomp raises TypeError when passed list of floats"
        self.assertRaises(ex.TypeError, lu_decomp, [2.34, 24.677])

    def test_listofints_input(self):
        "lu_decomp raises TypeError when passed list of ints"
        self.assertRaises(ex.TypeError, lu_decomp, [5654, 677])

    def test_nonsquare_input(self):
        "lu_decomp raises AttributeError when passed non-square matrix"
        self.assertRaises(ex.AttributeError, lu_decomp, [[5654, 677],[2.3]])
        self.assertRaises(ex.AttributeError, lu_decomp, [[5654],[2.3]])
        self.assertRaises(ex.AttributeError, lu_decomp, [[5654, 677],[2.3,4,5]])

    def test_matrixofstrings_input(self):
        "lu_decomp raises TypeError when passed matrix of strings"
        self.assertRaises(ex.TypeError, lu_decomp, [['1','2'],['3','4']])

    #def test_nondecomposable_input(self):
        #"lu_decomp raises AttributeError when passed a non-decomposable matrix"
        #tnum = 0
        #for matrix in self.known:
            #tnum += 1
            #try:
                #self.assertRaises(ex.AttributeError, lu_decomp, matrix)
            #except Exception as e:
                #e.args = ("AttributeError not raised in test case %d" % (tnum),)
                #raise e


################################################################################