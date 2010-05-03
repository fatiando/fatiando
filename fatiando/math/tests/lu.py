"""
Unit tests for fatiando.math.lu module
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 01-Mar-2010'


import unittest
import exceptions as ex
import pylab
import os
import sys

from fatiando.math.lu import decomp_nopivot, decomp, solve, inv


class LUDecompNoPivotTestCase(unittest.TestCase):

    label = 'fast'

    def setUp(self):
        #                  Matrix                LU
        self.known = ( ( [[1,2],[3,4]]  ,  [[1,2],[3,-2]] ), \
                       ( [[5,3],[1,8]]  ,  [[5,3],[0.2,7.4]] ), \
                       ( [[6,9],[7,4]]  ,  [[6,9],[7./6,4-63./6]] ), \
                       ( [[1,8,2],[2,6,5],[7,10,1]] ,  \
                         [[1,8,2],[2,-10,1],[7,4.6,-17.6]])
                     )

        # Non-decomposable matrices
        self.known_nondec = ( [[1,3,2],[2,6,4],[7,10,1]], \
                              [[3,4,6],[9,12,18],[7,4,2]], \
                            )


    def test_matrix_dim(self):
        "lu.decomp_nopivot returns matrix of the right dimension"

        tnum = 0
        for matrix, lu_known in self.known:

            lu_my = decomp_nopivot(matrix)

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


    def test_x_known_values(self):
        "lu.decomp_nopivot returns the correct results given test data"

        tnum = 0
        for system, lu_known in self.known:

            lu_my = decomp_nopivot(system)

            tnum += 1

            # Check each element in lu_my
            for i in range(len(lu_my)):
                for j in range(len(lu_my[i])):

                    self.assertAlmostEqual(lu_my[i][j], lu_known[i][j], \
                    places=8, \
                    msg="Failed test %d for matrix element %d,%d = %g\n"\
                        % (tnum, i, j, lu_my[i][j]))

    def test_string_input(self):
        "lu.decomp_nopivot raises TypeError when passed string"
        self.assertRaises(ex.TypeError, decomp_nopivot, "meh")

    def test_int_input(self):
        "lu.decomp_nopivot raises TypeError when passed int"
        self.assertRaises(ex.TypeError, decomp_nopivot, 533)

    def test_float_input(self):
        "lu.decomp_nopivot raises TypeError when passed float"
        self.assertRaises(ex.TypeError, decomp_nopivot, 34.2564)

    def test_tuple_input(self):
        "lu.decomp_nopivot raises TypeError when passed tuple"
        self.assertRaises(ex.TypeError, decomp_nopivot, ((1,2),(3,4)))

    def test_dict_input(self):
        "lu.decomp_nopivot raises TypeError when passed dict"
        self.assertRaises(ex.TypeError, decomp_nopivot, {1:2,3:4})

    def test_listofstrings_input(self):
        "lu.decomp_nopivot raises TypeError when passed list of strings"
        self.assertRaises(ex.TypeError, decomp_nopivot, ['meh','bla'])

    def test_listoffloats_input(self):
        "lu.decomp_nopivot raises TypeError when passed list of floats"
        self.assertRaises(ex.TypeError, decomp_nopivot, [2.34, 24.677])

    def test_listofints_input(self):
        "lu.decomp_nopivot raises TypeError when passed list of ints"
        self.assertRaises(ex.TypeError, decomp_nopivot, [5654, 677])

    def test_nonsquare_input(self):
        "lu.decomp_nopivot raises AttributeError when passed non-square matrix"
        self.assertRaises(ex.AttributeError, decomp_nopivot, [[5654, 677],[2.3]])
        self.assertRaises(ex.AttributeError, decomp_nopivot, [[5654],[2.3]])
        self.assertRaises(ex.AttributeError, decomp_nopivot, [[5654, 677],[2.3,4,5]])

    def test_matrixofstrings_input(self):
        "lu.decomp_nopivot raises TypeError when passed matrix of strings"
        self.assertRaises(ex.TypeError, decomp_nopivot, [['1','2'],['3','4']])

    def test_nondecomposable_input(self):
        "lu.decomp_nopivot raises AttributeError when passed a non-decomposable matrix"
        tnum = 0
        for matrix in self.known_nondec:
            tnum += 1
            try:
                self.assertRaises(ex.AttributeError, decomp_nopivot, matrix)
            except Exception as e:
                e.args = ("AttributeError not raised in test case %d" % (tnum),)
                raise e



class LUDecompTestCase(unittest.TestCase):

    label = 'fast'

    def setUp(self):

        self.testdatadir = os.path.join(os.path.dirname(__file__),'testdata')

        if not os.path.exists(self.testdatadir):
            import fatiando.math.tests.linsysgen
            sys.stderr.write("(Generating test data... ")
            fatiando.math.tests.linsysgen.mktestdata(self.testdatadir)
            sys.stderr.write("done) ")

        testcases = len(os.listdir(self.testdatadir)) / 6

        if self.label == 'full':
            self.number_testcases = testcases
        else:
            self.number_testcases = int(testcases/2)


    def test_x_known_values(self):
        "lu.decomp returns the correct results given test data"

        sys.stderr.write("(%d sets) " % (self.number_testcases))

        for tnum in range(1, self.number_testcases + 1):

            lu_known = pylab.loadtxt(os.path.join(self.testdatadir, 'lu%d.txt' % (tnum))).tolist()
            p_known = pylab.loadtxt(os.path.join(self.testdatadir, 'permut%d.txt' % (tnum))).tolist()
            matrix = pylab.loadtxt(os.path.join(self.testdatadir, 'matrix%d.txt' % (tnum))).tolist()

            lu_my, p_my = decomp(matrix)

            # Check the number of lines in lu
            nlines = len(lu_my)

            self.assertEqual(nlines, len(matrix), \
                msg="Failed test %d. Different number of lines in LU" % (tnum))

            # Check each element in lu_my
            for i in range(len(lu_my)):

                # Check the size of each line
                self.assertEqual(len(lu_my[i]), nlines, \
                msg="Failed test %d. Different number of columns in line %d of LU" \
                % (tnum, i))

                for j in range(len(lu_my[i])):

                    self.assertAlmostEqual(lu_my[i][j], lu_known[i][j], \
                    places=6, \
                    msg="Failed test %d for matrix element %d,%d: my = %.6f / known = %.6f\n"\
                        % (tnum, i, j, lu_my[i][j], lu_known[i][j]))

            # Check the size of the permutation vector
            self.assertEqual(len(p_my), len(matrix), \
                msg="Failed test %d. Different number of lines in permutation vector" % (tnum))

            # Check each element in p_my
            for i in range(len(p_my)):
                self.assertAlmostEqual(p_my[i], p_known[i], \
                places=6, \
                msg="Failed test %d for permutation element %d: my = %.6f / known = %.6f\n"\
                    % (tnum, i, p_my[i], p_known[i]))

    def test_string_input(self):
        "lu.decomp raises TypeError when passed string"
        self.assertRaises(ex.TypeError, decomp, "meh")

    def test_int_input(self):
        "lu.decomp raises TypeError when passed int"
        self.assertRaises(ex.TypeError, decomp, 533)

    def test_float_input(self):
        "lu.decomp raises TypeError when passed float"
        self.assertRaises(ex.TypeError, decomp, 34.2564)

    def test_tuple_input(self):
        "lu.decomp raises TypeError when passed tuple"
        self.assertRaises(ex.TypeError, decomp, ((1,2),(3,4)))

    def test_dict_input(self):
        "lu.decomp raises TypeError when passed dict"
        self.assertRaises(ex.TypeError, decomp, {1:2,3:4})

    def test_listofstrings_input(self):
        "lu.decomp raises TypeError when passed list of strings"
        self.assertRaises(ex.TypeError, decomp, ['meh','bla'])

    def test_listoffloats_input(self):
        "lu.decomp raises TypeError when passed list of floats"
        self.assertRaises(ex.TypeError, decomp, [2.34, 24.677])

    def test_listofints_input(self):
        "lu.decomp raises TypeError when passed list of ints"
        self.assertRaises(ex.TypeError, decomp, [5654, 677])

    def test_nonsquare_input(self):
        "lu.decomp raises AttributeError when passed non-square matrix"
        self.assertRaises(ex.AttributeError, decomp, [[5654, 677],[2.3]])
        self.assertRaises(ex.AttributeError, decomp, [[5654],[2.3]])
        self.assertRaises(ex.AttributeError, decomp, [[5654, 677],[2.3,4,5]])

    def test_matrixofstrings_input(self):
        "lu.decomp raises TypeError when passed matrix of strings"
        self.assertRaises(ex.TypeError, decomp, [['1','2'],['3','4']])
        
        
        
        
class LUSolveTestCase(unittest.TestCase):

    label = 'fast'

    def setUp(self):

        self.testdatadir = os.path.join(os.path.dirname(__file__),'testdata')

        if not os.path.exists(self.testdatadir):
            import fatiando.math.tests.linsysgen
            sys.stderr.write("(Generating test data... ")
            fatiando.math.tests.linsysgen.mktestdata(self.testdatadir)
            sys.stderr.write("done) ")

        testcases = len(os.listdir(self.testdatadir)) / 6

        if self.label == 'full':
            self.number_testcases = testcases
        else:
            self.number_testcases = int(testcases/2)


    def test_for_known_values(self):
        "lu.solve returns the correct results given test data"

        sys.stderr.write("(%d sets) " % (self.number_testcases))

        for tnum in range(1, self.number_testcases + 1):

            lu = pylab.loadtxt(os.path.join(self.testdatadir, 'lu%d.txt' % (tnum))).tolist()
            p = pylab.loadtxt(os.path.join(self.testdatadir, 'permut%d.txt' % (tnum)), dtype=int).tolist()
            y = pylab.loadtxt(os.path.join(self.testdatadir, 'data%d.txt' % (tnum))).tolist()
            x_known = pylab.loadtxt(os.path.join(self.testdatadir, 'solution%d.txt' % (tnum))).tolist()

            x_my = solve(lu, p, y)

            self.assertEqual(len(x_my), len(lu), \
                msg="Failed test %d. Different number of lines in returned solution x" % (tnum))

            # Check each element in x_my
            for i in range(len(x_my)):
                self.assertAlmostEqual(x_my[i], x_known[i], \
                places=8, \
                msg="Failed test %d for element %d: my = %.8f / known = %.8f\n"\
                    % (tnum, i, x_my[i], x_known[i]))

    def test_A_string_input(self):
        "lu.solve raises TypeError when passed string as A"
        self.assertRaises(ex.TypeError, solve, "meh", [1,2], [1,2])

    def test_A_int_input(self):
        "lu.solve raises TypeError when passed int as A"
        self.assertRaises(ex.TypeError, solve, 533, [1,2], [1,2])

    def test_A_float_input(self):
        "lu.solve raises TypeError when passed float as A"
        self.assertRaises(ex.TypeError, solve, 34.2564, [1,2], [1,2])

    def test_A_tuple_input(self):
        "lu.solve raises TypeError when passed tuple as A"
        self.assertRaises(ex.TypeError, solve, ((1,2),(3,4)), [1,2], [1,2])

    def test_A_dict_input(self):
        "lu.solve raises TypeError when passed dict as A"
        self.assertRaises(ex.TypeError, solve, {1:2,3:4}, [1,2], [1,2])

    def test_A_listofstrings_input(self):
        "lu.solve raises TypeError when passed list of strings as A"
        self.assertRaises(ex.TypeError, solve, ['m','e'], [1,2], [1,2])

    def test_A_listoffloats_input(self):
        "lu.solve raises TypeError when passed list of floats as A"
        self.assertRaises(ex.TypeError, solve, [2.34, 24.677], [1,2], [1,2])

    def test_A_listofints_input(self):
        "lu.solve raises TypeError when passed list of ints as A"
        self.assertRaises(ex.TypeError, solve, [5654, 677], [1,2], [1,2])

    def test_A_nonsquare_input(self):
        "lu.solve raises AttributeError when passed non-square matrix as A"
        self.assertRaises(ex.AttributeError, solve, [[5654, 677],[2.3]], [2,3], [1,2])
        self.assertRaises(ex.AttributeError, solve, [[5654],[2.3]], [1,4,5], [1,2])
        self.assertRaises(ex.AttributeError, solve, [[5654, 677],[2.3,4,5]], [2,3], [1,2])

    def test_A_matrixofstrings_input(self):
        "lu.solve raises TypeError when passed matrix of strings as A"
        self.assertRaises(ex.TypeError, solve, [['1','2'],['3','4']], [1,2], [1,2])
  
    def test_y_string_input(self):
        "lu.solve raises TypeError when passed string as y"
        self.assertRaises(ex.TypeError, solve, [[1,2],[3,4]], [1,2], "meh")

    def test_y_int_input(self):
        "lu.solve raises TypeError when passed int as y"
        self.assertRaises(ex.TypeError, solve, [[1,2],[3,4]], [1,2], 533)

    def test_y_float_input(self):
        "lu.solve raises TypeError when passed float as y"
        self.assertRaises(ex.TypeError, solve, [[1,2],[3,4]], [1,2], 34.2564)

    def test_y_tuple_input(self):
        "lu.solve raises TypeError when passed tuple as y"
        self.assertRaises(ex.TypeError, solve, [[1,2],[3,4]], [1,2], (1,2))

    def test_y_dict_input(self):
        "lu.solve raises TypeError when passed dict as y"
        self.assertRaises(ex.TypeError, solve, [[1,2],[3,4]], [1,2], {1:2,3:4})

    def test_y_listofstrings_input(self):
        "lu.solve raises TypeError when passed list of strings as y"
        self.assertRaises(ex.TypeError, solve, [[1,2],[3,4]], [1,2], ['m','e'])

    def test_y_wrongsize_input(self):
        "lu.solve raises AttributeError when passed wrong size y"
        self.assertRaises(ex.AttributeError, solve, [[1,2],[3,4]], [1,2], [1])
        self.assertRaises(ex.AttributeError, solve, [[1,2],[3,4]], [1,2], [1,2,3,4])
        
        
        
class LUInvTestCase(unittest.TestCase):

    label = 'fast'

    def setUp(self):

        self.testdatadir = os.path.join(os.path.dirname(__file__),'testdata')

        if not os.path.exists(self.testdatadir):
            import fatiando.math.tests.linsysgen
            sys.stderr.write("(Generating test data... ")
            fatiando.math.tests.linsysgen.mktestdata(self.testdatadir)
            sys.stderr.write("done) ")

        testcases = len(os.listdir(self.testdatadir)) / 6

        if self.label == 'full':
            self.number_testcases = testcases
        else:
            self.number_testcases = int(testcases/2)


    def test_for_known_values(self):
        "lu.inv returns the correct results given test data"

        sys.stderr.write("(%d sets) " % (self.number_testcases))

        for tnum in range(1, self.number_testcases + 1):

            lu = pylab.loadtxt(os.path.join(self.testdatadir, 'lu%d.txt' % (tnum))).tolist()
            p = pylab.loadtxt(os.path.join(self.testdatadir, 'permut%d.txt' % (tnum)), dtype=int).tolist()
            inv_known = pylab.loadtxt(os.path.join(self.testdatadir, 'inverse%d.txt' % (tnum))).tolist()

            inv_my = inv(lu, p)
            
            # Check the number of lines
            nlines = len(inv_my)

            self.assertEqual(nlines, len(lu), \
                msg="Failed test %d. Different number of lines in LU" % (tnum))

            # Check each element in inv_my
            for i in range(len(inv_my)):

                # Check the size of each line
                self.assertEqual(len(inv_my[i]), nlines, \
                msg="Failed test %d. Different number of columns in line %d of LU" \
                % (tnum, i))

                for j in range(len(inv_my[i])):

                    self.assertAlmostEqual(inv_my[i][j], inv_known[i][j], \
                    places=6, \
                    msg="Failed test %d for matrix element %d,%d: my = %.6f / known = %.6f\n"\
                        % (tnum, i, j, inv_my[i][j], inv_known[i][j]))
                    
                    
    def test_LU_string_input(self):
        "lu.inv raises TypeError when passed string as LU"
        self.assertRaises(ex.TypeError, inv, "meh", [1,2])

    def test_LU_int_input(self):
        "lu.inv raises TypeError when passed int as LU"
        self.assertRaises(ex.TypeError, inv, 533, [1,2])

    def test_LU_float_input(self):
        "lu.inv raises TypeError when passed float as LU"
        self.assertRaises(ex.TypeError, inv, 34.2564, [1,2])

    def test_LU_tuple_input(self):
        "lu.inv raises TypeError when passed tuple as LU"
        self.assertRaises(ex.TypeError, inv, ((1,2),(3,4)), [1,2])

    def test_LU_dict_input(self):
        "lu.inv raises TypeError when passed dict as LU"
        self.assertRaises(ex.TypeError, inv, {1:2,3:4}, [5.,3.6])

    def test_LU_listofstrings_input(self):
        "lu.inv raises TypeError when passed list of strings as LU"
        self.assertRaises(ex.TypeError, inv, ['m','e'], [1,2])

    def test_LU_listoffloats_input(self):
        "lu.inv raises TypeError when passed list of floats as LU"
        self.assertRaises(ex.TypeError, inv, [2.34, 24.677], [1,2])

    def test_LU_listofints_input(self):
        "lu.inv raises TypeError when passed list of ints as LU"
        self.assertRaises(ex.TypeError, inv, [5654, 677], [1,2])

    def test_LU_nonsquare_input(self):
        "lu.inv raises AttributeError when passed non-square matrix as LU"
        self.assertRaises(ex.AttributeError, inv, [[5654, 677],[2.3]], [2,3])
        self.assertRaises(ex.AttributeError, inv, [[5654],[2.3]], [1,4,5])
        self.assertRaises(ex.AttributeError, inv, [[5654, 677],[2.3,4,5]], [2,3])

    def test_LU_matrixofstrings_input(self):
        "lu.inv raises TypeError when passed matrix of strings as LU"
        self.assertRaises(ex.TypeError, inv, [['1','2'],['3','4']], [13,2.551])
  
    def test_p_string_input(self):
        "lu.inv raises TypeError when passed string as p"
        self.assertRaises(ex.TypeError, inv, [[1,2],[3,4]], "meh")

    def test_p_int_input(self):
        "lu.inv raises TypeError when passed int as p"
        self.assertRaises(ex.TypeError, inv, [[1,2],[3,4]], 533)

    def test_p_float_input(self):
        "lu.inv raises TypeError when passed float as p"
        self.assertRaises(ex.TypeError, inv, [[1,2],[3,4]], 34.2564)

    def test_p_tuple_input(self):
        "lu.inv raises TypeError when passed tuple as p"
        self.assertRaises(ex.TypeError, inv, [[1,2],[3,4]], (1,2))

    def test_p_dict_input(self):
        "lu.inv raises TypeError when passed dict as p"
        self.assertRaises(ex.TypeError, inv, [[1,2],[3,4]], {1:2,3:4})

    def test_p_listofstrings_input(self):
        "lu.inv raises TypeError when passed list of strings as p"
        self.assertRaises(ex.TypeError, inv, [[1,2],[3,4]], ['m','e'])

    def test_p_wrongsize_input(self):
        "lu.inv raises AttributeError when passed wrong size p"
        self.assertRaises(ex.AttributeError, inv, [[1,2],[3,4]], [1])
        self.assertRaises(ex.AttributeError, inv, [[1,2],[3,4]], [1,2,3,4])
        
        
# Return the test suit for this module
################################################################################
def suite(label='fast'):

    
    suite = unittest.TestSuite()

    LUDecompNoPivotTestCase.label=label    
    suite.addTest(unittest.makeSuite(LUDecompNoPivotTestCase, prefix='test'))
    
    LUDecompTestCase.label = label
    suite.addTest(unittest.makeSuite(LUDecompTestCase, prefix='test'))
    
    LUSolveTestCase.label = label
    suite.addTest(unittest.makeSuite(LUSolveTestCase, prefix='test'))
    
    LUInvTestCase.label = label
    suite.addTest(unittest.makeSuite(LUInvTestCase, prefix='test'))

    return suite

################################################################################

if __name__ == '__main__':

    unittest.main(defaultTest='suite')
