"""
Unit tests for fatiando.math.lu module
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 07-May-2010'


import unittest

import numpy

from fatiando.utils.points import Cart2DPoint


class Cart2DPointTestCase(unittest.TestCase):
    """
    Test the Cart2DPoint class.
    """

    label = 'fast'
    
    
    def setUp(self):
        "Set up the testing data"
        
        if self.label == 'fast':
            
            self.known_xs = numpy.arange(-10, 10, 0.143)
            self.knwon_ys = numpy.arange(-10, 10, 0.226)
            
        else:            
            
            self.known_xs = numpy.arange(-100, 100, 0.143)
            self.knwon_ys = numpy.arange(-100, 100, 0.226)
                        
            
    def test_known_values(self):
        "Cart2DPoint getter props return the correct results given test data"
        
        t_num = 1
        
        for x in self.known_xs:
            
            for y in self.knwon_ys:
                
                point = Cart2DPoint(x, y)
                
                my_x = point.x
                
                my_y = point.y
            
                self.assertEquals(x, my_x, \
                                  msg="Failed test case %d." % (t_num) + \
                                  " x returned: %g   correct: %g" % (x, my_x))
            
                self.assertEquals(y, my_y, \
                                  msg="Failed test case %d." % (t_num) + \
                                  " y returned: %g   correct: %g" % (y, my_y))
            
                
                
# Return the test suit for this module
################################################################################
def suite(label='fast'):

    
    testsuite = unittest.TestSuite()

    Cart2DPointTestCase.label=label    
    testsuite.addTest(unittest.makeSuite(Cart2DPointTestCase, prefix='test'))
    
    return testsuite

################################################################################

if __name__ == '__main__':

    unittest.main(defaultTest='suite')
        