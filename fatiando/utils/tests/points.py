"""
Unit tests for fatiando.math.lu module
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 07-May-2010'


import unittest

import numpy

from fatiando.utils import points


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
                
                point = points.Cart2DPoint(x, y)
                
                my_x = point.x
                
                my_y = point.y
            
                self.assertEquals(x, my_x, \
                                  msg="Failed test case %d." % (t_num) + \
                                  " x returned: %g   correct: %g" % (x, my_x))
            
                self.assertEquals(y, my_y, \
                                  msg="Failed test case %d." % (t_num) + \
                                  " y returned: %g   correct: %g" % (y, my_y))
            
                

class Cart3DPointTestCase(unittest.TestCase):
    """
    Test the Cart3DPoint class.
    """

    label = 'fast'
    
    
    def setUp(self):
        "Set up the testing data"
        
        if self.label == 'fast':
            
            self.known_xs = numpy.arange(-1, 1, 0.143)
            self.knwon_ys = numpy.arange(-1, 1, 0.226)
            self.knwon_zs = numpy.arange(-1, 1, 0.026)
            
        else:            
            
            self.known_xs = numpy.arange(-10, 10, 0.143)
            self.knwon_ys = numpy.arange(-10, 10, 0.226)
            self.knwon_zs = numpy.arange(-10, 10, 0.526)
                        
            
    def test_known_values(self):
        "Cart3DPoint getter props return the correct results given test data"
        
        t_num = 1
        
        for x in self.known_xs:
            
            for y in self.knwon_ys:
                
                for z in self.knwon_zs:
                
                    point = points.Cart3DPoint(x, y, z)
                    
                    my_x = point.x
                    
                    my_y = point.y
                    
                    my_z = point.z
                
                    self.assertEquals(x, my_x, \
                                      msg="Failed test case %d." % (t_num) + \
                                    " x returned: %g   correct: %g" % (x, my_x))
                
                    self.assertEquals(y, my_y, \
                                      msg="Failed test case %d." % (t_num) + \
                                    " y returned: %g   correct: %g" % (y, my_y))
                
                    self.assertEquals(z, my_z, \
                                      msg="Failed test case %d." % (t_num) + \
                                    " z returned: %g   correct: %g" % (z, my_z))
                
                
                
                
# Return the test suit for this module
################################################################################
def suite(label='fast'):

    
    testsuite = unittest.TestSuite()

    Cart2DPointTestCase.label=label    
    testsuite.addTest(unittest.makeSuite(Cart2DPointTestCase, prefix='test'))
    
    Cart3DPointTestCase.label=label    
    testsuite.addTest(unittest.makeSuite(Cart3DPointTestCase, prefix='test'))
    
    return testsuite

################################################################################

if __name__ == '__main__':

    unittest.main(defaultTest='suite')
        