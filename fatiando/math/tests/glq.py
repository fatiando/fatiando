"""
Unit tests for fatiando.math.lu module
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 07-May-2010'


import unittest
from math import sqrt

from fatiando.math.glq import nodes, weights, scale


class GLQTestCase(unittest.TestCase):
    """
    Mother class for nodes, weights and scale test cases.
    Does the test setup for all of them.
    """
    
    def setUp(self):
        """
        Test setup. Define the known values.
        """        

        self.known_nodes = ( \
                         (2, [sqrt(3)/3, -sqrt(3)/3]), \
                         (3, [sqrt(15)/5, 0, -sqrt(15)/5]), \
                         (4, [sqrt(525+70*sqrt(30))/35, \
                              sqrt(525-70*sqrt(30))/35, \
                             -sqrt(525-70*sqrt(30))/35, \
                             -sqrt(525+70*sqrt(30))/35]), \
                         (5, [sqrt(245+14*sqrt(70))/21, \
                              sqrt(245-14*sqrt(70))/21, \
                              0, \
                              -sqrt(245-14*sqrt(70))/21, \
                              -sqrt(245+14*sqrt(70))/21]) )
        
        self.known_weights = ( \
                     (2, [1, 1]), \
                     (3, [5.0/9, 8.0/9, 5.0/9]), \
                     (4, [(18-sqrt(30))/36.0, (18+sqrt(30))/36.0, \
                          (18+sqrt(30))/36.0, (18-sqrt(30))/36.0]), \
                     (5, [(322-13*sqrt(70))/900.0, (322+13*sqrt(70))/900.0, \
                          128.0/225, \
                          (322+13*sqrt(70))/900.0, (322-13*sqrt(70))/900.0]) )
        
        
        self.known_scaled = ( \
            (2, -2.54, 14.9, [8.72*sqrt(3)/3 + 6.18, -8.72*sqrt(3)/3 + 6.18]), \
            (3, -2.54, 14.9, [8.72*sqrt(15)/5 + 6.18, \
                              6.18, \
                              -8.72*sqrt(15)/5 + 6.18]), \
            (4, -2.54, 14.9, [8.72*sqrt(525+70*sqrt(30))/35 + 6.18, \
                              8.72*sqrt(525-70*sqrt(30))/35 + 6.18, \
                              -8.72*sqrt(525-70*sqrt(30))/35 + 6.18, \
                              -8.72*sqrt(525+70*sqrt(30))/35 + 6.18]), \
            (5, -2.54, 14.9, [8.72*sqrt(245+14*sqrt(70))/21 + 6.18, \
                              8.72*sqrt(245-14*sqrt(70))/21 + 6.18, \
                              6.18, \
                              -8.72*sqrt(245-14*sqrt(70))/21 + 6.18, \
                              -8.72*sqrt(245+14*sqrt(70))/21 + 6.18]), \
            (2, 125.6, 234.84, [54.62*sqrt(3)/3 + 180.22, \
                                -54.62*sqrt(3)/3 + 180.22]), \
            (3, 125.6, 234.84, [54.62*sqrt(15)/5 + 180.22, \
                                180.22, \
                                -54.62*sqrt(15)/5 + 180.22]), \
            (4, 125.6, 234.84, [54.62*sqrt(525+70*sqrt(30))/35 + 180.22, \
                                54.62*sqrt(525-70*sqrt(30))/35 + 180.22, \
                                -54.62*sqrt(525-70*sqrt(30))/35 + 180.22, \
                                -54.62*sqrt(525+70*sqrt(30))/35 + 180.22]), \
            (5, 125.6, 234.84, [54.62*sqrt(245+14*sqrt(70))/21 + 180.22, \
                                54.62*sqrt(245-14*sqrt(70))/21 + 180.22, \
                                180.22, \
                                -54.62*sqrt(245-14*sqrt(70))/21 + 180.22, \
                                -54.62*sqrt(245+14*sqrt(70))/21 + 180.22]), \
            (2, 3.5, -12.4, [-7.95*sqrt(3)/3 - 4.45, 7.95*sqrt(3)/3 - 4.45]), \
            (3, 3.5, -12.4, [-7.95*sqrt(15)/5 - 4.45, \
                             -4.45, \
                             7.95*sqrt(15)/5 - 4.45]), \
            (4, 3.5, -12.4, [-7.95*sqrt(525+70*sqrt(30))/35 - 4.45, \
                             -7.95*sqrt(525-70*sqrt(30))/35 - 4.45, \
                             7.95*sqrt(525-70*sqrt(30))/35 - 4.45, \
                             7.95*sqrt(525+70*sqrt(30))/35 - 4.45]), \
            (5, 3.5, -12.4, [-7.95*sqrt(245+14*sqrt(70))/21 - 4.45, \
                             -7.95*sqrt(245-14*sqrt(70))/21 - 4.45, \
                             -4.45, \
                             7.95*sqrt(245-14*sqrt(70))/21 - 4.45, \
                             7.95*sqrt(245+14*sqrt(70))/21 - 4.45])    )
        
    def mock_nodes(self, order):
        """
        Return the known nodes values.
        """
        known_nodes = { \
                         2:[sqrt(3)/3, -sqrt(3)/3], \
                         3:[sqrt(15)/5, 0, -sqrt(15)/5], \
                         4:[sqrt(525+70*sqrt(30))/35, \
                              sqrt(525-70*sqrt(30))/35, \
                             -sqrt(525-70*sqrt(30))/35, \
                             -sqrt(525+70*sqrt(30))/35], \
                         5:[sqrt(245+14*sqrt(70))/21, \
                              sqrt(245-14*sqrt(70))/21, \
                              0, \
                              -sqrt(245-14*sqrt(70))/21, \
                              -sqrt(245+14*sqrt(70))/21] }
        
        return known_nodes[order]


class NodesTestCase(GLQTestCase):
    """
    Test the nodes function in module fatiando.math.glq
    """

    label = 'fast'
            
    def test_known_values(self):
        "glq.nodes return the correct results given test data"
        
        t_num = 1
        for order, nodes_known in self.known_nodes:
            
            nodes_my = nodes(order)
            
            # Check if the order is correct
            self.assertEquals(order, len(nodes_my), \
                msg="Failed test case %d. Nodes with wrong order." % (t_num) + \
                    " my: %d   correct: %d" % (order, len(nodes_my)))
            
            for i in range(order):
                
                self.assertAlmostEquals(nodes_my[i], nodes_known[i], \
                                        places=10, \
                    msg="Failed test case %d." % (t_num) + \
                    " my: %g   correct: %g" % (nodes_my[i], nodes_known[i]))
                
            t_num += 1
        

class WeightsTestCase(GLQTestCase):    
    """
    Test the weights function in module fatiando.math.glq
    """

    label = 'fast'
                           
    def test_known_values(self):
        "glq.weights return the correct results given test data"
                
        for t_num in range(len(self.known_weights)):
                       
                                         
            order, weights_known = self.known_weights[t_num]
            
            nodes_known = self.mock_nodes(order)            
            
            weights_my = weights(nodes_known)
                        
            # Check if the order is correct
            self.assertEquals(order, len(weights_my), \
                msg="Failed test case %d." % (t_num) + \
                    " Weights with wrong order." + \
                    " my: %d   correct: %d" % (order, len(weights_my)))
            
            for i in range(order):
                
                self.assertAlmostEquals(weights_my[i], weights_known[i], \
                                        places=10, \
                    msg="Failed test case %d." % (t_num) + \
                    " my: %g   correct: %g" % (weights_my[i], weights_known[i]))
                
            t_num += 1
        
        
        
class ScaleTestCase(GLQTestCase):    
    """
    Test the scale function in module fatiando.math.glq
    """    

    label = 'fast'
                           
    def test_known_values(self):
        "glq.scale return the correct results given test data"
                
        for t_num in range(len(self.known_scaled)):
            
            order, a, b, scaled_known = self.known_scaled[t_num]
            
            nodes_known = self.mock_nodes(order)
            
            scaled_my = scale(a, b, nodes_known)
                        
            # Check if the order is correct
            self.assertEquals(order, len(scaled_my), \
                msg="Failed test case %d." % (t_num) + \
                    " Scaled nodes with wrong order." + \
                    " my: %d   correct: %d" % (order, len(scaled_my)))
            
            for i in range(order):
                
                self.assertAlmostEquals(scaled_my[i], scaled_known[i], \
                                        places=10, \
                    msg="Failed test case %d." % (t_num) + \
                    " my: %g   correct: %g" % (scaled_my[i], scaled_known[i]))
                
            t_num += 1
    
        
                
                
# Return the test suit for this module
################################################################################
def suite(label='fast'):
    
    testsuite = unittest.TestSuite()

    NodesTestCase.label=label    
    testsuite.addTest(unittest.makeSuite(NodesTestCase, prefix='test'))
    
    WeightsTestCase.label = label
    testsuite.addTest(unittest.makeSuite(WeightsTestCase, prefix='test'))
    
    ScaleTestCase.label = label
    testsuite.addTest(unittest.makeSuite(ScaleTestCase, prefix='test'))

    return testsuite

################################################################################

if __name__ == '__main__':

    unittest.main(defaultTest='suite')
        