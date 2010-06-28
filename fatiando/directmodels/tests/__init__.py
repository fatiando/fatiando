"""
Test suite for the fatiando.directmodels package.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 29-Mar-2010'

import unittest

import fatiando.directmodels.seismo.tests
import fatiando.directmodels.gravity.tests
import fatiando.directmodels.heat.tests


def suite(label='fast'):

    testsuite = unittest.TestSuite()

    testsuite.addTest(fatiando.directmodels.seismo.tests.suite(label))
    testsuite.addTest(fatiando.directmodels.gravity.tests.suite(label))
    testsuite.addTest(fatiando.directmodels.heat.tests.suite(label))

    return testsuite


if __name__ == '__main__':
    
    unittest.main(defaultTest='suite')