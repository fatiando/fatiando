"""
Test suite for the fatiando package.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 29-Mar-2010'

import unittest

import fatiando.directmodels.tests
import fatiando.geoinv.tests
import fatiando.math.tests
import fatiando.utils.tests


def suite(label='fast'):

    suite = unittest.TestSuite()

    suite.addTest(fatiando.directmodels.tests.suite(label))
    suite.addTest(fatiando.geoinv.tests.suite(label))
    suite.addTest(fatiando.math.tests.suite(label))
    suite.addTest(fatiando.utils.tests.suite(label))

    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')