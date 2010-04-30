"""
Test suite for the fatiando.directmodels.seismo package.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 29-Mar-2010'

import unittest

import fatiando.directmodels.seismo.tests.simple


def suite(label='fast'):

    suite = unittest.TestSuite()

    suite.addTest(fatiando.directmodels.seismo.tests.simple.suite(label))

    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')