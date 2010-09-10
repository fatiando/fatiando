"""
Test suite for the fatiando.gravity package.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 10-Sep-2010'

import unittest


def suite(label='fast'):

    testsuite = unittest.TestSuite()
    
    return testsuite


if __name__ == '__main__':
    
    unittest.main(defaultTest='suite')