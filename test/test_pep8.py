import unittest
import pep8


class TestCodeFormat(unittest.TestCase):

    def test_pep8_conformance(self):
        """all packages, tests, and cookbook conform to PEP8"""
        pep8style = pep8.StyleGuide(quiet=True)
        result = pep8style.check_files(['fatiando', 'test', 'setup.py',
                                        'cookbook'])
        self.assertEqual(result.total_errors, 0,
                         "Found code style errors (and warnings).")
