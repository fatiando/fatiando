"""
blaSetup file for building the C-coded extention modules for slooT.
"""
# Created on 01-Mar-2010
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__revision__ = '$Revision: -1 $'
__date__ = '$Date: $'

from distutils.core import setup, Extension


prismgrav_mod = Extension('sloot/_prismgrav',
                          sources=['c/prismgrav_wrap.c', 'c/prismgrav.c'],
                          libraries=['m'])

linalg_mod = Extension('sloot/_linalg',
                       sources=['c/linalg_wrap.c', 'c/linalg.c'],
                       libraries=['m'])

setup (name = 'sloot-c-extention-mods',
       version = '0.1',
       author      = "Leonardo Uieda",
       description = "C-coded extention modules for inversion tools (slooT).",
       ext_modules = [prismgrav_mod, linalg_mod],
       py_modules = ["prismgrav", "linalg"],
       )
