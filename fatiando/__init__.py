"""
The ``fatiando`` package contains all the subpackages and modules required for
most tasks.
"""
from ._version import get_versions
__version__ = get_versions()['version']
__commit__ = get_versions()['full']
del get_versions
