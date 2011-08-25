# Copyright 2010 The Fatiando a Terra Development Team
#
# This file is part of Fatiando a Terra.
#
# Fatiando a Terra is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Fatiando a Terra is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Fatiando a Terra.  If not, see <http://www.gnu.org/licenses/>.
"""
Geophysical direct and inverse modeling. 
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = '02-Apr-2010'
__version__ = '0.0.1'
__all__ = ['potential', 'seismic', 'inversion', 'gridder', 'mesher', 'vis',
           'stats', 'utils']

# Create a default NullHandler so that logging is only enabled explicitly
################################################################################ 
import logging
class NullHandler(logging.Handler):
    """
    Default null handler so that logging is only done when explicitly asked for.
    """
    def emit(self, record):
        pass
default_log_handler = NullHandler()
################################################################################
