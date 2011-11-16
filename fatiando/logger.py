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
Logging utilities for fatiando.

Use it in your script::

    from fatiando import logger
    # Get a logger to stderr
    log = logger.get()
    log.info("This is an info msg printed to stderr")
    logger.tofile('mylog.txt')
    log.warning('Warning printed to both stderr and log file')
    # Log a header with useful provenance information
    log.info(logger.header())

In a module, use a logger with a null handler so that it only logs if the script
wants to log::

    # in fatiando.package.module.py
    import fatiando.logger

    log = fatiando.logger.dummy()

    def myfunc(...):
        log.info("From myfunc. Only logs if a script calls logger.get")


"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 14-Sep-2011'

import logging
import time

import numpy

import fatiando


class NullHandler(logging.Handler):
    """
    Default null handler so that logging is only done when explicitly asked for.
    """
    def emit(self, record):
        pass

def get(level=logging.DEBUG):
    """
    Get a logger to ``stderr``.

    (Adds a stream handler to the base logger ``'fatiando'``)

    Parameters:
    * level
        The logging level. Default to ``logging.DEBUG``. See ``logging`` module
    Returns:
    * a logger object

    """
    logger = logging.getLogger('fatiando')
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def tofile(fname, level=logging.DEBUG):
    """
    Enable logging to a file.

    If called after fatiando.logger.get, will enable file logging to the
    returned logger.

    (Adds a file handler to the base logger ``'fatiando'``)

    Parameters:
    * fname
        Log file name
    * level
        The logging level. Default to ``logging.DEBUG``. See ``logging`` module
    Returns:
    * a logger object

    """
    logger = logging.getLogger('fatiando')
    fhandler = logging.FileHandler(fname, 'w')
    fhandler.setFormatter(logging.Formatter())
    logger.addHandler(fhandler)
    logger.setLevel(level)
    return logger

def dummy():
    """
    Get a logger for use inside a module.
    Returns:
    * logger
        A logger with a NullHandler so that it only prints when
        :func:`fatiando.logger.get` or :func:`fatiando.logger.tofile` are called

    """
    logger = logging.getLogger('fatiando.utils')
    logger.addHandler(NullHandler())
    return logger

def header(comment=''):
    """
    Generate a header message with the current version, changeset information
    and date.

    Parameters:
    * comment
        Character inserted at the beginning of each line. Use this to make a
        message that can be inserted in source code files as comments.
    Returns:
    * msg
        string with the header message

    """
    lines = ["%sFatiando a Terra:\n" % (comment),
             "%s  version: %s\n" % (comment, fatiando.__version__),
             "%s  date: %s\n" % (comment, time.asctime()),
             "%s  changeset: %s\n" % (comment, fatiando.__changeset__)
            ]
    return ''.join(lines)
