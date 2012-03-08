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

This module is basically a wrapper around Pythons `logging` module.

In a module, use a logger without any handlers so that it only logs if a script
wants to log::

    >>> # in fatiando.package.module.py
    >>> from fatiando import logger
    >>> def myfunc():
    ...     log = logger.dummy('fatiando.package.module.myfunc')
    ...     log.info("Only logs if a script calls logger.get")

Then it can be used in a script::

    >>> myfunc()
    >>> # Nothing happens
    >>> import sys
    >>> # Get a logger to stdout
    >>> log = logger.get(stream=sys.stdout)
    >>> myfunc()
    Only logs if a script calls logger.get
    >>> log.info("This is an info msg printed to stdout from the script")
    This is an info msg printed to stdout from the script
    >>> log.debug("This is a debug msg NOT printed")
    >>> log = logger.tofile(log, 'mylogfile.log')
    >>> log.warning('Warning printed to both stdout and log file')
    Warning printed to both stdout and log file
    >>> log.error('and this is an Error message.')
    and this is an Error message.

.. note:: Importing this module assigns a `logging.NullHandler` to the base
    logger of `fatiando`, whose name is ``'fatiando'``. This way, log messages
    are only printed if a script calls :func:`fatiando.logger.get` or assigns a
    Handler to it.

**Ignore the next few lines**: This is needed since `doctest` keeps all tests in
the same namespace. So I have to get rid of the handlers before running the
other tests in this module.

    >>> for h in log.handlers:
    ...     log.removeHandler(h)


:author: Leonardo Uieda (leouieda@gmail.com)
:date: Created 14-Sep-2011
:license: GNU Lesser General Public License v3 (http://www.gnu.org/licenses/)

----

"""

import sys
import logging
import time

import numpy

import fatiando


# Add the NullHandler to the root logger for fatiando so that nothing is printed
# until logger.get is called
logging.getLogger('fatiando').addHandler(logging.NullHandler())
    

def get(level=logging.INFO, stream=sys.stderr):
    """
    Create a logger using the default settings for Fatiando.

    Parameters:
    
    * level : int
        Default to `logging.INFO`. See `logging` module
    * stream : file
        A stream to log to. Default to `sys.stderr`        
        
    Returns:
    
    * log : `logging.Logger`
        A logger with the level and name set
    
    """
    logger = logging.getLogger('fatiando')
    handler = logging.StreamHandler(stream)
    fmt = '%(message)s'
    handler.setFormatter(logging.Formatter(fmt))
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger

def tofile(logger, fname, level=logging.DEBUG):
    r"""
    Enable logging to a file.

    Will enable file logging to the given *logger*.

    Parameters:

    * logger : `logging.Logger`
        A logger, as returned by :func:`fatiando.logger.get`
    * fname :  str
        Log file name
    * level : int
        The logging level. Default to `logging.DEBUG`. See `logging` module

    Returns:
    
    * log : a `logging.Logger` object
        *logger* with added FileHandler
        
    Examples:

        >>> import logging
        >>> # Need to mock the FileHandler so that it works with StringIO
        >>> from StringIO import StringIO
        >>> f = StringIO()
        >>> logging.FileHandler = lambda f, mode: logging.StreamHandler(f)
        >>> # Now for the actual logger example!
        >>> import sys
        >>> from fatiando import logger
        >>> log = logger.tofile(logger.get(stream=sys.stdout), f,
        ...                     level=logging.DEBUG)
        >>> log.debug("logged to file but not stdout!")
        >>> print f.getvalue().strip()
        DEBUG:fatiando: logged to file but not stdout!
                
    """
    fhandler = logging.FileHandler(fname, 'w')
    fmt = '%(levelname)s:%(name)s: %(message)s'
    fhandler.setFormatter(logging.Formatter(fmt))
    fhandler.setLevel(level)
    logger.addHandler(fhandler)
    return logger

def dummy(name='fatiando'):
    """
    Get a logger without any handlers.

    For use inside modules.
    
    Parameters:
    
    * name : str
        Name of the logger. Use the module name as *name* (including the full
        package hierarchy)
    
    Returns:
    
    * logger
        A logger without any handlers so that it only prints when
        :func:`fatiando.logger.get` or :func:`fatiando.logger.tofile` are called

    Examples:

        >>> # in fatiando.mymod.py
        >>> from fatiando import logger
        >>> def myfunc():
        ...     log = logger.dummy('fatiando.mymod.myfunc')
        ...     log.info("Not logged unless a script wants it to")
        >>> myfunc()
        >>>

    """
    return logging.getLogger(name)

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
    lines = ["%sFatiando a Terra:" % (comment),
             "%s  version: %s" % (comment, fatiando.__version__),
             "%s  date: %s" % (comment, time.asctime())]
    return '\n'.join(lines)
    
def _test():
    import doctest
    doctest.testmod()
    print "doctest finished"

if __name__ == '__main__':
    _test()
