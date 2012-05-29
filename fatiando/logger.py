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

----

"""
import inspect
import os
import sys
import logging
import time

import numpy


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
    
    * log : `logging.Logger`
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
    
    * log : `logging.Logger`
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
    
    * comment : str
        Character inserted at the beginning of each line. Use this to make a
        message that can be inserted in source code files as comments.
        
    Returns:
    
    * msg : str
        The header message

    """
    from fatiando import version
    csfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'changeset.txt')
    if os.path.exists(csfile):
        with open(csfile) as f:
            changeset = f.readline()
    else:
        changeset = "Unknown"
    msg = '\n'.join(
        ["########################################",
         "%sFatiando a Terra:" % (comment),
         "%s  date: %s" % (comment, time.asctime()),
         "%s  version: %s" % (comment, version),
         "%s  changeset: %s" % (comment, changeset),
         "########################################"])
    return msg
