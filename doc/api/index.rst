.. _api:

API Reference
=============

The `Application Programming Interface
<http://en.wikipedia.org/wiki/Application_programming_interface>`_ (API) is
the standard (and most powerful) way to interact with Fatiando a Terra.
The `fatiando` API is divided into packages and modules.
Different geophysical methods are represented as different packages (like
:ref:`Potential Fields <potential>`, :ref:`Geothermics <heat>`, etc.).
General purpose inverse problem solvers, as well as general purpose regularizing
functions, are in a :ref:`package of their own <inversion>`.
The remaining packages and modules offer other operations, such as
:ref:`plotting <vis>`, :ref:`gridding <gridder>`, :ref:`meshing <mesher>`,
:ref:`logging <logger>`, :ref:`interfacing with the user <ui>`, and
:ref:`general utilities <utils>`.

For a more detailed description of the usage and capabilities of individual
API components, see API reference bellow:

.. toctree::

    fatiando.rst
