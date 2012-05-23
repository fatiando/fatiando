================
Fatiando a Terra
================

Fatiando a Terra provides command-line tools and an API for a variety of
geophysical modeling applications. It makes heavy use of Numpy for linear
algebra operations, provides a high level interface to visualise your data and
results using Matplotlib for 2D and Mayavi2 for 3D. Optimized code is written in
Fortran and C and wrapped to Python using Numpy's f2py.

For more information and online documentation, visit http://www.fatiando.org


Downloading and Installing
==========================

Fatiando requires:

* numpy
* scipy
* matplotlib
* PIL
* mayavi

All of these can be easily found in most GNU/Linux distros.
You will also need the Python header files
(called python-dev in Debian and Ubuntu)
and a C compiler (GCC comes with almost every distro).

After you've installed the dependencies
you can proceed to install Fatiando using pip
(remember to install pip before if don't have it)::

    pip install fatiando

That's it!
Alternatively, you can run the setup.py script::

    python setup.py install


The team
========

Fatiando is being developed by a group of geophysics graduates from the
Universidade de Sao Paulo and the Observatorio Nacional in Brazil. Work done
here is part of some Masters and Phd projects.

See http://www.fatiando.org/people for a list of people involved.

License
=======

Fatiando a Terra is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
A copy of this license is provided in file LICENSE.txt
