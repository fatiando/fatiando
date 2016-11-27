.. title:: Fatiando a Terra: modeling and inversion

.. raw:: html

    <div class="row" style="margin-top: 60px">
        <div class="col-md-2">
        </div>
        <div class="col-md-8">
            <img src="_static/fatiando-banner.png" width="100%"
                style="margin-bottom: 50px;"></img>
            </div>
        <div class="col-md-2">
        </div>
    </div>

    <div class="text-center" style="font-size: 16pt; margin-bottom: 50px;">


An open-source Python library for modeling and inversion in geophysics.

Our goal is provide a comprehensive and extensible framework
for geophysical data analysis and the development of new methodologies.

.. raw:: html

    </div>

.. raw:: html

    <div class="row alert alert-info">

    <div class="col-md-4">

    <h3 class="text-center">Research</h3>

Make your research more **reproducible** by writing a Python script or
`Jupyter notebook`_ instead of clicking through complicated menus.

.. raw:: html

    </div>

    <div class="col-md-4">

    <h3 class="text-center">Development</h3>

Don't start from scratch! Build upon the existing tools in Fatiando to develop
new methods.

.. raw:: html

    </div>

    <div class="col-md-4">

    <h3 class="text-center">Teaching</h3>

Combine Fatiando with the `Jupyter notebook`_ to make rich, interactive
documents. Great for teaching fundamental concepts of geophysics!

.. raw:: html

    </div>

.. raw:: html

    </div> <!-- Row -->

.. _Jupyter notebook: https://jupyter.org/
.. _Python: https://www.python.org/
.. _matplotlib: http://matplotlib.org/
.. _Mayavi: http://code.enthought.com/projects/mayavi/
.. _Numpy: http://www.numpy.org/
.. _Scipy: http://scipy.org/
.. _Cython: http://www.cython.org/

.. raw:: html

    <div class="row">

    <div class="col-md-12">

.. raw:: html

    <h2>Overview</h2>

    <div class="col-md-12">
    <div class="row">
        <div class="col-md-6">
            <h3><a href="api/gravmag.html">Gravity and magnetics</a></h3>
            <p>
            Modeling, inversion, and processing for potential field methods.
            </p>
            <em>
            3D forward modeling with prisms,
            polygonal prisms, spheres, and tesseroids.
            Handles the potential, acceleration,
            gradient tensor, magnetic induction, total field magnetic anomaly.
            </em>
        </div>
        <div class="col-md-6">
            <h3><a href="api/seismic.html">Seismology and Seismics</a></h3>
            <p>
            Simple modeling functions for seismics and seismology.
            </p>
            <em>
            Toy problems for: Cartesian straight-ray tomography,
            VSP, epicenter estimation.
            Experimental finite-difference wave propagation.
            </em>
        </div>
    </div>
    <div class="row">
        <div class="col-md-6">
            <h3><a href="api/gridder.html">Grid manipulation</a></h3>
            <p>
            Functions for generating and operating on regular grids and data
            that is on a map.
            </p>
            <em>
            Generate regular grids and point scatters.
            Cut grids and extract profiles.
            Interpolate irregular data.
            </em>
        </div>
        <div class="col-md-6">
            <h3><a href="api/mesher.html">Geometric objects and meshes</a></h3>
            <p>
            Classes that represent geometric objects (points, prisms, polygons,
            tesseroids) and meshes (regular prism mesh, points on a grid).
            </p>
            <em>
            Standard classes used in all of Fatiando.
            Efficient classes for meshes that save storage and behave as
            <a href="https://docs.python.org/2/library/stdtypes.html#iterator-types">iterators</a>.
            </em>
        </div>
    </div>
    </div>

.. raw:: html

    </div>

.. raw:: html

    </div><!-- Row -->

.. raw:: html

    <div class="row">

    <div class="col-md-6">
    <h2>Get started</h2>

See the :ref:`install instructions <install>` to set up your computer and
install Fatiando.

Once you have everything installed,
take a look at the :ref:`Documentation <docs>`
for a detailed tour of the library.
You can also browse the :ref:`Gallery <gallery>` and :ref:`Cookbook <cookbook>`
for examples of what Fatiando can do.

.. raw:: html

    <div class="alert alert-success">
    <h4><strong>Stay informed</strong></h4>

Sign up to our `mailing list`_ to keep up-to-date with new releases and
events and give your feedback.

.. raw:: html

    </div> <!-- Alert bubble -->

.. raw:: html

    </div>

.. raw:: html

    <div class="col-md-6">
    <h2>Get help</h2>

There are many ways to contact us:

* Write to our `mailing list`_.
* Join us on our open `Gitter chat room`_.
* Report bugs through `Github`_.

If you come across a bug, please include in your message: your operating
system, Python version, Fatiando version, code that generated the error, the
full error message.

.. raw:: html

    </div>

.. raw:: html

    </div><!-- Row -->

.. raw:: html

    <div class="row">

.. raw:: html

    <div class="col-md-6">
    <h2>Cite us</h2>

Fatiando is research software **made by scientists**.
Your citations help us justify the effort that goes into building and
maintaining Fatiando.

**TL;DR**: If you just want to copy and paste something, include the following
in your Methods or Acknowledgments:

    The results presented here were obtained with the help of the open-source
    software Fatiando a Terra by Uieda et al. (2013).

and the reference:

    Uieda, L., V. C. Oliveira Jr, and V. C. F. Barbosa (2013), Modeling the
    Earth with Fatiando a Terra, Proceedings of the 12th Python in Science
    Conference, pp. 91 - 98.

See :ref:`Citing <cite>` for more details on how to cite Fatiando in your
publications.


.. raw:: html

    </div>

.. raw:: html

    <div class="col-md-6">
    <h2>Contribute</h2>

**Feedback**: Send us your bug reports, feature requests, spelling corrections,
usage examples, etc. We love to hear what the community thinks!

**Documentation**: We need a lot of help improving our documentation. You can
report typos, suggest new sections and improvements, and anything that you
think would make the docs better in any way.

**Code**: If you  want to get involved with the code,
take a look at our :ref:`Developer Guide <develop>`.
All source code development is done in the open on the Github repository
`fatiando/fatiando <https://github.com/fatiando/fatiando>`__.
A good place to start is with our `curated list of low-hanging fruit
<https://github.com/fatiando/fatiando/issues?q=is%3Aissue+is%3Aopen+label%3A%22low-hanging+fruit%22>`__.

If you want to help but are not sure how, ask on the `Gitter chat room`_ and
we'll help you get started.
**Don't be afraid to ask for help!**

.. _mailing list: https://groups.google.com/d/forum/fatiando
.. _issues on Github: https://github.com/fatiando/fatiando/issues?q=is%3Aopen
.. _Github: https://github.com/fatiando/fatiando/issues?q=is%3Aopen
.. _+Fatiando a Terra: https://plus.google.com/+FatiandoOrg
.. _Gitter chat room: https://gitter.im/fatiando/fatiando

.. raw:: html

    </div>

.. raw:: html

    </div><!-- Row -->

.. raw:: html

    <div class="row">

.. raw:: html

    <div class="col-md-6">
    <h2>Announcements</h2>

* **October 2016**: Fatiando a Terra v0.5 was released!
  This version introduces some new features, breaking changes, and starts a
  major refactoring of the library that will span the next few releases.
  See what is new in this released in the :ref:`Changelog <changelog-0.5>`.

* **April 2016**: Fatiando a Terra v0.4 was released! See what is new in this
  released in the :ref:`Changelog <changelog-0.4>`.

* **October 2014**: Fatiando was featured on volume 89 of the bulletin of the
  Brazilian Geophysical Society (SBGf). Read it on page 13 of the `PDF file
  <http://sbgfisica.org/portal/images/stories/Arquivos/Boletim_89-2014.pdf>`__
  (in Portuguese).

* **July 2014**: We presented a poster at Scipy 2014 about the
  ``fatiando.inversion`` package. See the
  `Github repo <https://github.com/leouieda/scipy2014>`__ for the poster and
  source code behind it.

Read :ref:`all announcements <news>`.

.. raw:: html

    </div>

.. raw:: html

    <div class="col-md-6">
    <h2>Watch</h2>

Watch an introduction to what Fatiando is all about in this presentation from
`Scipy 2013 <https://github.com/leouieda/scipy2013>`__.

.. raw:: html

    <div class="responsive-embed">
        <iframe width="100%" height="350"
        src="https://www.youtube.com/embed/Ec38h1oB8cc" frameborder="0"
        allowfullscreen></iframe>
    </div>

.. raw:: html

    </div>

.. raw:: html

    </div> <!-- Row -->

.. toctree::
    :maxdepth: 2
    :hidden:

    news.rst
    license.rst
    cite.rst
    changelog.rst
    install.rst
    api/fatiando.rst
    develop.rst
    contributors.rst
    cookbook.rst
    gallery/index.rst
    docs.rst
