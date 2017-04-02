.. title:: Fatiando a Terra: modeling and inversion

.. raw:: html

    <div class="container-fluid banner">
        <div class="container">
            <div class="row">
                <div class="col-lg-4 col-sm-3 col-xs-1">
                </div>
                <div class="col-lg-4 col-sm-6 col-xs-10">
                    <img class="banner-logo center-block" src="_static/fatiando-logo-no-background.png">
                </div>
                <div class="col-lg-4 col-sm-3 col-xs-1">
                </div>
            </div>

            <div class="row site-title">
                <div class="col-lg-3 col-sm-2">
                </div>
                <div class="col-lg-6 col-sm-8">
                    <img src="_static/fatiando-banner.png" width="100%">
                </div>
                <div class="col-lg-3 col-sm-2">
                </div>
            </div>

            <div class="row">
                <div class="col-md-1">
                </div>
                <div class="col-md-10">
                    <p class="text-center site-slogan">
                    Open-source Python library for modeling and inversion in geophysics.
                    </p>

                    <p class="text-center site-description">
                    Our goal is provide a comprehensive and extensible framework
                    for geophysical data analysis and the development of new
                    methodologies.
                    </p>
                </div>
                <div class="col-md-1">
                </div>
            </div>
        </div>
    </div>


.. raw:: html

    <div class="container">
    <div class="row">
    <div class="col-md-1">
    </div>
    <div class="col-md-10">


.. raw:: html

    <div class="home-row">

        <h2>Overview</h2>

        <div class="row">
            <div class="col-md-6 home-overview">
                <h3><a href="api.html#module-fatiando.gravmag">Gravity and magnetics</a></h3>
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
            <div class="col-md-6 home-overview">
                <h3><a href="api.html#module-fatiando.seismic">Seismology and seismics</a></h3>
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
            <div class="col-md-6 home-overview">
                <h3><a href="api.html#module-fatiando.gridder">Grid generation and manipulation</a></h3>
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
            <div class="col-md-6 home-overview">
                <h3><a href="api.html#module-fatiando.mesher">Geometric objects and meshes</a></h3>
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

        <div class="row">
            <div class="col-md-6 home-overview">
                <h3><a href="api.html#module-fatiando.datasets">Datasets and I/O</a></h3>
                <p>
                Experiment with our packaged test datasets or load your data
                with some of our functions for input and output.
                </p>
                <em>
                Test gravity and magnetic data, load data from Surfer ASCII
                grids, generate data and models from images.
                </em>
            </div>
            <div class="col-md-6 home-overview">
                <h3><a href="api.html#module-fatiando.inversion">Inverse problems</a></h3>
                <p>
                Build your own inversions by implementing the bare minimum. We
                provide standard regularization and optimization.
                </p>
                <em>
                Classes for least-squares problems, Tikhonov regularization,
                gradient-descent optimization, and more.
                </em>
            </div>
        </div>

    </div>


.. raw:: html

    <div class="home-row">

    <h2>Get started</h2>

See the :ref:`install instructions <install>` to set up your computer and
install Fatiando.

Once you have everything installed,
take a look at the :ref:`Documentation <docs>`
for a detailed tour of the library.
You can also browse the :ref:`Gallery <gallery>` and :ref:`Cookbook <cookbook>`
for examples of what Fatiando can do.

Want more inspiration? Check out how Fatiando is being used in the
:ref:`Use cases <use_cases>` page.

**Stay informed**: Sign up to our `mailing list`_ to keep up-to-date with new
releases and events and give your feedback.


.. raw:: html

    </div>

.. raw:: html

    <div class="home-row">

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

    <div class="home-row">

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

    <div class="home-row">

    <h2>Support</h2>

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

    <div class="home-row">

    <h2>Watch</h2>

    <div class="text-center">

Watch an introduction to what Fatiando is all about in this presentation from
`Scipy 2013 <http://www.leouieda.com/talks/scipy2013.html>`__.

.. raw:: html

    </div>

    <div class="row home-row-video">
        <div class="col-md-2">
        </div>
        <div class="col-md-8">
            <div class="embed-responsive embed-responsive-16by9">
                <iframe class="embed-responsive-item"
                        src="https://www.youtube.com/embed/Ec38h1oB8cc"
                        frameborder="0"
                        allowfullscreen>
                </iframe>
            </div>
        </div>
        <div class="col-md-2">
        </div>
    </div>

.. raw:: html

    </div>

.. raw:: html

    </div>  <!-- col -->
    <div class="col-md-1">
    </div>
    </div>  <!-- row -->
    </div>  <!-- container -->


.. toctree::
    :maxdepth: 2
    :hidden:

    news.rst
    license.rst
    cite.rst
    use_cases.rst
    changelog.rst
    install.rst
    api.rst
    develop.rst
    contributors.rst
    cookbook.rst
    gallery/index.rst
    docs.rst
