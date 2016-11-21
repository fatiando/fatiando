.. _develop:

Developer guide
===============

Getting started
---------------

The first thing you'll need is a `Github <https://github.com/>`__ account.
You can sign up for free.
If you are an academic or student,
`request a free pro-account <https://education.github.com/>`__ to get access to
private repositories.

.. tip::

    The easiest way to contribute is to
    `submit issues and bug reports
    <https://github.com/fatiando/fatiando/issues>`__.
    Feature requests, typo fixes, suggestions for the documentation, it's all
    welcome!

.. note::

    If you are new to **version control** (or don't know what that means),
    start with the `Software Carpentry lessons on version control with Git
    <http://software-carpentry.org/>`__.
    After you've gone through those, the
    Github `help page <https://help.github.com/>`__
    and `guides <https://guides.github.com/>`__
    are great resources for finding your way around the quirks of git and
    Github.

All of the "official" code for Fatiando lives in the
`fatiando/fatiando repository <https://github.com/fatiando/fatiando>`__
(the first ``fatiando`` refers to the
`fatiando Github account <https://github.com/fatiando>`__).
The *master* branch of the repository contains the latest code that is
**stable** (should have tested and working code).
Code that is in `other branches
<https://github.com/fatiando/fatiando/branches>`__
are things that are under development by the
`main developers <https://github.com/fatiando/fatiando/graphs/contributors>`__.

To contribute some code/fix/documentation, start by forking
`fatiando/fatiando <https://github.com/fatiando/fatiando/>`__
(click the "Fork" button).
This will grab a complete copy of the code repository and add it to your
account.
This "fork" is isolated from the main repository, so you don't have to worry
about "breaking" anything.
Go nuts!
If you break your fork beyond repair you can always delete it and make a new
fork.
Beware that you will lose **everything** in your fork if you delete it.

Once you have your fork, clone a copy to your computer::

    git clone https://github.com/USERNAME/fatiando.git

.. note::

    Not sure what to work on? Have a look at the
    `open issues <https://github.com/fatiando/fatiando/issues>`__
    and pick one that you find interesting.
    **Please leave a comment on the issue if you are going to work on it**.
    This helps us keep track of who is doing what and avoid duplicated work.
    We are trying to curate a
    `list of "low-hanging fruit"
    <https://github.com/fatiando/fatiando/issues?q=is%3Aopen+is%3Aissue+label%3A%22low-hanging+fruit%22>`__
    that are suitable for new-comers.
    Note that "low-hanging" does not necessarily mean trivial,
    but that it doesn't require extensive knowledge of the project.
    **Helping by writing/reviewing/improving the documentation is VERY
    appreciated**. We welcome even the simplest typo fix!

Setting up
----------

You will need some extra dependencies installed for development.
See files ``ci/requirements-conda.txt`` and ``ci/requirements-pip.txt``.
If you are using Anaconda (and you should),
the repository provides an ``environment.yml`` file that specifies a ``conda``
virtual environment with all packages that you'll need.
This will keep the Fatiando development related installation from you main
Python.
The main advantage is that you can make changes to Fatiando and test them while
still using a stable release (like 0.4) for your main work.
Otherwise, changes to the code would likely break everything else you're
working on.

Run the following from the repository base directory to create the environment
using the specification in the ``environment.yml`` file::

    conda env create

Now, whenever you want to run code using the ``fatiando-dev`` environment we
just created, you must run this first to activate the environment::

    source activate fatiando-dev

or on Windows::

    activate fatiando-dev

Optionally, you can use `make <http://www.gnu.org/software/make/>`__ to take
advantage of the project ``Makefile`` to compile and run tests.

Once you have your fork and local clone as well as the environment created and
activated, you need to make sure that Python can import the ``fatiando`` code
from your fork. This way, you can make changes and run code to test that your
changes work.

.. note::

    **Don't** set the ``PYTHONPATH`` environment variable. This can be tricky
    under Windows. It's better to use ``pip`` in editable mode (below).

First, make sure you have uninstalled Fatiando from your environment (just to
be sure)::

    pip uninstall fatiando

To make it so that Python can find the code in the repository, run ``pip
install`` in `editable mode
<https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs>`__::

    pip install -e .

If you're using the ``Makefile`` you can run ``make develop`` to do this.

Now, changes to the repository code will be accessible from your installed
Fatiando.


.. _develop_test:

Testing
-------

Fatiando uses automated tests to check that the code works and
produces the expected results.
There are two types of tests currently being used:
unit tests and doc tests.

.. note::

    If you are new to automated tests, see the Software Carpentry lessons on
    `Testing: Unit Testing
    <http://software-carpentry.org/>`__.

Unit tests are implemented in ``tests`` folders inside each subpackage of
``fatiando``.
Doctests are part of the docstrings of functions and modules.
You'll recognize them by the ``>>>`` in each line of code.
Both tests are found and run automatically by
`py.test <http://pytest.org/>`__.

There are different ways of running the tests. From the command line using the
``py.test`` app::

    py.test --doctest-modules --pyargs fatiando

Or from Python using the ``fatiando.test`` function::

    import fatiando
    fatiando.test()

Or from the command line using the function::

    python -c "import fatiando; fatiando.test()"


If you use the ``Makefile``, running ``make test`` will perform the above in a
temporary folder.

You can also check the test coverage (how much of each module is tested) by::

    py.test --cov=fatiando --doctest-modules --pyargs fatiando

or passing ``coverage=True`` to the ``fatiando.test`` function::

    python -c "import fatiando; fatiando.test(coverage=True)"


.. important::

    **All new code contributed must be tested**.
    This means that it must have unit
    tests and/or doctests that make sure it gives the expected results.
    Tests should also make sure that the proper errors happen when the code is
    given bad input.
    A good balance would be to have both
    doctests that run a simple example (they are documentation, after all)
    and unit tests that are more elaborate and complete
    (using more data, testing corner/special cases, etc).

**Our goal** is to reach at least 90% test coverage
`by version 1.0 <https://github.com/fatiando/fatiando/issues/102>`__.


Adding new code/fixes/docs
--------------------------

**All new code** should be committed to a **new branch**.
Fatiando uses the
`"Github Flow" <http://scottchacon.com/2011/08/31/github-flow.html>`__
for managing branches in the repository.
The tutorial `"Understanding the Github flow"
<https://guides.github.com/introduction/flow/index.html>`__
offers a quick visual introduction to how that works.
See the :ref:`Pull Requests <develop_pr>` section below.

.. important::

    Don't edit the *master* branch directly!

Before working on the code for a new feature/fix/documentation,
you'll need to create a *branch* to store your commits.
Make sure you always start your new branch from *master*::

    git checkout master
    git checkout -b NAME_OF_NEW_BRANCH

Replace ``NAME_OF_NEW_BRANCH`` to something relevant to the changes you are
proposing.
For example, ``doc-devel-start-guide``, ``refactor-gravmag-prism``,
``seismic-tomo-module``, etc.

.. important::

    **Don't make multiple large changes in a single branch.**
    For example,
    refactoring a module to make it faster and adding a new function to a
    different module.
    If you do this, we will only be able to merge your code once **all** new
    features are tested, discussed, and documented.
    Make separate branches for different things you are working on
    (and start all of them from *master*).
    This way we can merge new changes as they are finished instead of having to
    wait a long time to merge everything.
    It will be even worse if one of the changes is controversial or needs a lot
    of discussion and planning.


Once you have your new branch, you're all set to start coding/writing.
Remember to run ``make test`` and check if your changes didn't break anything.
**Write tests sooner rather than later**.
They will not only help you check if your new code is working properly,
but also provide you with a "deadline" of sorts.
When your code passes your tests, then it is probably "good enough".

You should consider :ref:`openning a Pull Request <develop_pr>`
as soon as have any code that you might want to share.
The sooner you open the PR, the sooner we can start reviewing it and helping
you make your contribution.


Code Style
----------

Fatiando follows the `PEP8 <http://legacy.python.org/dev/peps/pep-0008/>`__
conventions for code style.

Conformance to PEP8 can be checked automatically using the
`pep8 <https://pypi.python.org/pypi/pep8>`__ package.
To see which if any code is not following the standard, run::

	pep8 --show-source --ignore=W503,E226,E241 --exclude=_version.py fatiando cookbook gallery

or::

    make pep8

This command will tell you exactly which file and line broke PEP8 compliance
and what was wrong with it.


.. _develop_docs:

Documentation
-------------

The documentation for Fatiando is built using
`sphinx <http://sphinx-doc.org/>`__.
The source files for the documentation are in the ``doc`` folder of the
repository.
The most sections of the docs are built from the ``doc/*.rst`` files.
The :ref:`API <fatiando>` section is automatically built from the
`docstrings <http://legacy.python.org/dev/peps/pep-0257/>`__ of
packages, modules, functions, and classes.

.. note::

    Source files and docstrings are written in reStructuredText (rst)
    and converted by sphinx to HTML.
    This `quick guide to rst <http://sphinx-doc.org/rest.html>`__
    is a good reference to get started with rst.

**Docstrings** are formatted in a style particular to Fatiando.
`PEP257 <http://legacy.python.org/dev/peps/pep-0257/>`__
has some good general guidelines.
Have a look at the other docstrings in Fatiando and format your own to follow
that style.

Some brief guidelines:

* Module docstrings should include a list of module classes and functions
  followed by brief descriptions of each.
* Function docstrings::

        def foo(x, y=4):
            r"""
            Brief description, like 'calculates so and so using bla bla bla'

            A more detailed description follows after a blank line. Can have
            multiple paragraphs, citations (Bla et al.,  2014), and equations.

            .. math::

                g(y) = \int_V y x dx

            After this, give a full description of ALL parameters the
            function takes.

            Parameters:

            * x : float or numpy array
                The variable that goes on the horizontal axis. In Meh units.
            * y : float or numpy array
                The variable that goes on the vertical axis. In Meh units.
                Default: 4.

            Returns:

            * g : float or numpy array
                The value of g(y) as calculated by the equation above.

            Examples:

            You can include examples as doctests. These are automatically found
            by the test suite and executed. Lines starting with >>> are code.
            Lines below them that don't have >>> are the result of that code.
            The tests compare the given result with what you put as the
            expected result.

            >>> foo(3)
            25
            >>> import numpy as np
            >>> foo(np.array([1, 2])
            array([ 45.  34. ])

            References:

            Include a list of references cited.

            Bla B., and Meh M. (2014). Some relevant article describing the
            methods. Journal. doi:82e1hd1puhd7
            """
* Class docstrings will contain a description of the class and the parameters
  that `__init__` takes. It should also include examples (as doctests when
  possible) and references. Pretty much like function docstrings.


You'll need to install the `Sphinx bootstrap theme
<https://github.com/ryan-roemer/sphinx-bootstrap-theme>`__ to build the docs.
Run this in your terminal/cmd.exe::

    pip install sphinx_bootstrap_theme

To compile the documentation, run::

    cd doc
    make all

To view the compiled HTML files, run this inside the ``doc`` folder::

    make serve

This will start a server in the ``doc/_build/html`` folder.
Point your browser to `http://127.0.0.1:8008 <http://127.0.0.1:8008/>`__
to view the site.
Use ``Ctrl+C`` to stop the server.


.. _develop_pr:

Pull Requests
-------------

Pull requests (PRs) are how we submit new code and fixes to Fatiando.
The PRs are were your contribution will be revised by other developers.
This works a lot like peer-review does in Science, but we hope you'll find it a
much nicer experience!

.. note::

    To get the general idea of the Pull Request cycle, see
    `"Understanding the Github flow"
    <https://guides.github.com/introduction/flow/index.html>`__.

After you have your set of changes in a new branch of your ``fatiando`` fork,
make a Pull Request to the *master* branch of
`fatiando/fatiando <https://github.com/fatiando/fatiando>`__.
Use the main text of the PR to describe in detail what you have done and why.
Explain the purpose of the PR.
What changes are you proposing and why they are
good/awesome/necessary/desirable?
See `PR 137 <https://github.com/fatiando/fatiando/pull/137>`__ for an example.

PRs serve as a platform for reviewing the code.
Ideally, someone else will go through your code to make sure there aren't any
obvious mistakes.
The reviewer can also suggest improvements, help with unfixed problems, etc.
This is the same as the peer-review processes in scientific publication
(or at least what it should be).
See the
`list of completed pull requests
<https://github.com/fatiando/fatiando/pulls?q=is%3Apr+is%3Amerged>`__
for examples of how the process works.

.. warning::

    Reviewers should **always be polite** in their **constructive** criticism.
    Rudeness and prejudice will not be tolerated.
    **Beware of wit, humor, and sarcasm**.
    It might not always be understood in writting
    and not always translates accross native languages.

PRs will only be merged if they meet certain criteria:

* New code must be have :ref:`automated tests <develop_test>`
* All tests must pass (this will be evaluated automatically by
  `TravisCI <https://travis-ci.org/fatiando/fatiando/>`__)
* All code must follow the
  `PEP8 <http://legacy.python.org/dev/peps/pep-0008/>`__ style conventions.
  This will also be check automatically by the tests and TravisCI
* All new code and changes must be documented with
  :ref:`docstrings <develop_docs>`
* New code must not cause merge conflicts (someone will help you resolve this
  in case it happens and you don't know what to do)

Even if all of these requirements are met,
features that fall outside of the scope of the project might not be
accepted (but we will discuss the possibility).
So **before you start coding**
open `an issue <https://github.com/fatiando/fatiando/issues>`__ explaining what
you mean to do first so that we can discuss it.
Check if there isn't an issue open for this already.
This way we can keep track of who is working on what and avoid duplicated work.

To help keep track of what you need to do,
a checklist will be automatically inserted into the pull request description
(adapted from the
`khmer docs
<http://khmer.readthedocs.io/en/v1.1/development.html#checklist>`__)::

    ## Checklist:

    - [ ] Make tests for new code
    - [ ] Create/update docstrings
    - [ ] Include relevant equations and citations in docstrings
    - [ ] Code follows PEP8 style conventions
    - [ ] Code and docs have been spellchecked
    - [ ] Include new dependencies in docs, README, and .travis.yml
    - [ ] Documentation builds properly
    - [ ] All tests pass
    - [ ] Can be merged
    - [ ] Changelog entry (leave for last)
    - [ ] Firt-time contributor? Add yourself to `doc/contributors.rst` (leave for last)

This will create check boxes that you can mark as you complete each of the
requirements.
If you don't know how to do some of them, contact a developer
by writing a comment on the PR @-mentioning their user name
(e.g., `@leouieda <https://github.com/leouieda/>`__
or `@birocoles <https://github.com/birocoles/>`__).
