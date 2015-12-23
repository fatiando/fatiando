.. _develop:

Developer guide
===============

Getting started
---------------

The first thing you'll need is a `Github <http://github.com/>`__ account.
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
    start with the `Software Carpentry lessons on version control
    <http://software-carpentry.org/v5/novice/git/index.html>`__.
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

To contribute some code/fix/documentation, start by
`forking fatiando/fatiando <https://github.com/fatiando/fatiando/fork>`__
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

Once you have your fork and local clone, you should add it to your
``PYTHONPATH`` environment variable so that you can ``import fatiando``
directly from the source (without installing it).

First, make sure you have uninstalled Fatiando::

    pip uninstall fatiando

In GNU/Linux (or MacOS), add the following lines to your ``~/.bashrc`` file::

    export PYTHONPATH=path/to/fatiando/repository:$PYTHONPATH

In Windows,
create a ``PYTHONPATH`` variable and set it to the path to the ``fatiando``
repository (e.g., ``C:\Documents\User\repos\fatiando``).
Follow
`this guide <http://www.computerhope.com/issues/ch000549.htm>`__
for instructions on setting environment variables.

You will need some extra dependencies installed for development.
If you are using Anaconda (and you should) run the following from the
repository base directory::

    conda install --file test/requirements-conda.txt

and::

    pip install -r test/requirements-pip.txt

You will also need `make <http://www.gnu.org/software/make/>`__, which usually
comes with GNU/Linux by default. On windows, you can get it through
`msysGit <http://msysgit.github.io/>`__.
**Don't use the first "Download" button**.
That will get you "Git for Windows".
What you want is "msysGit" (download link at the bottom of the page).
That will give you a
`bash shell <http://en.wikipedia.org/wiki/Bash_%28Unix_shell%29>`__,
git, and make.


Building
--------

The `Makefile <https://github.com/fatiando/fatiando/blob/master/Makefile>`__
contains all the commands for building the C extensions, testing, checking code
style, building the documentation, etc.

.. note::

    If you don't know what ``make`` is or need a quick start, read the
    `Software Carpentry lessons on Make
    <http://software-carpentry.org/v4/make/index.html>`__.


To build the C-extensions (all of the ``.c`` files in the ``fatiando``
folder)::

    make build

If the repository is in your ``PYTHONPATH`` you will now be able to ``import
fatiando`` and use it directly from the repository.
This is important for running and testing the new code you are making
(that is why you can't use the installed version of Fatiando).

The ``.c`` files were automatically generated using
`Cython <http://cython.org/>`__ from the ``.pyx`` files.
If you don't change the ``.pyx``  files, you have nothing to worry about.
If you make changes to the Cython code, then you'll need to re-generate the
corresponding ``.c`` files.
To do that, run::

    make cython

This will also compile the newly generated C code.
Once you are done editing the ``.pyx`` files, make sure to commit the generated
``.c`` file as well.

.. _develop_test:

Testing
-------

Fatiando uses automated tests to check that the code works and
produces the expected results.
There are two types of tests currently being used:
unit tests and doc tests.

.. note::

    If you are new to automated tests, see the (you guessed it)
    `Software Carpentry lessons on testing
    <http://software-carpentry.org/v4/test/index.html>`__.

Unit tests are implemented in the ``test/test_*.py`` files of the repository.
Doctests are part of the docstrings of functions and modules.
You'll recognize them by the ``>>>`` in each line of code.
Both tests are found and run automatically by
`nose <https://nose.readthedocs.org/en/latest/>`__.

To run all tests and check that your build was successful::

    make test

This will also build the extensions if they are not built. Failures will
indicate which test failed and print some useful information.

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

Conformance to PEP8 is checked automatically using the
`pep8 <https://pypi.python.org/pypi/pep8>`__ package.
The check is part of the unit tests and will report a test failure when new
code is incorrectly formatted.
The test failure message will be something like this::

    ======================================================================
    FAIL: all packages, tests, and cookbook conform to PEP8
    ----------------------------------------------------------------------
    Traceback (most recent call last):
      File "/home/leo/src/fatiando/test/test_pep8.py", line 13, in test_pep8_conformance
        "Found code style errors (and warnings).")
    AssertionError: Found code style errors (and warnings).

    ----------------------------------------------------------------------

To see which files/lines caused the error, run::

    $ make pep8
    pep8 --exclude=_version.py fatiando test cookbook
    fatiando/gravmag/prism.py:977:1: E302 expected 2 blank lines, found 1
    make: *** [pep8] Error 1

This command will tell you exactly which file and line broke PEP8 compliance
and what was wrong with it.
In this case, line 977 of ``fatiando/gravmag/prism.py`` needs to have an extra
blank line.


.. _develop_docs:

Documentation
-------------

The documentation for Fatiando is built using
`sphinx <http://sphinx-doc.org/>`__. The documentation audience is the end-user.
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
  Private methods will be hiden by sphinx on porpouse.


You'll need to install the `Sphinx bootstrap theme
<https://github.com/ryan-roemer/sphinx-bootstrap-theme>`__ to build the docs.
Run this in your terminal/cmd.exe::

    pip install sphinx_bootstrap_theme

To compile the documentation, run::

    make docs

To view the compiled HTML files, run::

    make view-docs

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
copy this checklist to the PR description
(adapted from the
`khmer docs
<http://khmer.readthedocs.org/en/v1.1/development.html#checklist>`__)::

    ## Checklist:

    - [ ] Make tests for new code
    - [ ] Create/update docstrings
    - [ ] Include relevant equations and citations in docstrings
    - [ ] Code follows PEP8 style conventions
    - [ ] Code and docs have been spellchecked
    - [ ] Include new dependencies in docs, requirements.txt, README, and .travis.yml
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

Making a release
----------------

This is intended as a checklist for packaging to avoid forgetting some
important steps.
Packaging is not something that has to be done very frequently and few
developers will need to worry about this.

These steps have to made from a clone of the main repository
(the one on the `fatiando <https://github.com/fatiando>`__ Github organization).
You'll need push rights to this repository for making a release.
If you don't have the rights,
send a message to
`the mailing list <https://groups.google.com/d/forum/fatiando>`__
and we'll see what we can do.

You'll also need to have maintainer rights on `PyPI
<https://pypi.python.org/pypi>`__.
Sign-up for an account there if you don't
have one and ask to be added as a maintainer.

0. Make sure you have a ``.pypirc`` file in your home directory. It should look
   something like this::

        [distutils]
        index-servers=
            pypi

        [pypi]
        repository = https://pypi.python.org/pypi
        username = <your username>

1. Make sure you're on the ``master`` branch and your repository is
   up-to-date::

       git checkout master
       git pull

2. Include the version number (e.g. ``0.3``) and the release date on
   ``doc/changelog.rst``. **Make sure to commit your changes!**

3. Check that the documentation builds properly. ``make view-docs`` will serve
   the generated HTML files. Point your browser to
   `http://127.0.0.1:8008 <http://127.0.0.1:8008>`__ to view them.
   Use ``Ctrl+C`` to stop the server.::

       make docs
       make view-docs

.. note:: Install the ReadTheDocs theme for sphinx if you don't have it
    ``pip install sphinx-rtd-theme``.

4. Make sure all tests pass::

       make test

5. Try to build the source packages. Check for any error messages and inspect
   the zip and tar files, just to make sure::

       make package

6. If everything is tested and works properly, you're ready to tag this release
   with a version number. **Make sure you have don't have any uncommited
   changes!**. The version number should be the same as the corresponding
   `Github milestone <https://github.com/fatiando/fatiando/milestones>`__
   (e.g., 0.3). The version number should have a ``v`` before it::

       git tag v0.3

7. Check if versioneer is setting the correct version number (should print
   something like ``v0.3``::

       python -c "import fatiando; print fatiando.__version__"

8. Push the tag to Github::

       git push --tags

9. Upload the built package (zip and tar files) to PyPI. Uses `twine
   <https://github.com/pypa/twine>`__ for the upload. Install it using
   ``pip install twine``.::

       make clean
       make package
       twine upload dist/* -p YOUR_PYPI_PASSWORD

10. Test the upload::

       pip install --upgrade fatiando
       export PYTHONPATH=""; cd ~; python -c "import fatiando; print fatiando.__version__"

11. Edit the
    `release on Github <https://github.com/fatiando/fatiando/releases>`__
    with some highlights of the changelog.
