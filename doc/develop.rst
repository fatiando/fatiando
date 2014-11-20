.. _develop:

Developer guide
===============


Pull Requests
-------------

Pull requests (PRs) are how we submit new code and fixes to Fatiando.
They also serve as a platform for reviewing the code.
Ideally, someone else will go through your code to make sure there aren't any
obvious mistakes.
The reviewer can also suggest improvements, help with unfixed problems, etc.
This is the same as the peer-review processes in scientific publication
(or at least what it should be).

.. warning:: Reviewers should **always be polite** in their **constructive**
    criticism. Rudeness and prejudice will not be tolerated.

See the
`list of completed pull requests <https://github.com/fatiando/fatiando/pulls?q=is%3Apr+is%3Amerged>`__
for examples of how the process works.

PRs will only be merged if they meet certain criteria:

* New code must be have automated tests
* All tests must pass (this will be evaluated automatically by
  `TravisCI <https://travis-ci.org/fatiando/fatiando/>`__
* All new code and changes must be documented with
  `docstrings <http://legacy.python.org/dev/peps/pep-0257/>`__
* New code must not cause merge conflicts (someone will help you resolve this
  in case it happens and you don't know what to do)
* All code must follow the
  `PEP8 <http://legacy.python.org/dev/peps/pep-0008/>`__ style conventions.
  This will also be check automatically by the tests (and TravisCI)

If you don't know what these things are, I recommend that you read through
the `Software Carpentry <http://software-carpentry.org/>`__ lessons.
Particularly the lessons "Testing", "Version control with git", and
"Program design" (all of the lessons are great though).

Even if all of these requirements are met,
features that fall outside of the scope of the project might not be
accepted (but we will discuss the possibility).
So **before you start coding**
open `an issue <https://github.com/fatiando/fatiando/issues>`__ explaining what
you mean to do first so that we can discuss it.
Check if there isn't an issue open for this already.
This way we can keep track of who is working on what and avoid duplicated work.

PRs should be made to the ``master`` branch of the
main repository:
`fatiando/fatiando <https://github.com/fatiando/fatiando>`__

When submitting a PR, explain in the description what the purpose of the PR is.
What changes are you proposing and why?
To help keep track of what you need to do,
copy this checklist to the PR description
(adapted from the
`khmer docs
<http://khmer.readthedocs.org/en/v1.1/development.html#checklist>`__)::

    ## Checklist:

    - [ ] Make tests for new code
    - [ ] Create/update docstrings
    - [ ] Code follows PEP8 style conventions
    - [ ] Code and docs have been spellchecked
    - [ ] Changelog entry
    - [ ] Documentation builds properly
    - [ ] All tests pass
    - [ ] Can be merged

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

7. Check is versioneer is setting the correct version number (should print
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
