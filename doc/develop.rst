.. _develop:

Developer guide
===============

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
   <https://github.com/pypa/twine>`__ for the upload.::

       make clean
       make package
       make upload

10. Test the upload::

       pip install --upgrade fatiando
       export PYTHONPATH=""; cd ~; python -c "import fatiando; print fatiando.__version__"

11. Edit the
    `release on Github <https://github.com/fatiando/fatiando/releases>`__
    with some highlights of the changelog.
