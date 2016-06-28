#!/bin/bash

# To be able to run mayavi in headless mode (no windows) follow the
# instructions at:
# https://docs.travis-ci.com/user/gui-and-headless-browsers/#Using-xvfb-to-Run-Tests-That-Require-a-GUI
# Copied from the sphinx-gallery config file:
# https://github.com/sphinx-gallery/sphinx-gallery/blob/master/.travis.yml

# Only build the docs on Linux and Python 2,7. Mayavi won't work in Python 3
# yet or on OS X in headless mode (it might, but I don't want to find out how)

if [ "$TRAVIS_OS_NAME" == "linux" ] && [ "$PYTHON" == "2.7" ];
then
    export DISPLAY=:99.0
    sh -e /etc/init.d/xvfb start
    sleep 3  # give xvfb some time to start
    make -C doc
    echo "Finished building documentation"
else
    echo "Not building docs. Only build on Linux and Python 2.7"
fi
