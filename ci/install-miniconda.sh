#!/bin/bash

MINICONDA_URL="http://repo.continuum.io/miniconda"

if [ "$TRAVIS_OS_NAME" == "osx" ]; then
    MINICONDA_FILE=Miniconda-latest-MacOSX-x86_64.sh
    export PATH=/Users/travis/miniconda2/bin:$PATH
else
    MINICONDA_FILE=Miniconda-latest-Linux-x86_64.sh
    export PATH=/home/travis/miniconda2/bin:$PATH
fi
wget $MINICONDA_URL/$MINICONDA_FILE -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b
