#!/bin/bash
if [ "$TRAVIS_PULL_REQUEST" != "false" ]; then
    echo -e "Overwritting CNAME file to:"
    echo "www.fatiando.org/pulls/$TRAVIS_PULL_REQUEST" > doc/CNAME
    cat doc/CNAME
fi
