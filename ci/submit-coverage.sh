#!/bin/bash
# Submit coverage information to coveralls from TravisCI builds
# Only do this for PRs and pushes to the master branch on Linux

# To return a failure if any commands inside fail
set -e

if [ "$TRAVIS_OS_NAME" == "linux" ] &&
   ([ "$TRAVIS_PULL_REQUEST" == "true" ] || [ "$TRAVIS_BRANCH" == "master" ]);
then
    pip install coveralls==0.5
    echo "Pushing coverage to coveralls."
    coveralls
else
    echo "Not pushing coverage."
fi
