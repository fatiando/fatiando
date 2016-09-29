#!/bin/bash
# Submit coverage information to coveralls from TravisCI builds
# Only do this for PRs and pushes to the master branch on Linux

# To return a failure if any commands inside fail
set -e

if [ "$COVERAGE" == "true" ];
then
    echo "Pushing coverage to coveralls."
    coveralls
else
    echo "Not pushing coverage."
fi
