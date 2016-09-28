#!/bin/bash

# Run the tests with or without coverage

# To return a failure if any commands inside fail
set -e

mkdir -p tmp
cd tmp
echo "Running tests inside: "`pwd`
# Use the 'fatiando.test()' command to make sure we're testing an installed
# version of the package.
if [ "$COVERAGE" == "true"];
then
    python -c "import fatiando; assert fatiando.test(verbose=True, coverage=True) == 0" && cp .coverage ..
else
    python -c "import fatiando; assert fatiando.test(verbose=True) == 0"
fi
cd ..

