#!/bin/bash

# Run the tests with or without coverage

# To return a failure if any commands inside fail
set -e

# Run the tests in an isolated directory to make sure I'm running the installed
# version of the package.
mkdir -p tmp
cd tmp
echo "Running tests inside: "`pwd`

python -c "import fatiando; print('Fatiando version:', fatiando.__version__)"

# Use the 'fatiando.test()' command to make sure we're testing an installed
# version of the package.
if [ "$COVERAGE" == "true" ];
then
    python -c "import fatiando; assert fatiando.test(verbose=True, coverage=True) == 0" && cp .coverage ..
else
    python -c "import fatiando; assert fatiando.test(verbose=True) == 0"
fi
