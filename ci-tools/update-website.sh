#!/bin/bash
# Based on
# http://sleepycoders.blogspot.com.au/2013/03/sharing-travis-ci-generated-files.html
# and https://github.com/richfitz/wood
echo -e "Preparing to copy generated files to fatiando.github.io"
if [ "$TRAVIS_PULL_REQUEST" == "false" ]; then
    if [ "$TRAVIS_BRANCH" == "master" ]; then
        echo -e "Starting to update website\n"
        cp -R doc/_build/html/ $HOME/keep
        # Go to home and setup git
        cd $HOME
        git config --global user.email "leouieda@gmail.com"
        git config --global user.name "Leonardo Uieda"
        git config --global github.user "leouieda"
        echo -e "Cloning fatiando.github.io"
        # Clone the project, using the secret token. Uses /dev/null to avoid leaking decrypted key
        git clone --quiet --branch=master --single-branch https://${GH_TOKEN}@github.com/fatiando/fatiando.github.io.git fatiando.org > /dev/null
        cd fatiando.org
        # Move the old branch out of the way and create a new one:
        git branch -m master-old
        git checkout --orphan master
        # Delete all the files and replace with our good set
        git rm -rf .
        cp -Rf $HOME/keep/. $HOME/fatiando.org
        # add, commit and push files
        git add -f .
        git commit -m "Travis build $TRAVIS_BUILD_NUMBER. Triggered by $TRAVIS_COMMIT"
        echo -e "Pushing..."
        git push -fq origin master > /dev/null
        echo -e "Uploaded generated files\n"
    else
        echo -e "This isn't the master branch. Not updated website."
    fi
else
    echo -e "This is a Pull Request. Pushing built docs to fatiando/pulls.\n"
    echo -e "Starting to update website\n"
    cp -R doc/_build/html/ $HOME/keep
    # Go to home and setup git
    cd $HOME
    git config --global user.email "leouieda@gmail.com"
    git config --global user.name "Leonardo Uieda"
    git config --global github.user "leouieda"
    echo -e "Cloning fatiando/pulls"
    # Clone the project, using the secret token. Uses /dev/null to avoid leaking decrypted key
    git clone --quiet --depth=50 --branch=gh-pages --single-branch https://${GH_TOKEN}@github.com/fatiando/pulls.git > /dev/null
    cd pulls
    git checkout gh-pages
    # Make sure the PR folder is present
    mkdir -p $TRAVIS_PULL_REQUEST
    # Delete all the files and replace with our good set
    git rm -rf $TRAVIS_PULL_REQUEST/*
    cp -Rf $HOME/keep/. $HOME/pulls/$TRAVIS_PULL_REQUEST
    # add, commit and push files
    git add -f $TRAVIS_PULL_REQUEST
    git commit -m "Travis build $TRAVIS_BUILD_NUMBER: PR$TRAVIS_PULL_REQUEST commit $TRAVIS_COMMIT"
    echo -e "Pushing..."
    git push -fq origin gh-pages > /dev/null
    echo -e "Uploaded generated files\n"
fi
echo -e "Done"
