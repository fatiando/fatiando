#!/bin/bash
# Based on
# http://sleepycoders.blogspot.com.au/2013/03/sharing-travis-ci-generated-files.html
# and https://github.com/richfitz/wood
if [ "$TRAVIS_PULL_REQUEST" == "false" ]; then
    if [ "$TRAVIS_BRANCH" == "master" ]; then
        echo -e "This is the master branch. Pushing built docs to fatiando.github.io"
        echo -e "Copying generated files."
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
        echo -e "Create and empty master branch"
        git branch -m master-old
        git checkout --orphan master
        # Delete all the files and replace with our good set
        echo -e "Remove old files from previous builds"
        git rm -rf .
        cp -Rf $HOME/keep/. $HOME/fatiando.org
        # add, commit and push files
        git add -f .
        echo -e "Commit changes"
        git commit -m "Travis build $TRAVIS_BUILD_NUMBER. Triggered by $TRAVIS_COMMIT"
        echo -e "Pushing..."
        git push -fq origin master > /dev/null
        echo -e "Uploaded generated files\n"
    else
        echo -e "This isn't the master branch. Not updated website."
    fi
else
    echo -e "This is a Pull Request. Pushing built docs to fatiando/pulls.\n"
    echo -e "Copying generated files."
    cp -R doc/_build/html/ $HOME/keep
    # Go to home and setup git
    cd $HOME
    git config --global user.email "leouieda@gmail.com"
    git config --global user.name "Leonardo Uieda"
    git config --global github.user "leouieda"
    echo -e "Cloning fatiando/pulls"
    # Clone the project, using the secret token. Uses /dev/null to avoid leaking decrypted key
    git clone --quiet --depth=50 --branch=gh-pages --single-branch https://${GH_TOKEN}@github.com/fatiando/pull.git > /dev/null
    cd pull
    git checkout gh-pages
    # Make sure the PR folder is present
    mkdir -p $TRAVIS_PULL_REQUEST
    # Delete all the files and replace with our good set
    echo -e "Remove old files from previous builds"
    git rm -rf $TRAVIS_PULL_REQUEST/*
    echo -e "Copy new files to the PR folder"
    cp -Rf $HOME/keep/. $HOME/pull/$TRAVIS_PULL_REQUEST
    # Make sure .nojekyll is in the root dir
    touch .nojekyll
    # add, commit and push files
    git add -f .
    echo -e "Commit changes"
    git commit -m "Travis build $TRAVIS_BUILD_NUMBER: PR $TRAVIS_PULL_REQUEST commit $TRAVIS_COMMIT"
    echo -e "Pushing..."
    git push -fq origin gh-pages > /dev/null
    echo -e "Uploaded generated files\n"
fi
echo -e "Done"
