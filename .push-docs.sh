#!/bin/bash
# Based on
# http://sleepycoders.blogspot.com.au/2013/03/sharing-travis-ci-generated-files.html
# and https://github.com/richfitz/wood
echo -e "Preparing to copy generated files to gh-pages"
#if [ "$TRAVIS_PULL_REQUEST" == "false" ] && [ "$TRAVIS_BRANCH" == "master" ]; then
    echo -e "Starting to update docs website\n"
    cp -R doc/_build/html/ $HOME/keep
    # Go to home and setup git
    cd $HOME
    git config --global user.email "leouieda@gmail.com"
    git config --global user.name "Leonardo Uieda"
    git config --global github.user "leouieda"
    echo -e "Cloning project"
    # Clone the project, using the secret token. Uses /dev/null to avoid leaking decrypted key
    git clone --quiet --branch=gh-pages --single-branch https://${GH_TOKEN}@github.com/fatiando/docs.git gh-pages > /dev/null
    cd gh-pages
    # Move the old branch out of the way and create a new one:
    git branch -m gh-pages-old
    git checkout --orphan gh-pages
    # Delete all the files and replace with our good set
    git rm -rf .
    cp -Rf $HOME/keep/. $HOME/gh-pages
    echo "http://fatiando.github.io/docs/" > CNAME
    echo "" > .nojekyll
    # add, commit and push files
    git add -f .
    git commit -m "Travis build $TRAVIS_BUILD_NUMBER pushed to fatiando/docs gh-pages"
    echo -e "Pushing to gh-pages"
    git push -fq origin gh-pages > /dev/null
    echo -e "Uploaded generated files\n"
#else
    #echo -e "This is a pull request, not copying files"
#fi
echo -e "Done"
