#!/bin/bash
# Based on
# http://sleepycoders.blogspot.com.au/2013/03/sharing-travis-ci-generated-files.html
# and https://github.com/richfitz/wood
if [ "$TRAVIS_PULL_REQUEST" == "false" ] && [ "$TRAVIS_BRANCH" == "master" ]; then
    echo -e "This is the master branch. Pushing built docs to the fatiando/dev repository"
    echo -e "Copying generated files."
    cp -R doc/_build/html/ $HOME/keep
    # Go to home and setup git
    cd $HOME
    git config --global user.email "leouieda@gmail.com"
    git config --global user.name "Leonardo Uieda"
    git config --global github.user "leouieda"
    echo -e "Cloning fatiando.github.io"
    # Clone the project, using the secret token. Uses /dev/null to avoid leaking decrypted key
    git clone --quiet --branch=gh-pages --single-branch https://${GH_TOKEN}@github.com/fatiando/dev website > /dev/null
    cd website
    # Move the old branch out of the way and create a new one:
    echo -e "Create and empty master branch"
    git checkout gh-pages
    git branch -m gh-pages-old
    git checkout --orphan gh-pages
    # Delete all the files and replace with our good set
    echo -e "Remove old files from previous builds"
    git rm -rf .
    cp -Rf $HOME/keep/. $HOME/website
    # Remove the CNAME file because this will be under the /dev sufix
    rm CNAME
    # add, commit and push files
    git add -f .
    echo -e "Commit changes"
    git commit -m "Travis build $TRAVIS_BUILD_NUMBER. Triggered by $TRAVIS_COMMIT"
    echo -e "Pushing..."
    git push -fq origin gh-pages > /dev/null
    echo -e "Uploaded generated files\n"
else
    echo -e "This is a PR or isn't the master branch. Not updating website."
fi
echo -e "Done"
