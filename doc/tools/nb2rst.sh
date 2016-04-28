#!/bin/bash

# Use jupyter nbconvert to execute notebooks in a folder and convert them to
# rst.  After execution, will strip the output from the notebook.

cd $1
for nb in *.ipynb
do
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=60 \
        $nb
    jupyter nbconvert --to rst $nb
    jupyter nbconvert --to notebook --inplace \
        --ClearOutputPreprocessor.enabled=True \
        $nb
done
