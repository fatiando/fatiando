"""
Generate the rst files for the cookbook from the recipes.
"""
import sys
import os
import shutil

print os.path.abspath(__file__)

body = r"""
.. raw:: html

    [<a href="%s">source code</a>]

.. literalinclude:: %s
    :language: python
    :linenos:
"""

def build(recipes_dir):
    """
    Read the recipes from the dir and make rst files out of them.
    """
    sys.stderr.write("\nBUILDING THE COOKBOOK RECIPES:\n")
    # Create the cookbook dir in the doc dir
    cbdir = 'cookbook'
    if not os.path.exists(cbdir):
        os.mkdir(cbdir)
    static = os.path.join('_static', cbdir)
    if not os.path.exists(static):
        os.mkdir(static)

    recipes = [f for f in sorted(os.listdir(recipes_dir))
               if os.path.splitext(f)[-1] == '.py']

    for i, recipe in enumerate(recipes):
        output = os.path.splitext(recipe)[0] + '.rst'
        path_to_recipe = os.path.join(recipes_dir, recipe)
        sys.stderr.write("  %s --> %s\n" % (path_to_recipe, output))
        # Copy the recipe to the _static dir so that I can link to it
        shutil.copy(path_to_recipe, os.path.join(static, recipe))
        # Get the title from the first lines of the recipe docstring
        title = ''
        with open(path_to_recipe) as f:
            for line in f.readlines()[1:]:
                line = line.strip()
                if line == '"""' or not line:
                    break
                title = ' '.join([title, line])
        with open(os.path.join(cbdir, output), 'w') as f:
            f.write('.. _cookbook_%s:\n\n' % (os.path.splitext(recipe)[0]))
            f.write(title.strip() + '\n')
            f.write('='*len(title) + '\n')
            f.write(body % (
                os.path.join(os.path.pardir, static, recipe),
                os.path.join(os.path.pardir, path_to_recipe)))
    sys.stderr.write("Recipes built: %d\n\n" % (i + 1))
