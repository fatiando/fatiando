"""
Generate the rst files for the cookbook from the recipes.
"""
import sys
import os

body = r"""

**Download** source code: :download:`{recipe}<{code}>`

.. literalinclude:: {code}
    :language: python
"""

def recipe_to_rst(recipe):
    """
    Convert a .py recipe to a .rst entry for sphinx
    """
    sys.stderr.write("Converting {} to rst ...".format(recipe))
    recipe_file = os.path.split(recipe)[-1]
    recipe_name = os.path.splitext(recipe_file)[0]
    output = recipe_name + '.rst'
    # Get the title from the first lines of the recipe docstring
    title = ''
    with open(recipe) as f:
        for line in f.readlines()[1:]:
            line = line.strip()
            if line == '"""' or not line:
                break
            title = ' '.join([title, line])
    with open(output, 'w') as f:
        f.write('.. _cookbook_{}:\n\n'.format(recipe_name))
        f.write(title.strip() + '\n')
        f.write('='*len(title) + '\n')
        f.write(body.format(
            recipe=recipe_file,
            code='../_static/cookbook/{}'.format(recipe_file)))
    sys.stderr.write(" done\n")


if __name__ == '__main__':
    for recipe in sys.argv[1:]:
        recipe_to_rst(recipe)
