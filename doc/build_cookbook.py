"""
Generate the rst files for the cookbook.
"""
import os

title = """
%s
======================================================================
"""
code = """
.. include:: %s
    :code: python
    :number-lines:
"""
doc = """
.. automodule:: %s
   :members:
   :show-inheritance:
"""
# Create the cookbook dir in the doc dir
cbdir = 'cookbook'
if not os.path.exists(cbdir):
    os.mkdir(cbdir)
recipes = [f for f in sorted(os.listdir('../cookbook/recipes'))
           if os.path.splitext(f)[-1] == '.py']
for recipe in recipes[:3]:
    name = os.path.splitext(recipe)[0]
    path = '../../cookbook/recipes/%s' % (recipe)
    with open(os.path.join(cbdir, name + '.rst'), 'w') as f:
        f.write(title % (recipe))
        f.write(doc % (name))
        f.write(code % (path))
