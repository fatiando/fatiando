"""
Create a model based on an image file
-------------------------------------

Function :func:`fatiando.datasets.from_image` allows us to create a model
template from an image file. We can use it to assign a physical property value
to each color of an image. The model can later be fed to a forward modeling
function to produce synthetic data or even as a starting estimate for an
inversion.
"""
from fatiando import datasets
import matplotlib.pyplot as plt

# Use a sample image packaged with Fatiando for this example
template = datasets.from_image(datasets.SAMPLE_IMAGE)

# Use the template to assign velocity values to each color of the image
model = template.copy()
model[template == 0] = 3000
model[template == 1] = 6000
model[template == 2] = 0
model[template == 3] = 8000

# Now we can plot our velocity model
plt.figure(figsize=(7, 5.5))
plt.title('Velocity model')
plt.imshow(model, cmap="viridis")
plt.colorbar(pad=0, aspect=40).set_label('velocity (m/s)')
plt.tight_layout()
plt.show()
