"""
Geothermal: Forward and inverse modeling of a linear change in temperature
measured in a well
"""
import numpy
from fatiando import logger, utils
from fatiando.geothermal import climsig
from fatiando.vis import mpl

log = logger.get()
log.info(logger.header())
log.info(__doc__)

# Generating synthetic data
amp = 5.43
age = 78.2
zp = numpy.arange(0, 100, 1)
temp, error = utils.contaminate(climsig.linear(amp, age, zp),
    0.02, percent=True, return_stddev=True)

# Preparing for the inversion
p, residuals = climsig.ilinear(temp, zp)
est_amp, est_age = p

mpl.figure(figsize=(12,5))
mpl.subplot(1, 2, 1)
mpl.title("Climate signal (linear)")
mpl.plot(temp, zp, 'ok', label='Observed')
mpl.plot(temp - residuals, zp, '--r', linewidth=3, label='Predicted')
mpl.legend(loc='lower right', numpoints=1)
mpl.xlabel("Temperature (C)")
mpl.ylabel("Z")
mpl.ylim(100, 0)
ax = mpl.subplot(1, 2, 2)
ax2 = mpl.twinx()
mpl.title("Age and amplitude")
width = 0.3
ax.bar([1 - width], [age], width, color='b', label="True")
ax.bar([1], [est_age], width, color='r', label="Estimate")
ax2.bar([2 - width], [amp], width, color='b')
ax2.bar([2], [est_amp], width, color='r')
ax.legend(loc='upper center', numpoints=1)
ax.set_ylabel("Age (years)")
ax2.set_ylabel("Amplitude (C)")
ax.set_xticks([1, 2])
ax.set_xticklabels(['Age', 'Amplitude'])
ax.set_ylim(0, 100)
ax2.set_ylim(0, 7)
mpl.show()
