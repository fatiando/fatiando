"""
Simulate the climate signal of an abrupt change in temperature measured in a
temperature well log. Invert for the amplitude and age of the change.
"""
from matplotlib import pyplot
import numpy
from fatiando.heat import climatesignal
from fatiando.inversion.gradient import levmarq
from fatiando import logger, vis, utils

log = logger.get()
log.info(logger.header())
log.info(__doc__)

log.info("Generating synthetic data")
amp = 3
age = 54
zp = numpy.arange(0, 100, 1)
temp, error = utils.contaminate(climatesignal.abrupt(amp, age, zp), 0.01,
                                percent=True, return_stddev=True)

log.info("Preparing for the inversion")
solver = levmarq(initial=(0.5, 53))
p, residuals = climatesignal.invert_abrupt(temp, zp, solver)
est_amp, est_age = p

pyplot.figure(figsize=(12,5))
pyplot.subplot(1, 2, 1)
pyplot.title("Climate signal (abrupt)")
pyplot.plot(temp, zp, 'ok', label='Observed')
pyplot.plot(temp - residuals, zp, '--r', linewidth=3, label='Predicted')
pyplot.legend(loc='lower right', numpoints=1)
pyplot.xlabel("Temperature (C)")
pyplot.ylabel("Z")
pyplot.ylim(100, 0)
ax = pyplot.subplot(1, 2, 2)
ax2 = pyplot.twinx()
pyplot.title("Age and amplitude")
width = 0.3
ax.bar([1 - width], [age], width, color='b', label="True")
ax.bar([1], [est_age], width, color='r', label="Estimate")
ax2.bar([2 - width], [amp], width, color='b')
ax2.bar([2], [est_amp], width, color='r')
ax.legend(loc='upper center', numpoints=1)
ax.set_ylabel("Years")
ax2.set_ylabel("C")
ax.set_xticks([1, 2])
ax.set_xticklabels(['Age', 'Amplitude'])
ax.set_ylim(0, 70)
ax2.set_ylim(0, 4)
pyplot.show()
