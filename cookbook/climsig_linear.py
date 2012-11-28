"""
Geothermal: Forward and inverse modeling of a linear change in temperature
measured in a well
"""
import numpy
import fatiando as ft

log = ft.logger.get()
log.info(ft.logger.header())
log.info(__doc__)

# Generating synthetic data
amp = 5.43
age = 78.2
zp = numpy.arange(0, 100, 1)
temp, error = ft.utils.contaminate(ft.geothermal.climsig.linear(amp, age, zp),
    0.02, percent=True, return_stddev=True)

# Preparing for the inversion
p, residuals = ft.geothermal.climsig.ilinear(temp, zp)
est_amp, est_age = p

ft.vis.figure(figsize=(12,5))
ft.vis.subplot(1, 2, 1)
ft.vis.title("Climate signal (linear)")
ft.vis.plot(temp, zp, 'ok', label='Observed')
ft.vis.plot(temp - residuals, zp, '--r', linewidth=3, label='Predicted')
ft.vis.legend(loc='lower right', numpoints=1)
ft.vis.xlabel("Temperature (C)")
ft.vis.ylabel("Z")
ft.vis.ylim(100, 0)
ax = ft.vis.subplot(1, 2, 2)
ax2 = ft.vis.twinx()
ft.vis.title("Age and amplitude")
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
ft.vis.show()
