"""
Wavelets for seismic modeling
-----------------------------

The :mod:`fatiando.seismic` package defines classes to evaluate and sample
wavelets. Here is an example of how to use the
:class:`~fatiando.seismic.RickerWavelet` class.
"""
import matplotlib.pyplot as plt
import numpy as np
from fatiando.seismic import RickerWavelet

# Make three wavelets to show how to evaluate them at given times.
w1 = RickerWavelet(f=5)
w2 = RickerWavelet(f=1, delay=0)
w3 = RickerWavelet(f=10, amp=-0.5, delay=0.7)

times = np.linspace(0, 1, 200)
# Call the wavelets like functions to get their values
v1 = w1(times)
v2 = w2(times)
v3 = w3(times)

# You can also sample the wavelets with a given time interval (dt). The default
# duration of the sampling is 2*delay, which guarantees that the whole wavelet
# is sampled.
w = RickerWavelet(f=60)
samples = w.sample(dt=0.001)

plt.figure(figsize=(8, 5))

ax = plt.subplot(1, 2, 1)
ax.set_title('Ricker wavelets')
ax.plot(times, v1, '-', label='f=5')
ax.plot(times, v2, '--', label='f=1')
ax.plot(times, v3, '-.', label='f=10')
ax.grid()
ax.legend()
ax.set_ylabel('Amplitude')
ax.set_xlabel('time (s)')

ax = plt.subplot(1, 2, 2)
ax.set_title('Wavelet sampled at 0.001s')
ax.plot(samples, '.k')
ax.grid()
ax.set_ylabel('Amplitude')
ax.set_xlabel('sample number')

plt.tight_layout()
plt.show()
