"""
Classes to evaluate and sample wavelets.
"""
from __future__ import division
from future.utils import with_metaclass
from future.builtins import super, object
from abc import ABCMeta, abstractmethod
import copy

import numpy as np


class BaseWavelet(with_metaclass(ABCMeta)):

    def __init__(self, amp, delay):
        self.amp = amp
        self.delay = delay

    @abstractmethod
    def __call__(self, time):
        pass

    def copy(self):
        """
        Return a deep copy of the wavelet.
        """
        return copy.deepcopy(self)

    def sample(self, dt, duration=None):
        """
        Sample the wavelet with the given sampling interval and duration.

        Parameters:

        * dt : float
            The sampling interval.
        * duration : float or ``None``
            The time duration of the sampling. If not ``None`` will use twice
            the *delay* attribute.

        Returns:

        * samples : 1d-array
            An array of sample values of the wavelet.

        """
        if duration is None:
            duration = 2*self.delay
        times = np.arange(0, duration, dt)
        samples = self(times)
        return samples


class RickerWavelet(BaseWavelet):
    """
    The Ricker (mexican hat) wavelet.

    .. math::

        w(t) = A (1 - 2(\pi f t)^2)\exp^{-(\pi f t)^2}

    Parameters:

    * f : float
        The peak (or central) frequency of the wavelet (:math:`f`).
    * amp : float
        The amplitude of the wavelet (:math:`A`).
    * delay : float or ``'default'``
        A time delay to be applied to the wavelet function. If ``'default'``
        will delay the wavelet by :math:`1/f` to guarantee that the wavelet
        starts approximately after time 0.

    Examples:

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from fatiando.seismic import RickerWavelet
        >>> w1 = RickerWavelet(f=5)
        >>> w2 = RickerWavelet(f=5, delay=0)
        >>> w3 = RickerWavelet(f=5, amp=0.5, delay=0.5)
        >>> times = np.linspace(0, 1, 200)
        >>> plt.figure()
        >>> plt.plot(times, w1(times), '-k')
        >>> plt.plot(times, w2(times), '--k')
        >>> plt.plot(times, w3(times), '-.k')
        >>> plt.show()
        >>> w = RickerWavelet(f=60)
        >>> samples = w.sample(dt=0.001)
        >>> plt.figure()
        >>> plt.plot(samples, '.k')
        >>> plt.show()

    """

    def __init__(self, f, amp=1, delay='default'):
        super().__init__(amp, delay)
        self.f = f
        if self.delay == 'default':
            # Standard delay to make the wavelet start at time zero and be
            # causal. A good approximation for the this is the time interval
            # between the two side lobes of the wavelet.
            self.delay = 1/f

    def __call__(self, time):
        """
        Evaluate the wavelet function at a given time.

        Parameters:

        * time : int, float or array
            The time(s) at which to evaluate the wavelet.

        Returns:

        * wavelet : int, float or array
            The wavelet function values at the given time(s).

        """
        t = time - self.delay
        aux = (np.pi*self.f*t)**2
        res = (1 - 2*aux)*np.exp(-aux)
        return self.amp*res
