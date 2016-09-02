from __future__ import division, print_function
from future.utils import with_metaclass
from future.builtins import super, object, range
from abc import ABCMeta, abstractmethod
import copy

import numpy as np


class BaseWavelet(with_metaclass(ABCMeta)):

    def __init__(self, amp):
        self.amp = amp

    @abstractmethod
    def __call__(self, time):
        pass

    def copy(self):
        return copy.deepcopy(self)


class GaussianWavelet(BaseWavelet):

    def __init__(self, amp, f_cut, delay=0):
        super().__init__(amp)
        self.f_cut = f_cut
        self.delay = delay

    def __call__(self, time):
        sqrt_pi = np.sqrt(np.pi)
        fc = self.f_cut/(3*sqrt_pi)
        # Standard delay to make the wavelet start at time zero and be causal
        td = time - 2*sqrt_pi/self.f_cut
        # Apply the user defined delay on top
        t = td - self.delay
        scale = self.amp/(2*np.pi*(np.pi*fc)**2)
        res = scale*np.exp(-np.pi*(np.pi*fc*t)**2)
        return res


class RickerWavelet(BaseWavelet):

    def __init__(self, amp, f_cut, delay=0):
        super().__init__(amp)
        self.f_cut = f_cut
        self.delay = delay

    def __call__(self, time):
        sqrt_pi = np.sqrt(np.pi)
        fc = self.f_cut/(3*sqrt_pi)
        # Standard delay to make the wavelet start at time zero and be causal
        td = time - 2*sqrt_pi/self.f_cut
        # Apply the user defined delay on top
        t = td - self.delay
        scale = self.amp*(2*np.pi*(np.pi*fc*t)**2 - 1)
        res = scale*np.exp(-np.pi*(np.pi*fc*t)**2)
        return res
