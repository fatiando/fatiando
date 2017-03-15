from __future__ import absolute_import
from pytest import raises
from .. import RickerWavelet


def test_ricker_copy():
    "Make sure the copy method doesn't give the same object"
    w = RickerWavelet(f=3, amp=4, delay=5)
    wcopy = w.copy()
    wcopy.f = 1
    wcopy.amp = -2
    wcopy.delay = -3
    assert w is not wcopy, 'Not a copy'
    assert w.f != wcopy.f, 'frequencies are the same'
    assert w.amp != wcopy.amp, 'amplitudes are the same'
    assert w.delay != wcopy.delay, 'delays are the same'


def test_ricker_fail_zero_frequency():
    "Wavelet creation should fail if f=0"
    with raises(AssertionError):
        RickerWavelet(f=-1)
    with raises(AssertionError):
        RickerWavelet(f=0)
