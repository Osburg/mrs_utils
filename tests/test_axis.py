import numpy as np
import pytest

from mrs_utils.axis import Axis
from mrs_utils.constants import CS_0, GAMMA_H1


# Amir's conversion functions
def ppm2p(f0, dt, r, len):
    r = CS_0 - r
    return int(f0 * r * dt * len + len / 2)


def ppm2f(self, r):
    return r * self.trnfreq


def test_axis():
    time = np.linspace(0, 9, 10)

    a1 = Axis.from_time_axis(time, b0=1)
    a2 = Axis.from_frequency_axis(a1._frequency, b0=1)
    a3 = Axis.from_ppm_axis(a1._ppm, b0=1)

    assert a1.b0 == 1
    assert a1.cs0 == CS_0
    assert np.allclose(a1._time, time)

    assert np.allclose(a1._time, a2._time)
    assert np.allclose(a1._frequency, a2._frequency)
    assert np.allclose(a1._ppm, a2._ppm)

    assert np.allclose(a1._time, a3._time)
    assert np.allclose(a1._frequency, a3._frequency)
    assert np.allclose(a1._ppm, a3._ppm)

    time = np.linspace(0, 9, 20)

    a1 = Axis.from_time_axis(time, b0=1)
    a2 = Axis.from_frequency_axis(a1._frequency, b0=1)
    a3 = Axis.from_ppm_axis(a1._ppm, b0=1)

    assert np.allclose(a1._time, a2._time)
    assert np.allclose(a1._frequency, a2._frequency)
    assert np.allclose(a1._ppm, a2._ppm)

    assert np.allclose(a1._time, a3._time)
    assert np.allclose(a1._frequency, a3._frequency)
    assert np.allclose(a1._ppm, a3._ppm)


def test_to_index():

    time = np.linspace(0, 9, 10)

    a1 = Axis.from_time_axis(time, b0=1)

    assert a1.to_index(value=0.4, domain="time") == 0
    assert a1.to_index(value=0.6, domain="time") == 1
    with pytest.warns():
        a1.to_index(value=10.5, domain="time")

    assert a1.to_index(value=42.57747823, domain="frequency") == 2

    assert a1.to_index(value=4.705, domain="ppm") == 2

    # check that the conversion is the same as Amir's up to rounding errors
    for i, x in enumerate(np.linspace(np.min(a1._ppm), np.max(a1._ppm), 100)):
        assert (
            a1.to_index(value=x, domain="ppm") == ppm2p(GAMMA_H1, 1, x, 10)
            or a1.to_index(value=x, domain="ppm") == ppm2p(GAMMA_H1, 1, x, 10) - 1
        )


def test_frequency_to_ppm():
    time = np.linspace(0, 9, 10)

    a1 = Axis.from_time_axis(time, b0=1)

    assert a1.frequency_to_ppm(GAMMA_H1, frequency_shift=True) == 1
    assert a1.ppm_to_frequency(1, frequency_shift=True) == GAMMA_H1

    assert np.allclose(
        a1.frequency_to_ppm(GAMMA_H1 * (1 - CS_0 * 1e-6), frequency_shift=False), 0
    )
    assert np.allclose(
        a1.ppm_to_frequency(0, frequency_shift=False), GAMMA_H1 * (1 - CS_0 * 1e-6)
    )
