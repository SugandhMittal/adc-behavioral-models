"""
nonidealities.py

Non-idealities can be added to SARADC to simulate the behavior of real life circuits.

Implemented Non-idealities:
    - ThermalNoise      : kT/C noise on the sampling capacitor
    - ClockJitter       : aperture jitter on the sampling clock
    - ComparatorOffset  : static input-referred comparator offset
    - CapacitorMismatch : random mismatch in binary-weighted CDAC
    - ReferenceNoise    : thermal noise on the reference voltage
    - OffsetError       : static offset shift of the transfer curve

Author: Sugandh Mittal
Date Modified: 25 February 2026
"""

import numpy as np
from adc_core import SARADC


class ThermalNoise(SARADC):
    """
    Models the kT/C thermal noise on the sampling capacitor.

    This noise is introduced when the sampling switch opens,
    freezing a random thermal noise voltage onto the capacitor.

    v_noise_rms = sqrt(kT/C)

    :param C_sample: Sampling capacitor value in Farads, default 1 pF
    :type C_sample: float
    :param T_kelvin: Temperature in Kelvin, default 300 K
    :type T_kelvin: float
    """

    def __init__(self, C_sample=1e-12, T_kelvin=300.0, **kwargs):
        super().__init__(**kwargs)
        k_B = 1.380649e-23  # Boltzmann constant J/K
        self.noise_rms = np.sqrt(k_B * T_kelvin / C_sample)

    def conversion(self, v_in):
        """
        Adds kT/C thermal noise to input before conversion.

        :param v_in: Input voltage in volts
        :type v_in: float or np.ndarray
        :return: Digital output code
        :rtype: int or np.ndarray
        """
        v_in = np.atleast_1d(np.array(v_in, dtype=float))
        noise = np.random.normal(0, self.noise_rms, size=v_in.shape)
        return super().conversion(v_in + noise)


class ClockJitter(SARADC):
    """
    Models aperture jitter on the sampling clock.

    The clock jitter results in the sample being taken at time t + delta_t
    instead of time t. For any input waveform the voltage error is:

    delta_v = dv/dt * delta_t

    The instantaneous slope is computed numerically using np.gradient,
    making the model valid for any input waveform, not just sinusoids.

    :param jitter_rms: RMS clock jitter in seconds, default 1 ps
    :type jitter_rms: float
    """

    def __init__(self, jitter_rms=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.jitter_rms = jitter_rms

    def conversion(self, v_in):
        """
        Adds aperture jitter noise to input before conversion.

        :param v_in: Input voltage in volts
        :type v_in: float or np.ndarray
        :return: Digital output code
        :rtype: int or np.ndarray
        """
        v_in = np.atleast_1d(np.array(v_in, dtype=float))
        instantaneous_slope = np.gradient(v_in, self.dt)
        delta_t = np.random.normal(0, self.jitter_rms, size=v_in.shape)
        jitter_noise = instantaneous_slope * delta_t
        return super().conversion(v_in + jitter_noise)


class ComparatorOffset(SARADC):
    """
    Models a static input-referred offset on the comparator.

    In a real ADC the comparator threshold is shifted by a random
    offset Vos, causing the transfer curve to shift horizontally.

    :param offset_v: Offset voltage in volts, can be positive or negative
    :type offset_v: float
    """

    def __init__(self, offset_v=0.1, **kwargs):
        super().__init__(**kwargs)
        self.offset_v = offset_v

    def conversion(self, v_in):
        """
        Subtracts comparator offset from input before conversion.

        :param v_in: Input voltage in volts
        :type v_in: float or np.ndarray
        :return: Digital output code
        :rtype: int or np.ndarray
        """
        v_in = np.atleast_1d(np.array(v_in, dtype=float))
        return super().conversion(v_in - self.offset_v)


class CapacitorMismatch(SARADC):
    """
    Models random mismatch in the binary-weighted CDAC capacitors.

    In a real CDAC, each capacitor C_k = 2^k * C_unit has a random
    error: C_k_actual = C_k * (1 + epsilon_k), where epsilon_k is
    drawn from a Gaussian distribution. This causes non-linearity
    because the DAC voltage steps are no longer perfectly binary-weighted.

    :param mismatch_sigma: Fractional standard deviation of capacitor mismatch
    :type mismatch_sigma: float
    """

    def __init__(self, mismatch_sigma=0.001, **kwargs):
        super().__init__(**kwargs)
        self.mismatch_sigma = mismatch_sigma
        # Fixed seed so one ADC instance has consistent mismatch
        rng = np.random.default_rng(seed=42)
        self.cap_errors = rng.normal(0, mismatch_sigma, size=self.n_bits)

    def _dac(self, code):
        """
        Overrides ideal DAC with mismatch-affected capacitor weights.

        :param code: Digital code to convert
        :type code: int or np.ndarray
        :return: Corresponding voltage with mismatch error
        :rtype: float or np.ndarray
        """
        code = np.atleast_1d(np.asarray(code, dtype=int))
        voltage = np.zeros(code.shape, dtype=float)

        for bit in range(self.n_bits):
            bit_set = (code >> bit) & 1
            ideal_weight = 2 ** bit
            actual_weight = ideal_weight * (1 + self.cap_errors[bit])
            voltage += bit_set * actual_weight

        ideal_fullscale = self.n_levels - 1
        voltage = self.vmin + (voltage / ideal_fullscale) * (self.vref - self.vmin)
        return voltage if voltage.size > 1 else float(voltage[0])


class ReferenceNoise(SARADC):
    """
    Models thermal noise on the ADC reference voltage.

    Even a well-regulated reference has small random fluctuations
    each conversion cycle due to thermal noise in the bandgap reference.
    Since the entire CDAC is ratiometric to vref, any noise on it
    directly scales all bit decisions.

    :param ref_noise_rms: RMS reference voltage noise in volts
    :type ref_noise_rms: float
    """

    def __init__(self, ref_noise_rms=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.ref_noise_rms = ref_noise_rms

    def conversion(self, v_in):
        """
        Applies random reference voltage fluctuation before conversion.

        :param v_in: Input voltage in volts
        :type v_in: float or np.ndarray
        :return: Digital output code
        :rtype: int or np.ndarray
        """
        v_in = np.atleast_1d(np.array(v_in, dtype=float))
        ref_noise = np.random.normal(0, self.ref_noise_rms, size=v_in.shape)
        noisy_vref = self.vref + ref_noise
        v_in_scaled = v_in * (self.vref / noisy_vref)
        return super().conversion(v_in_scaled)

