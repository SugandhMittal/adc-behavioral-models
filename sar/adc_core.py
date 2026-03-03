"""
adc_core.py

This is a behavioral model of an N-bit Successive Approximation Register ADC (SAR-ADC).

It models the ideal behaviour using capacitive DAC and a comparator.
The non-idealities are added in nonidealities.py which can be modified according to the parameters.

Author: Sugandh Mittal
Date Modified: 24 February 2026
"""

import numpy as np

class SARADC:
    """
    Ideal N-bit SAR-ADC behavioral model.

    :param n_bits: ADC resolution, default 8
    :type n_bits: int
    :param vref: Reference voltage in volts, default 1.0
    :type vref: float
    :param vmin: Minimum input voltage in volts, default 0.0
    :type vmin: float
    :param fs: Sampling frequency in Hz, default 10 MHz
    :type fs: float
    """

    def __init__(self, n_bits=8, vref=1.0, vmin=0.0, fs=10e6):
        self.n_bits = n_bits
        self.vref = vref
        self.vmin = vmin
        self.fs = fs
        self.dt = 1.0 / fs
        self.n_levels = 2 ** n_bits
        self.lsb = (vref - vmin) / self.n_levels #The step size

    def conversion(self, v_in):
        """
        Convert a single input voltage to a digital code using the SAR algorithm.

        This is similar to what happens inside the hardware:
        - Start with MSB = 1, all others 0
        - Compare CDAC output to input
        - Keep the bit if CDAC < input, clear it if CDAC > input
        - Move to next bit

        :param v_in: Input voltage to convert in volts
        :type v_in: float or np.ndarray
        :return: Digital output code
        :rtype: int or np.ndarray
        """

        v_in = np.atleast_1d(np.array(v_in, dtype=float))

        # Detects if the input value is out of the range of ADC
        self.overdriven_high = v_in > self.vref
        self.overdriven_low = v_in < self.vmin
        self.overdriven = self.overdriven_high | self.overdriven_low

        code = np.zeros(v_in.shape, dtype=int)

        for bit in range(self.n_bits - 1, -1, -1):
            # Switches
            trial_code = code | (1 << bit)

            # CDAC output
            v_cdac = self._dac(trial_code)

            # Comparator decision
            bit_accepted = v_cdac <= v_in
            code = np.where(bit_accepted, trial_code, code)

        return code

    def _dac(self, code):
        """
        Convert the digital code to voltage (in case of ideal CDAC)
        :param code: Digital code (either trial or final)
        :type code: ndarray
        :return: Corresponding voltage
        :rtype: float or ndarray
        """
        return self.vmin + code * self.lsb

    def outputcode_to_voltage(self, code):
        """
        Convert the final output code to analog voltage at the end.
        (Will use this for error between input and output signal)

        :param code: Digital output code to convert back to voltage
        :type code: int or np.ndarray
        :return: Reconstructed analog voltage in volts
        :rtype: float or np.ndarray
        """
        return self._dac(np.asarray(code))

    def convert_signal(self, signal):
        """
        Convert an array of time domain samples to digitized codes.

        :param signal: Array of input voltages in time domain
        :type signal: np.ndarray
        :return: Tuple of (digital output codes, reconstructed analog waveform, overdrive mask)
        :rtype: tuple(np.ndarray, np.ndarray, np.ndarray)
        """
        signal = np.atleast_1d(np.array(signal, dtype=float))
        codes = self.conversion(signal)
        reconstructed = self.outputcode_to_voltage(codes)

        return codes, reconstructed, self.overdriven
