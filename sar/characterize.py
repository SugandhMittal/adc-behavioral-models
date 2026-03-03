"""
characterize.py

Computes standard ADC performance metrics from the output code sequence:
    - FFT spectrum
    - SNDR (Signal-to-Noise-and-Distortion Ratio)
    - SFDR (Spurious-Free Dynamic Range)
    - ENOB (Effective Number of Bits)
    - DNL (Differential Non-Linearity)
    - INL (Integral Non-Linearity)

These are the same metrics measured in Cadence post-layout simulation
or on a real ADC test bench.

Author: Sugandh Mittal
Date Modified: 25 February 2026
"""

import numpy as np


def compute_fft(codes, n_bits, fs=1.0, coherent=True):
    """
    Compute the normalized power spectrum of the ADC output code sequence.

    For coherent sampling use rectangular window, no spectral leakage.
    For non-coherent signals set coherent=False to apply Hanning window.

    :param codes: Digital output codes from the ADC
    :type codes: np.ndarray
    :param n_bits: ADC resolution
    :type n_bits: int
    :param fs: Sampling frequency in Hz, default 1.0
    :type fs: float
    :param coherent: If True use rectangular window, if False use Hanning
    :type coherent: bool
    :return: Tuple of (frequency axis in Hz, power spectrum in dBFS)
    :rtype: tuple(np.ndarray, np.ndarray)
    """
    N = len(codes)
    normalized = codes / (2 ** (n_bits - 1)) - 1.0

    if coherent:
        windowed = normalized
        norm_factor = N / 2
    else:
        window = np.hanning(N)
        windowed = normalized * window
        norm_factor = np.sum(window) / 2

    spectrum = np.fft.rfft(windowed)
    power = (np.abs(spectrum) / norm_factor) ** 2
    power_db = 10 * np.log10(power + 1e-300)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    return freqs, power_db


def compute_sndr(codes, n_bits, signal_bin=None):
    """
    Compute SNDR (Signal-to-Noise and Distortion Ratio) in dBFS.

    SNDR = 10 * log10(signal_power / (noise + distortion power))

    Uses rectangular window assuming coherent sampling.
    Fundamental tone bin is auto-detected or supplied manually.

    :param codes: Digital output codes from the ADC
    :type codes: np.ndarray
    :param n_bits: ADC resolution
    :type n_bits: int
    :param signal_bin: FFT bin of fundamental tone, auto-detected if None
    :type signal_bin: int, optional
    :return: Tuple of (SNDR in dBFS, detected signal bin index)
    :rtype: tuple(float, int)
    """

    N = len(codes)
    normalized = codes / (2 ** (n_bits - 1)) - 1.0

    spectrum = np.fft.rfft(normalized)
    power = np.abs(spectrum) ** 2

    if signal_bin is None:
        # Find the peak bin (skip DC at bin 0)
        signal_bin = np.argmax(power[1:]) + 1

    signal_power = power[signal_bin]

    # All other power (except DC bin 0) is noise + distortion
    noise_power = np.sum(power[1:]) - signal_power

    sndr_db = 10 * np.log10(signal_power / (noise_power + 1e-300))
    return sndr_db, signal_bin


def compute_sfdr(codes, n_bits, signal_bin=None):
    """
    Compute SFDR (Spurious-Free Dynamic Range) in dBc.

    SFDR is the difference in dB between the fundamental tone
    and the largest spur in the spectrum.

    :param codes: Digital output codes from the ADC
    :type codes: np.ndarray
    :param n_bits: ADC resolution
    :type n_bits: int
    :param signal_bin: FFT bin of fundamental tone, auto-detected if None
    :type signal_bin: int, optional
    :return: SFDR in dBc
    :rtype: float
    """
    N = len(codes)
    normalized = codes / (2 ** (n_bits - 1)) - 1.0
    spectrum = np.fft.rfft(normalized)
    power = np.abs(spectrum) ** 2

    if signal_bin is None:
        signal_bin = np.argmax(power[1:]) + 1

    signal_power = power[signal_bin]
    spur_power = power.copy()
    spur_power[0] = 0
    spur_power[signal_bin] = 0
    largest_spur = np.max(spur_power)

    sfdr_db = 10 * np.log10(signal_power / (largest_spur + 1e-300))
    return sfdr_db


def compute_enob(sndr_db):
    """
    Compute ENOB (Effective Number of Bits) from SNDR.

    Standard formula: ENOB = (SNDR_dB - 1.76) / 6.02

    An ideal N-bit ADC has SNDR = 6.02*N + 1.76 dB, so ENOB = N
    for a perfect converter.

    :param sndr_db: SNDR in dBFS
    :type sndr_db: float
    :return: Effective number of bits
    :rtype: float
    """
    return (sndr_db - 1.76) / 6.02


def compute_dnl_inl(adc_instance, n_samples=100000):
    """
    Compute DNL and INL using the histogram code density method.

    Sweeps a ramp across the full ADC input range and counts how many
    samples fall into each output code bin. For a perfectly linear ADC
    with a uniform ramp, each bin should have equal counts.

    DNL[k] = (actual_bin_width[k] / ideal_bin_width) - 1  in LSB
    INL[k]  = cumulative sum of DNL up to code k           in LSB

    :param adc_instance: ADC model to characterize
    :type adc_instance: SARADC or subclass
    :param n_samples: Number of ramp samples, more = lower histogram noise
    :type n_samples: int
    :return: Tuple of (DNL in LSB, INL in LSB) for each output code
    :rtype: tuple(np.ndarray, np.ndarray)
    """
    ramp = np.linspace(adc_instance.vmin,
                       adc_instance.vref- adc_instance.lsb * 0.01,
                       n_samples)

    codes = adc_instance.conversion(ramp)
    n_levels = 2 ** adc_instance.n_bits
    counts = np.bincount(codes, minlength=n_levels).astype(float)

    ideal_count = n_samples / n_levels
    dnl = (counts - ideal_count) / ideal_count
    inl = np.cumsum(dnl)
    inl -= np.linspace(inl[0], inl[-1], n_levels)

    return dnl, inl


def print_summary(sndr_db, enob, sfdr_db, dnl, inl):
    """
    Print a formatted ADC performance summary table.

    :param sndr_db: SNDR in dBFS
    :type sndr_db: float
    :param enob: Effective number of bits
    :type enob: float
    :param sfdr_db: SFDR in dBc
    :type sfdr_db: float
    :param dnl: DNL array in LSB
    :type dnl: np.ndarray
    :param inl: INL array in LSB
    :type inl: np.ndarray
    """
    print("=" * 45)
    print("       SAR-ADC PERFORMANCE SUMMARY")
    print("=" * 45)
    print(f"  SNDR          : {sndr_db:+.2f} dBFS")
    print(f"  ENOB          : {enob:.2f} bits")
    print(f"  SFDR          : {sfdr_db:.2f} dBc")
    print(f"  DNL (peak)    : {np.max(np.abs(dnl)):.3f} LSB")
    print(f"  INL (peak)    : {np.max(np.abs(inl)):.3f} LSB")
    print("=" * 45)