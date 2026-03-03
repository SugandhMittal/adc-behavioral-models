"""
frequency_sweep.py

ENOB vs input frequency sweep for the SAR-ADC behavioral model.

Sweeps the input frequency from low to near-Nyquist and plots how ENOB
degrades due to clock jitter. Overlays the theoretical jitter limit curve
to validate the dynamic jitter model.

Theoretical jitter limit:
    ENOB_jitter = log2(1 / (2 * pi * f_in * t_jitter))

Usage:
    python frequency_sweep.py

Author: Sugandh Mittal
Date Modified: 03 March 2026
"""

import numpy as np
import matplotlib.pyplot as plt

from adc_core import SARADC
from nonidealities import ClockJitter
from characterize import compute_sndr, compute_enob, compute_fft

# Parameters
N_BITS      = 8
VREF        = 1.0
N_SAMPLES   = 4096
AMPLITUDE   = 0.45
DC_OFFSET   = 0.5
FS         = 1e9
# Note: 1ps is realistic jitter for a GHz frequency but 10ps makes
# the visualization better
JITTER_RMS = 10e-12

BASE = dict(n_bits=N_BITS, vref=VREF, fs=FS)
# Sweep is limited to 0.2 * FS because above this, jitter noise sidebands
# alias back near the fundamental in the time-domain model, causing
# SNDR to be artificially overestimated. A frequency-domain jitter
# model would be needed for accurate results near Nyquist which is way
# complex and can be only done for standard signals.
MAX_FREQ =  0.2 * FS

# M values are chosen as primes to avoid harmonic aliasing.
# Harmonics of the input (2M, 3M) land on unique FFT bins
# since primes share no common factors with N_SAMPLES (4096 = 2^12).
# This is standard practice per IEEE 1241 for ADC testing.
M_VALUES    = [m for m in [3, 7, 13, 23, 37, 53, 71, 97, 127, 163, 197,
                            251, 307, 379, 449, 521, 601, 701, 811, 937,
                            1009, 1117, 1229, 1367, 1499]
               if m * FS / N_SAMPLES < MAX_FREQ]
FREQUENCIES = np.array([m * FS / N_SAMPLES for m in M_VALUES])

def make_signal(f_in):
    t = np.arange(N_SAMPLES) / FS
    return DC_OFFSET + AMPLITUDE * np.sin(2 * np.pi * f_in * t)

def run_sweep(adc_class, label_for_adc, **kwargs):
    """
    Sweep input frequency and compute ENOB at each point.

    :param adc_class: ADC class to instantiate (SARADC or ClockJitter)
    :type adc_class: class
    :param label_for_adc: Label for progress printing
    :type label_for_adc: str
    :return: Array of ENOB values at each frequency
    :rtype: np.ndarray
    """
    enobs = []
    print(f"\nRunning sweep: {label_for_adc}")

    for f_in in FREQUENCIES:
        adc = adc_class(**kwargs, **BASE)
        signal = make_signal(f_in)
        codes, _, _ = adc.convert_signal(signal)
        _, _, power = compute_fft(codes, N_BITS, fs=FS)
        sndr, sig_bin = compute_sndr(power)
        enobs.append(compute_enob(sndr))

    return np.array(enobs)

def theoretical_combined(frequencies, jitter_rms, n_bits):
    """
    Calculates the combined noise due to jitter and the quantization

    :param frequencies: Input frequency array in Hz
    :type frequencies: np.ndarray
    :param jitter_rms: RMS jitter in seconds
    :type jitter_rms: float
    :param n_bits: Number of bits
    :type n_bits: int
    :return: Theoretical ENOB limit at each frequency
    :rtype: np.ndarray
    """
    enob_jitter = np.log2(1.0 / (2 * np.pi * frequencies * jitter_rms))
    enob_quant  = n_bits - 0.5  # quantization limit
    # Total noise power combines both
    sndr_jitter = 6.02 * enob_jitter + 1.76
    sndr_quant  = 6.02 * enob_quant  + 1.76
    # Convert to linear power, add, convert back
    noise_total = 10**(-sndr_jitter/10) + 10**(-sndr_quant/10)
    sndr_total  = -10 * np.log10(noise_total)
    return (sndr_total - 1.76) / 6.02

def print_summary(frequencies, enob_ideal, enob_jitter):
    enob_theory = np.clip(theoretical_combined(frequencies, JITTER_RMS, N_BITS), 0, N_BITS + 0.5)

    print(f"  ENOB vs FREQUENCY: {JITTER_RMS*1e12:.1f} ps jitter")
    print(f"  {'Frequency':>12}  {'Ideal':>8}  {'Jitter':>8}  {'Theory':>8}")

    n = len(frequencies)
    indices = np.linspace(0, n - 1, min(7, n), dtype=int)

    for i in indices:
        print(f"  {frequencies[i]/1e3:>10.1f} kHz   {enob_ideal[i]:>8.2f}   {enob_jitter[i]:>8.2f}  {enob_theory[i]:>8.2f}")

def plot_sweep(frequencies, enob_ideal, enob_jitter):
    """
    Plot ENOB vs frequency for ideal and jitter-affected ADC,
    with theoretical jitter limit overlay.

    :param frequencies: Input frequency array in Hz
    :type frequencies: np.ndarray
    :param enob_ideal: ENOB array for ideal ADC
    :type enob_ideal: np.ndarray
    :param enob_jitter: ENOB array for jitter ADC
    :type enob_jitter: np.ndarray
    """
    freq_mhz    = frequencies / 1e6
    enob_theory = np.clip(theoretical_combined(frequencies, JITTER_RMS, n_bits=8), 0, N_BITS + 0.5)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(freq_mhz, enob_ideal, 'o-', color='deepskyblue', label='Ideal ADC')
    ax.semilogx(freq_mhz, enob_jitter, 's-', color='orange', label=f'Clock Jitter ({JITTER_RMS*1e12:.1f} ps RMS)')
    ax.semilogx(freq_mhz, enob_theory, '--', color='orange', label='Theoretical jitter limit')

    # Mark Nyquist
    ax.axvline(FS / 2e6, color='gray',linestyle=':', label=f'Nyquist ({FS/2e6:.1f} MHz)')

    # Mark quantization limit
    ax.axhline(N_BITS - 0.5, color='green', linestyle='--', label=f'Quantization limit ({N_BITS-0.5} bits)')

    ax.set_xlabel("Input Frequency (MHz)")
    ax.set_ylabel("ENOB (bits)")
    ax.set_title(f"ENOB vs Input Frequency: {N_BITS}-bit SAR-ADC, fs={FS/1e9:.1f} GHz, {JITTER_RMS*1e12:.1f} ps jitter")
    ax.set_ylim(0, max(np.max(enob_theory), N_BITS) + 1)
    ax.set_xlim(freq_mhz[0], freq_mhz[-1])
    ax.legend()
    ax.grid(alpha = 0.3)
    plt.savefig('enob_vs_frequency.png')
    print("\nPlot saved: enob_vs_frequency.png")
    plt.show()


if __name__ == "__main__":
    enob_ideal  = run_sweep(SARADC,      "Ideal ADC")
    enob_jitter = run_sweep(ClockJitter, "Clock Jitter", jitter_rms=JITTER_RMS)
    print_summary(FREQUENCIES, enob_ideal, enob_jitter)
    plot_sweep(FREQUENCIES, enob_ideal, enob_jitter)