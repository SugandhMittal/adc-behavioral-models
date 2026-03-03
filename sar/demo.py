"""
demo.py

Demonstration script for the SAR-ADC behavioral model.
Runs the ideal ADC and each non-ideality model, computes performance
metrics, and generates comparison plots.

Note: This visualization script was generated with AI assistance (Claude).
The core model, non-idealities, and characterization logic in adc_core.py,
nonidealities.py, and characterize.py were written independently.

Usage:
    python demo.py

Author: Sugandh Mittal
Date Modified: 25 February 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from adc_core import SARADC
from nonidealities import (ComparatorOffset, ThermalNoise,
                           CapacitorMismatch, ClockJitter, ReferenceNoise)
from characterize import (compute_fft, compute_sndr, compute_sfdr,
                          compute_enob, compute_dnl_inl, print_summary)

# Parameters
N_BITS    = 8
VREF      = 1.0
FS        = 10e6
N_SAMPLES = 4096
M_CYCLES  = 127
F_IN      = M_CYCLES * FS / N_SAMPLES
AMPLITUDE = 0.45
DC_OFFSET = 0.5

COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#00BCD4']
BASE   = dict(n_bits=N_BITS, vref=VREF, fs=FS)


def make_signal():
    t = np.arange(N_SAMPLES) / FS
    return DC_OFFSET + AMPLITUDE * np.sin(2 * np.pi * F_IN * t)


def run_adc(adc, signal):
    codes, _, _ = adc.convert_signal(signal)
    sndr, sig_bin = compute_sndr(codes, N_BITS)
    return {
        "sndr"    : sndr,
        "sfdr"    : compute_sfdr(codes, N_BITS, signal_bin=sig_bin),
        "enob"    : compute_enob(sndr),
        "fft"     : compute_fft(codes, N_BITS, fs=FS),
        "dnl_inl" : compute_dnl_inl(adc),
    }


def plot_col(gs, col, name, r, color):
    freqs, power_db = r["fft"]
    dnl, inl        = r["dnl_inl"]
    codes_axis      = np.arange(len(dnl))

    def label_box(ax, text, va='top'):
        ax.text(0.97, 0.95 if va == 'top' else 0.05, text,
                transform=ax.transAxes, fontsize=7, ha='right', va=va,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Row 0 — FFT
    ax0 = plt.subplot(gs[0, col])
    ax0.plot(freqs / 1e6, power_db, color=color, linewidth=0.8)
    ax0.axhline(-6.02 * N_BITS, color='gray', linestyle='--', linewidth=0.7)
    ax0.set_xlim(0, FS / 2e6)
    ax0.set_ylim(-120, 10)
    ax0.set_title(name, fontsize=8, fontweight='bold')
    ax0.set_xlabel("Frequency (MHz)", fontsize=7)
    if col == 0: ax0.set_ylabel("Power (dBFS)", fontsize=8)
    ax0.tick_params(labelsize=7)
    ax0.grid(True, alpha=0.3)
    label_box(ax0, f"SNDR={r['sndr']:.1f}dB\nENOB={r['enob']:.2f}b\nSFDR={r['sfdr']:.1f}dB",
              va='bottom')

    # Row 1 — DNL
    ax1 = plt.subplot(gs[1, col])
    dnl_peak = max(np.max(np.abs(dnl)), 0.1)
    ax1.bar(codes_axis, dnl, color=color, alpha=0.7, width=1.0)
    ax1.axhline(0, color='black', linewidth=0.7)
    ax1.set_xlim(0, 2 ** N_BITS)
    ax1.set_ylim(-dnl_peak * 1.5, dnl_peak * 1.5)
    ax1.set_xlabel("Output Code", fontsize=7)
    if col == 0: ax1.set_ylabel("DNL (LSB)")
    ax1.tick_params(labelsize=7)
    ax1.grid(True, alpha=0.3)
    label_box(ax1, f"Peak={dnl_peak:.3f} LSB")

    # Row 2 — INL
    ax2 = plt.subplot(gs[2, col])
    inl_peak = max(np.max(np.abs(inl)), 0.1)
    ax2.plot(codes_axis, inl, color=color, linewidth=1.2)
    ax2.fill_between(codes_axis, inl, alpha=0.2, color=color)
    ax2.axhline(0, color='black', linewidth=0.7)
    ax2.set_xlim(0, 2 ** N_BITS)
    ax2.set_ylim(-inl_peak * 1.5, inl_peak * 1.5)
    ax2.set_xlabel("Output Code")
    if col == 0: ax2.set_ylabel("INL (LSB)")
    ax2.tick_params(labelsize=7)
    ax2.grid(True, alpha=0.3)
    label_box(ax2, f"Peak={inl_peak:.3f} LSB")


def main():
    signal = make_signal()

    models = {
        "Ideal"                : SARADC(**BASE),
        "Comp. Offset\n(+5mV)" : ComparatorOffset(offset_v=0.005,       **BASE),
        "kT/C\n(C=100fF)"      : ThermalNoise(C_sample=100e-15,         **BASE),
        "Cap. Mismatch\n(0.5%)": CapacitorMismatch(mismatch_sigma=0.050, **BASE),
        "Clock Jitter\n(1ps)"  : ClockJitter(jitter_rms=1e-12,          **BASE),
        "Ref. Noise\n(1mV)"    : ReferenceNoise(ref_noise_rms=1e-3,     **BASE),
    }

    results = {name: run_adc(adc, signal) for name, adc in models.items()}

    # Print summary
    print(f"\nInput: {F_IN/1e3:.1f} kHz, A={AMPLITUDE}V, "
          f"fs={FS/1e6:.0f}MHz, {N_BITS}-bit\n")

    for name, r in results.items():
        dnl, inl = r["dnl_inl"]
        print(f"\n[{name.replace(chr(10), ' ')}]")
        print_summary(r["sndr"], r["enob"], r["sfdr"], dnl, inl)

    # Plot
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(f"SAR-ADC: {N_BITS}-bit, fs={FS/1e6:.0f}MHz, "
                 f"fin={F_IN/1e3:.1f}kHz", y=0.98)

    gs = gridspec.GridSpec(3, len(models), figure=fig, hspace=0.5, wspace=0.35)

    for col, (name, r) in enumerate(results.items()):
        plot_col(gs, col, name, r, COLORS[col])

    for label, row in [("FFT Spectrum", 0), ("DNL", 1), ("INL", 2)]:
        fig.text(0.005, 0.83 - row * 0.31, label,
                 fontsize=10, fontweight='bold', rotation=90, va='center')

    plt.savefig('sar_adc_characterization.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved: sar_adc_characterization.png")
    plt.show()


if __name__ == "__main__":
    main()