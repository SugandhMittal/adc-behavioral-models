"""
monte_carlo.py

Monte Carlo yield analysis for the SAR-ADC behavioral model.

Runs N iterations across multiple mismatch sigma values to simulate chips
from different process nodes. For each sigma, computes ENOB, SNDR, DNL,
and INL distributions and plots yield vs mismatch curve.

Usage:
    python monte_carlo.py

Author: Sugandh Mittal
Date Modified: 27 February 2026
"""

import numpy as np
import matplotlib.pyplot as plt

from nonidealities import CapacitorMismatch
from characterize import compute_sndr, compute_enob, compute_dnl_inl

# Parameters
N_BITS       = 8
VREF         = 1.0
FS           = 10e6
N_SAMPLES    = 4096
M_CYCLES     = 127
F_IN         = M_CYCLES * FS / N_SAMPLES
AMPLITUDE    = 0.45
DC_OFFSET    = 0.5
N_ITERATIONS = 500

# Sigma values to sweep
SIGMAS = [0.002, 0.005, 0.007, 0.01, 0.015, 0.02]

BASE = dict(n_bits=N_BITS, vref=VREF, fs=FS)

def make_signal():
    t = np.arange(N_SAMPLES) / FS
    return DC_OFFSET + AMPLITUDE * np.sin(2 * np.pi * F_IN * t)

def run_sigma(sigma, signal):
    """
    Run N_ITERATIONS chips for a single mismatch sigma value.

    :param sigma: Capacitor mismatch fractional sigma
    :type sigma: float
    :param signal: Input test signal
    :type signal: array
    :return: Dict of metric arrays, one value per chip
    :rtype: dict
    """
    metrics = {"sndr": [], "enob": [], "dnl_peak": [], "inl_peak": []}

    for seed in range(N_ITERATIONS):
        adc = CapacitorMismatch(mismatch_sigma=sigma, **BASE)
        adc.cap_errors = np.random.default_rng(seed=seed).normal(
            0, sigma, size=N_BITS
        )
        codes, _, _ = adc.convert_signal(signal)
        sndr, _     = compute_sndr(codes, N_BITS)
        dnl, inl    = compute_dnl_inl(adc)

        metrics["sndr"].append(sndr)
        metrics["enob"].append(compute_enob(sndr))
        metrics["dnl_peak"].append(np.max(np.abs(dnl)))
        metrics["inl_peak"].append(np.max(np.abs(inl)))

    return {k: np.array(v) for k, v in metrics.items()}


def run_all_sigmas(signal):
    """
    Run Monte Carlo simulation across all sigma values in SIGMAS.

    :param signal: Input test signal
    :type signal: np.ndarray
    :return: Dict mapping sigma value to its metrics dict
    :rtype: dict
    """
    all_results = {}

    for sigma in SIGMAS:
        print(f"  σ={sigma*100:.2f}%  is running")
        all_results[sigma] = run_sigma(sigma, signal)

    return all_results


def print_statistics(all_results):
    """
    Print summary statistics for each sigma value.

    :param all_results: Dict mapping sigma to metrics dict
    :type all_results: dict
    """
    print(f"  {'Sigma':>6}  {'ENOB mean':>10} {'ENOB std':>10} "
          f"{'DNL peak':>10} {'Yield':>8}")

    for sigma, r in all_results.items():
        yield_pct = np.mean(r["dnl_peak"] < 1.0) * 100
        print(f"  {sigma*100:>5.2f}%  "
              f"{np.mean(r['enob']):>10.3f} "
              f"{np.std(r['enob']):>10.3f} "
              f"{np.mean(r['dnl_peak']):>10.3f} "
              f"{yield_pct:>7.1f}%")

def plot_distributions(all_results):
    """
    Plot ENOB and DNL distributions for each sigma as overlapping histograms.

    :param all_results: Dict mapping sigma to metrics dict
    :type all_results: dict
    """
    colors = ['#2196F3', '#03A9F4', '#4CAF50', '#8BC34A',
              '#FF9800', '#FF5722', '#E91E63', '#9C27B0', '#673AB7']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Monte Carlo Distributions for {N_BITS}-bit SAR-ADC")
    for i, (sigma, r) in enumerate(all_results.items()):
        color = colors[i % len(colors)]
        label = f"σ={sigma*100:.2f}%"

        # ENOB distribution
        axes[0].hist(r["enob"], bins=30, color=color, alpha=0.4,
                     label=label, edgecolor='none')

        # DNL peak distribution
        axes[1].hist(r["dnl_peak"], bins=30, color=color, alpha=0.4,
                     label=label, edgecolor='none')

    axes[0].set_xlabel("ENOB (bits)", fontsize=10)
    axes[0].set_ylabel("Number of chips", fontsize=10)
    axes[0].set_title("ENOB Distribution", fontsize=10)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].axvline(1.0, color='red',
                    linestyle='--', label='DNL = 1 LSB limit')
    axes[1].set_xlabel("DNL Peak (LSB)")
    axes[1].set_ylabel("Number of chips")
    axes[1].set_title("DNL Peak Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('monte_carlo_distributions.png', dpi=150, bbox_inches='tight')
    print("Plot saved: monte_carlo_distributions.png")
    plt.show()


def plot_yield_sweep(all_results):
    """
    Plot yield vs mismatch sigma curve derived from all_results.

    :param all_results: Dict mapping sigma to metrics dict
    :type all_results: dict
    """
    sigmas = np.array(list(all_results.keys()))
    yields = np.array([
        np.mean(r["dnl_peak"] < 1.0) * 100
        for r in all_results.values()
    ])
    enob_means = np.array([np.mean(r["enob"]) for r in all_results.values()])
    enob_stds  = np.array([np.std(r["enob"])  for r in all_results.values()])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Yield & ENOB vs Mismatch: {N_BITS}-bit SAR-ADC")

    axes[0].plot(sigmas * 100, yields, 'o-', color='#2196F3',
                 linewidth=2, markersize=7, markerfacecolor='white',
                 markeredgewidth=2, label='Simulated yield')
    axes[0].axvspan(0, 0.5, alpha=0.08, color='red',
                    label='Statistically unreliable (σ < 0.5%)')
    axes[0].axvline(0.5, color='red', linewidth=1.0, linestyle='--', alpha=0.6)
    axes[0].set_xlabel("Capacitor Mismatch σ (%)")
    axes[0].set_ylabel("Yield: DNL < 1 LSB (%)")
    axes[0].set_title("Yield vs Mismatch")
    axes[0].set_xlim(0, max(sigmas * 100) * 1.1)
    axes[0].set_ylim(0, 55)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ENOB mean ± std curve
    axes[1].plot(sigmas * 100, enob_means, 'o-', color='#FF5722',
                 linewidth=2, markersize=7, markerfacecolor='white',
                 markeredgewidth=2, label='Mean ENOB')
    axes[1].fill_between(sigmas * 100,
                          enob_means - enob_stds,
                          enob_means + enob_stds,
                          alpha=0.2, color='#FF5722', label='±1σ')
    axes[1].axhline(N_BITS - 0.5, color='green', linestyle='--',
                    linewidth=1.0, label=f'{N_BITS - 0.5} bits target')
    axes[1].set_xlabel("Capacitor Mismatch σ (%)")
    axes[1].set_ylabel("ENOB (bits)")
    axes[1].set_title("ENOB vs Mismatch")
    axes[1].set_xlim(0, max(sigmas * 100) * 1.1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('yield_sweep.png', dpi=150, bbox_inches='tight')
    print("Plot saved: yield_sweep.png")
    plt.show()


if __name__ == "__main__":
    signal = make_signal()

    print(f"Running Monte Carlo: {len(SIGMAS)} sigma values x "
          f"{N_ITERATIONS} chips each\n")

    all_results = run_all_sigmas(signal)
    print_statistics(all_results)
    plot_distributions(all_results)
    plot_yield_sweep(all_results)