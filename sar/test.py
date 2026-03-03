"""
test.py

Unit tests for the SAR-ADC behavioral model.
Tests each module independently to verify correct behavior.

Note: This test suite was generated with AI assistance (Claude, Anthropic)
and verified by Sugandh Mittal.

Run with: python test.py

Author: Sugandh Mittal
Date Modified: 25 February 2026
"""

import numpy as np
import sys

# Test counters
passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}{' — ' + detail if detail else ''}")
        failed += 1

from adc_core import SARADC

adc = SARADC(n_bits=8, vref=1.0, vmin=0.0, fs=10e6)

# Basic attributes
check("LSB value",         abs(adc.lsb - 1/256) < 1e-10)
check("n_levels",          adc.n_levels == 256)
check("dt = 1/fs",         abs(adc.dt - 1/10e6) < 1e-20)

# Known conversions
check("Zero input → code 0",     adc.conversion(np.array([0.0]))[0] == 0)
check("Mid scale → code ~128",   abs(adc.conversion(np.array([0.5]))[0] - 128) <= 1)
check("Full scale → code 255",   adc.conversion(np.array([1.0 - adc.lsb]))[0] == 255)

# Overdrive detection
adc.conversion(np.array([1.5]))
check("Overdrive high detected",  adc.overdriven_high[0] == True)

adc.conversion(np.array([-0.1]))
check("Overdrive low detected",   adc.overdriven_low[0] == True)

adc.conversion(np.array([0.5]))
check("No overdrive in range",    adc.overdriven[0] == False)

# Saturation behaviour (no clip, algorithm saturates naturally)
high_code = adc.conversion(np.array([2.0]))[0]
low_code  = adc.conversion(np.array([-1.0]))[0]
check("Overdrive high saturates at 255",  high_code == 255)
check("Overdrive low saturates at 0",     low_code  == 0)

# Output code to voltage
v = adc.outputcode_to_voltage(128)
check("Code 128 → ~0.5 V",   abs(v - 0.5) < adc.lsb)

# convert_signal returns correct shapes
sig = np.linspace(0.1, 0.9, 100)
codes, recon, overdriven = adc.convert_signal(sig)
check("convert_signal codes shape",      codes.shape    == (100,))
check("convert_signal recon shape",      recon.shape    == (100,))
check("convert_signal overdriven shape", overdriven.shape == (100,))
check("No overdrive for in-range signal", np.sum(overdriven) == 0)

# Quantization error < 1 LSB
q_error = np.abs(sig - recon)
check("Quantization error < 1 LSB",  np.all(q_error < adc.lsb))


# Non-idealities
from nonidealities import (ThermalNoise, ClockJitter,
                           ComparatorOffset, CapacitorMismatch,
                           ReferenceNoise)

kwargs = dict(n_bits=8, vref=1.0, vmin=0.0, fs=10e6)
N = 4096
t = np.arange(N) / 10e6
signal = 0.5 + 0.45 * np.sin(2 * np.pi * 310e3 * t)

# ThermalNoise
tn = ThermalNoise(C_sample=1e-12, **kwargs)
check("ThermalNoise noise_rms > 0",        tn.noise_rms > 0)
check("ThermalNoise noise_rms < 1 LSB",    tn.noise_rms < adc.lsb)
codes_tn, _, _ = tn.convert_signal(signal)
check("ThermalNoise output in range",      np.all((codes_tn >= 0) & (codes_tn <= 255)))

# ClockJitter
cj = ClockJitter(jitter_rms=1e-12, **kwargs)
check("ClockJitter jitter_rms stored",     cj.jitter_rms == 1e-12)
codes_cj, _, _ = cj.convert_signal(signal)
check("ClockJitter output in range",       np.all((codes_cj >= 0) & (codes_cj <= 255)))

# ComparatorOffset — positive offset shifts codes down
co_pos = ComparatorOffset(offset_v=0.1, **kwargs)
codes_co, _, _ = co_pos.convert_signal(signal)
codes_ideal, _, _ = SARADC(**kwargs).convert_signal(signal)
check("ComparatorOffset shifts codes",     np.mean(codes_co) < np.mean(codes_ideal))

# ComparatorOffset — negative offset shifts codes up
co_neg = ComparatorOffset(offset_v=-0.1, **kwargs)
codes_co_neg, _, _ = co_neg.convert_signal(signal)
check("Negative offset shifts codes up",  np.mean(codes_co_neg) > np.mean(codes_ideal))

# CapacitorMismatch
cm = CapacitorMismatch(mismatch_sigma=0.01, **kwargs)
check("CapacitorMismatch cap_errors length", len(cm.cap_errors) == 8)
codes_cm, _, _ = cm.convert_signal(signal)
check("CapacitorMismatch output in range",   np.all((codes_cm >= 0) & (codes_cm <= 255)))

# ReferenceNoise
rn = ReferenceNoise(ref_noise_rms=1e-4, **kwargs)
codes_rn, _, _ = rn.convert_signal(signal)
check("ReferenceNoise output in range",    np.all((codes_rn >= 0) & (codes_rn <= 255)))


# characterize.py
from characterize import (compute_fft, compute_sndr, compute_sfdr,
                          compute_enob, compute_dnl_inl)
FS = 10e6
N = 4096
M_CYCLES = 127
F_IN = M_CYCLES * FS / N   # exact coherent frequency
t = np.arange(N) / FS
signal = 0.5 + 0.45 * np.sin(2 * np.pi * F_IN * t)

kwargs = dict(n_bits=8, vref=1.0, vmin=0.0, fs=FS)

ideal_adc = SARADC(**kwargs)
codes_ideal, _, _ = ideal_adc.convert_signal(signal)

print(f"  DEBUG codes type: {type(codes_ideal)}, shape: {np.array(codes_ideal).shape}")
print(f"  DEBUG codes range: {codes_ideal.min()} to {codes_ideal.max()}")
print(f"  DEBUG unique codes: {len(np.unique(codes_ideal))}")

# FFT
freqs, power_db, power = compute_fft(codes_ideal, n_bits=8, fs=10e6)
check("FFT freqs length",        len(freqs) == N // 2 + 1)
check("FFT max frequency",       abs(freqs[-1] - 5e6) < 1)
check("FFT power has peak",      np.max(power_db) > -20)

# SNDR
sndr, sig_bin = compute_sndr(power)
check("SNDR > 40 dB for ideal",  sndr > 40)
check("SNDR < 60 dB for 8-bit",  sndr < 60)
check("Signal bin detected",     sig_bin > 0)

# SFDR
sfdr = compute_sfdr(power)
check("SFDR > SNDR",             sfdr > sndr)
check("SFDR reasonable range",   20 < sfdr < 100)

# ENOB
enob = compute_enob(sndr)
check("ENOB close to 8 for ideal", abs(enob - 8) < 0.5)
check("ENOB = (SNDR-1.76)/6.02",   abs(enob - (sndr - 1.76) / 6.02) < 1e-10)

# DNL / INL
dnl, inl = compute_dnl_inl(ideal_adc)
check("DNL length = 256",         len(dnl) == 256)
check("INL length = 256",         len(inl) == 256)
check("Ideal DNL peak < 1 LSB",   np.max(np.abs(dnl)) < 1.0)
check("Ideal INL peak < 1 LSB",   np.max(np.abs(inl)) < 1.0)

# Non-idealities degrade SNDR
_, _, power_tn = compute_fft(codes_tn, n_bits=8)
sndr_tn, _ = compute_sndr(power_tn)
_, _, power_cm = compute_fft(codes_cm, n_bits=8)
sndr_cm, _ = compute_sndr(power_cm)
check("ThermalNoise degrades SNDR",    sndr_tn <= sndr + 1)
check("CapacitorMismatch degrades SNDR", sndr_cm < sndr)


#Final Results

total = passed + failed
print(f"\n  {passed}/{total} tests passed")

if failed > 0:
    print(f"  {failed} test(s) failed, fix before committing\n")
    sys.exit(1)
else:
    print("  All tests passed, safe to commit\n")
    sys.exit(0)