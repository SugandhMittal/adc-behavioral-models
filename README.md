# SAR-ADC Behavioral Model

A Python behavioral model of an N-bit **Successive Approximation Register ADC (SAR-ADC)** with non-ideality injection and standard performance characterization.

Built as a portfolio project to demonstrate analog IC design knowledge. I have also done a similar analysis in Cadence Virtuoso post-layout simulation). But since the Candence project can't be open-source, I implemented in Python to demonstrate understanding.

---

## What it does

Models the complete SAR conversion algorithm and injects five real-world non-idealities:

| Non-ideality | Physical origin | Effect on performance |
|---|---|---|
| Comparator offset | Input-referred static offset Vos | Shifts transfer curve, code offset error |
| kT/C thermal noise | Noise frozen on CDAC sampling capacitor when switch opens | SNR floor set by √(kT/C) |
| Capacitor mismatch | Process variation in binary-weighted CDAC capacitors | DNL/INL errors, ENOB degradation |
| Clock jitter | Aperture uncertainty on the sampling instant | Noise floor that grows with input frequency |
| Reference noise | Thermal noise on the bandgap reference voltage | Random scaling error each conversion cycle |

Then computes standard ADC metrics from the output codes:
- **FFT spectrum** (dBFS)
- **SNDR** : Signal-to-Noise and Distortion Ratio
- **SFDR** : Spurious-Free Dynamic Range
- **ENOB** : Effective Number of Bits
- **DNL / INL** : Differential and Integral Non-Linearity (histogram method)

Also models **overdrive detection** when the input exceeds the ADC range the algorithm saturates naturally at code 0 or 255 without artificial clipping, matching real hardware behaviour.

---

## Project structure

```
sar_adc/
├── adc_core.py         Core SAR-ADC behavioral model (ideal)
├── nonidealities.py    Five non-ideality models
├── characterize.py     FFT, SNDR, SFDR, ENOB, DNL, INL computation
├── demo.py             Run all models, print summary, generate plots
├── test.py             Unit tests — run before committing
├── monte_carlo.py      Monte Carlo yield analysis across process corners
├── frequency_sweep.py  ENOB vs input frequency sweep with jitter limit
└── README.md
```

---

## Quickstart

```bash
# Clone
git clone https://github.com/SugandhMittal/sar-adc-model.git
cd sar-adc-model

# Install dependencies
pip install numpy matplotlib

# Run tests first
python test.py

# Run the demo
python demo.py
```

---

## Example output

Running `demo.py` compares the ideal ADC against each non-ideality and prints:

```
[Ideal]
       SAR-ADC PERFORMANCE SUMMARY
  SNDR          : +48.89 dBFS
  ENOB          :  7.83 bits
  SFDR          :  66.66 dBc
  DNL (peak)    :  0.496 LSB
  INL (peak)    :  0.498 LSB


[Cap. Mismatch (0.5%)]
  SNDR          : +48.22 dBFS
  ENOB          :  7.72 bits
  SFDR          :  59.69 dBc
  DNL (peak)    :  1.000 LSB
  INL (peak)    :  1.099 LSB

```

And generates a characterization plot showing FFT spectrum, DNL, and INL for all six models side by side.

---

## Key concepts

**Why SAR?**
The SAR architecture is the dominant ADC topology for medium-speed, medium-resolution applications (8–16 bit, 1 kS/s - 100 MS/s). It is used in microcontrollers, sensor interfaces, and data acquisition systems. Understanding its non-idealities is fundamental to any analog IC design role.

**Why behavioral modeling?**
Transistor-level simulation in Cadence Virtuoso is accurate but slow and tool-locked. Behavioral models run in milliseconds, can be swept over thousands of Monte Carlo samples, and let you explore architecture trade-offs before committing to silicon. This is how ADC architects think about designs before writing a single schematic.

**Coherent sampling:**
The test signal uses a coherent frequency, an exact integer number of cycles fits within the FFT window. This eliminates spectral leakage, giving a clean single-bin fundamental tone and accurate SNDR measurement. The same requirement applies on real ADC test benches.

**kT/C noise floor:**
For a 1 pF sampling capacitor at 300 K:
`v_noise_rms = √(kT/C) = √(1.38e-23 × 300 / 1e-12) ≈ 64 μV RMS`
For a 1 V reference and 8-bit ADC, 1 LSB = 3.9 mV, so kT/C noise is well below 1 LSB at 1 pF. At 100 fF it becomes significant this is the fundamental trade-off between capacitor size, power, and noise in scaled CMOS.

**Dynamic jitter model:**
Clock jitter noise is computed from the instantaneous signal slope using `np.gradient` rather than a worst-case constant. This means the jitter error is correctly small near signal peaks and maximum at zero crossings, matching the physics of aperture error.

---

## Monte Carlo Analysis

Running `monte_carlo.py` simulates N chips across multiple process corners
by varying capacitor mismatch sigma. Each seed represents a unique chip
with its own random mismatch pattern.

### What it computes
For each sigma value across 500 simulated chips:
- ENOB and SNDR distribution (mean, std, min, max)
- DNL and INL peak distribution
- Yield: percentage of chips meeting DNL < 1 LSB (monotonicity criterion)

### Example results
```
   Sigma   ENOB mean   ENOB std   DNL peak    Yield
   0.20%       7.787      0.048      0.891    46.0%
   0.50%       7.598      0.194      0.894    40.2%
   0.70%       7.446      0.284      0.993    32.4%
   1.00%       7.216      0.392      1.221    19.2%
   1.50%       6.868      0.513      1.707     8.8%
   2.00%       6.572      0.588      2.262     4.4%
```

### Key Finding
Yield drops sharply from 0.5% to 1.0% mismatch. 

### Limitation of Monte Carlo
Results below σ = 0.5% are statistically unreliable at N = 500
iterations due to finite histogram sampling artifacts, the yield variance
across runs exceeds 4% in this region. The yield sweep plot marks this
zone explicitly. Above σ = 0.5% results converge to within 1-2%,
confirming physical accuracy.

## ENOB vs Frequency Sweep

Running `frequency_sweep.py` sweeps the input frequency from low to 0.2 × fs
and plots how ENOB degrades due to clock jitter.

### What it shows
- **Ideal ADC** : ENOB is flat across all frequencies (quantization noise only)
- **Jitter ADC** : ENOB flat at low frequencies, degrading at higher frequencies
- **Theoretical limit** : combined effect of quantization + jitter noise, should closely match simulation

### Jitter crossover frequency
For 1 ps jitter at 1 GHz sampling rate the jitter limit crosses the
quantization limit at:

```
f_crossover = 1 / (2π × 2^N × t_jitter)
            = 1 / (2π × 256 × 1e-12)
            ≈ 621 MHz
```
For 1 GHz input frequency, this is not a limitation since it is above Nyquist frequency. Therefore, we simulate 
10 ps jitter to properly visualise.

### Limitation in Jitter Time Domain Modelling
The time-domain jitter model is valid below 0.2 × fs (200 MHz). Above this,
jitter noise sidebands alias back near the fundamental in the behavioral model,
causing SNDR to be artificially overestimated. A frequency-domain jitter model
would be needed for accurate results near Nyquist which is more complex and limited to standard signals


---

## Roadmap (contributions welcome)

- [x] Monte Carlo sweep: ENOB distribution over N mismatch samples
- [x] ENOB vs. input frequency sweep (jitter limit plot)
- [ ] Delta-Sigma ADC model for comparison
- [ ] Jupyter notebook version with interactive widgets
- [ ] Export metrics to CSV for batch analysis

---

## Acknowledgements

Core behavioral model, non-ideality physics, characterization algorithms, 
Monte Carlo analysis and ENOB vs Frequency Sweep are written independently. 
Visualization scripts (`demo.py`, `monte_carlo.py`) and test suite (`test.py`) 
generated with AI assistance (Claude, Anthropic).

---

## Author

**Sugandh Mittal**
Erasmus Mundus Joint Master's in Microelectronics (RADMEP)
([linkedin.com/sugandhmittal](https://www.linkedin.com/in/sugandh-m-a75b97215/)) · ([github.com/SugandhMittal](https://github.com/SugandhMittal))