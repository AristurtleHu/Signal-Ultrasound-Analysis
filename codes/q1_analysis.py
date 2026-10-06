"""Q1: time-domain, FFT and STFT analysis of signals A and B.

Inputs : data/q1_signals.npz (60 MHz, 4800 samples)
Outputs: image/q1a_time_domain.png, image/q1a_fft_spectra.png, image/q1b_stft.png
Run    : conda run -n ai python codes/q1_analysis.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend so figures are written to image/
import matplotlib.pyplot as plt
from scipy.signal import hilbert, stft

from hw1_common import (burst_lobes, chirp_ridge, load_q1, save_fig,
                        single_sided_spectrum, to_db, tone_frequencies)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
signal_a, signal_b, time_s, fs = load_q1()
time_us = time_s * 1e6

# ---------------------------------------------------------------------------
# Q1(a) - processing: time-domain envelopes and FFT spectra
# ---------------------------------------------------------------------------
envelope_a = np.abs(hilbert(signal_a))
envelope_b = np.abs(hilbert(signal_b))

freq_a, amp_a = single_sided_spectrum(signal_a, fs)
freq_b, amp_b = single_sided_spectrum(signal_b, fs)

# Shared dB reference: largest single-sided amplitude across both signals.
fft_ref = max(amp_a.max(), amp_b.max())
fft_db_a = to_db(amp_a, fft_ref)
fft_db_b = to_db(amp_b, fft_ref)

# ---------------------------------------------------------------------------
# Q1(a) - figures
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
for ax, name, x, env in [(axes[0], "A", signal_a, envelope_a),
                         (axes[1], "B", signal_b, envelope_b)]:
    ax.plot(time_us, x, lw=0.5, color="tab:blue", alpha=0.6, label="signal")
    ax.plot(time_us, env, lw=1.2, color="tab:red", label="Hilbert envelope")
    ax.set_ylabel("amplitude")
    ax.set_title(f"Signal {name}: time domain")
    ax.set_ylim(-2.2, 2.2)
    ax.legend(loc="upper right", fontsize=8)
axes[1].set_xlabel(r"time ($\mu$s)")
axes[1].set_xlim(0, 80)
save_fig(fig, "q1a_time_domain.png")

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
for ax, name, db in [(axes[0], "A", fft_db_a), (axes[1], "B", fft_db_b)]:
    ax.plot(freq_a / 1e6, db, lw=0.7, color="tab:blue")
    ax.set_ylabel("magnitude (dB)")
    ax.set_title(f"Signal {name}: single-sided FFT magnitude")
    ax.set_ylim(-100, 5)
axes[1].set_xlabel("frequency (MHz)")
axes[1].set_xlim(0, 20)
save_fig(fig, "q1a_fft_spectra.png")

# ---------------------------------------------------------------------------
# Q1(b) - processing: STFT with the required parameters
#   Hann window, 240 samples, hop 6 samples, N_FFT = 4096, no boundary extension.
# ---------------------------------------------------------------------------
NPERSEG = 240
HOP = 6
NFFT = 4096

freq_stft, time_stft, z_a = stft(
    signal_a, fs=fs, window="hann", nperseg=NPERSEG, noverlap=NPERSEG - HOP,
    nfft=NFFT, detrend=False, boundary=None, padded=False, scaling="spectrum")
_, _, z_b = stft(
    signal_b, fs=fs, window="hann", nperseg=NPERSEG, noverlap=NPERSEG - HOP,
    nfft=NFFT, detrend=False, boundary=None, padded=False, scaling="spectrum")

# Common normalization reference for both signals; fixed -80..0 dB colour scale.
stft_ref = max(np.abs(z_a).max(), np.abs(z_b).max())
stft_db_a = to_db(np.abs(z_a), stft_ref)
stft_db_b = to_db(np.abs(z_b), stft_ref)
VMIN, VMAX = -80.0, 0.0

# ---------------------------------------------------------------------------
# Q1(b) - measurements shared with Q2
# ---------------------------------------------------------------------------
measurements = {}
for name, x, env in [("A", signal_a, envelope_a), ("B", signal_b, envelope_b)]:
    _, _, slope, intercept = chirp_ridge(x, fs, nperseg=512)
    lobe_times, _, lobe_freq, _ = burst_lobes(x, fs)
    measurements[name] = {
        "env_peak_us": time_us[np.argmax(env)],
        "slope_mhz_per_us": slope,
        "intercept_mhz": intercept,
        "lobe_times_us": lobe_times,
        "lobe_sep_us": np.diff(lobe_times).mean() if lobe_times.size > 1 else np.nan,
        "lobe_freq_mhz": lobe_freq / 1e6,
        "tones_mhz": tone_frequencies(x, fs) / 1e6,
    }

for name in ("A", "B"):
    m = measurements[name]
    print(f"--- Signal {name} ---")
    print(f"  envelope peak        : {m['env_peak_us']:.2f} us")
    print(f"  fixed tones          : {m['tones_mhz'][0]:.3f} + {m['tones_mhz'][1]:.3f} MHz "
          f"(separation {abs(m['tones_mhz'][1] - m['tones_mhz'][0]) * 1e3:.0f} kHz)")
    print(f"  chirp fit            : f = {m['intercept_mhz']:.3f} + "
          f"{m['slope_mhz_per_us']:+.5f} t /us  ({m['slope_mhz_per_us'] * 1e3:+.1f} GHz/s)")
    print(f"  9 MHz pulse lobes    : " +
          ", ".join(f"{tt:.2f}" for tt in m["lobe_times_us"]) + " us "
          f"(separation {m['lobe_sep_us']:.2f} us, f0 = {m['lobe_freq_mhz']:.3f} MHz)")

# ---------------------------------------------------------------------------
# Q1(b) - figure with common scale and feature annotations
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, sharey=True,
                         constrained_layout=True)
for ax, name, db in [(axes[0], "A", stft_db_a), (axes[1], "B", stft_db_b)]:
    mesh = ax.pcolormesh(time_stft * 1e6, freq_stft / 1e6, db,
                         shading="auto", cmap="viridis", vmin=VMIN, vmax=VMAX)
    ax.set_ylabel("frequency (MHz)")
    ax.set_title(f"Signal {name}: STFT magnitude (Hann 240, hop 6, NFFT 4096)")
    ax.set_ylim(0, 12)
    m = measurements[name]
    # Arrow along the fitted chirp ridge.
    ax.annotate("", xy=(70, m["intercept_mhz"] + m["slope_mhz_per_us"] * 70),
                xytext=(15, m["intercept_mhz"] + m["slope_mhz_per_us"] * 15),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.6))
    direction = "up" if m["slope_mhz_per_us"] > 0 else "down"
    ax.text(30, m["intercept_mhz"] + m["slope_mhz_per_us"] * 30 + 0.5,
            f"{direction}-chirp {m['slope_mhz_per_us'] * 1e3:+.0f} GHz/s",
            color="white", fontsize=9)
    # The 9 MHz pulse pair.
    ax.annotate(f"9 MHz pulse pair\n{m['lobe_times_us'][0]:.0f} / {m['lobe_times_us'][1]:.0f} us",
                xy=(m["lobe_times_us"][1], 9.0),
                xytext=(m["lobe_times_us"][1] - 22, 10.6),
                color="white", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="white", lw=1.0))
    ax.text(66, 3.6, "fixed tones\n3.8 + 4.2 MHz", color="white", fontsize=8)
axes[1].set_xlabel(r"time ($\mu$s)")
axes[1].set_xlim(0, 80)
fig.colorbar(mesh, ax=axes, label="magnitude (dB, common reference)", shrink=0.9)
save_fig(fig, "q1b_stft.png")
