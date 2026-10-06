"""Q2: STFT window-length study and continuous wavelet analysis of Signal A.

Inputs : data/q1_signals.npz (only signal_A is used)
Outputs: image/q2a_stft_windows.png, image/q2a_zoom_slices.png, image/q2c_cwt.png
Usage  : python codes/q2_analysis.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend so figures are written to image/
import matplotlib.pyplot as plt
import pywt
from scipy.signal import stft

from hw1_common import (burst_lobes, chirp_ridge, load_q1, save_fig, to_db,
                        tone_frequencies)

signal_a, _, time_s, fs = load_q1()
time_us = time_s * 1e6

HOP = 8            # 0.133 us between frames, identical for both windows
NFFT = 4096
SHORT_WIN = 128    # 2.13 us  -> Hann main lobe ~ 1.88 MHz
LONG_WIN = 1024    # 17.07 us -> Hann main lobe ~ 234 kHz
VMIN, VMAX = -80.0, 0.0


def spectrogram_db(x, nperseg, reference=None):
    """Hann-window STFT in dB with an optional shared reference.

    boundary=None: no signal extension, only complete windows inside the
    record; window centers span nperseg/2 to N - nperseg/2.
    """
    freq, t, z = stft(x, fs=fs, window="hann", nperseg=nperseg,
                      noverlap=nperseg - HOP, nfft=NFFT, detrend=False,
                      boundary=None, padded=False, scaling="spectrum")
    magnitude = np.abs(z)
    if reference is None:
        reference = magnitude.max()
    return freq, t, to_db(magnitude, reference), reference


# Q2(a): short vs long window with a common reference and colour scale
freq_s, t_s, _, short_ref = spectrogram_db(signal_a, SHORT_WIN)
freq_l, t_l, _, long_ref = spectrogram_db(signal_a, LONG_WIN)
stft_ref = max(short_ref, long_ref)  # common reference for both windows
_, _, db_short, _ = spectrogram_db(signal_a, SHORT_WIN, reference=stft_ref)
_, _, db_long, _ = spectrogram_db(signal_a, LONG_WIN, reference=stft_ref)

fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, sharey=True,
                         constrained_layout=True)
for ax, label, fq, tt, db in [
        (axes[0], "short window (128 samples = 2.13 us)", freq_s, t_s, db_short),
        (axes[1], "long window (1024 samples = 17.07 us)", freq_l, t_l, db_long)]:
    mesh = ax.pcolormesh(tt * 1e6, fq / 1e6, db, shading="auto",
                         cmap="viridis", vmin=VMIN, vmax=VMAX)
    ax.set_title(f"Signal A STFT, {label}")
    ax.set_ylabel("frequency (MHz)")
    ax.set_ylim(0, 12)
axes[1].set_xlabel(r"time ($\mu$s)")
axes[1].set_xlim(0, 80)
fig.colorbar(mesh, ax=axes, label="magnitude (dB, common reference)", shrink=0.9)
save_fig(fig, "q2a_stft_windows.png")

# Q2(a): component measurements
tones = tone_frequencies(signal_a, fs) / 1e6
tone_sep_mhz = tones[1] - tones[0]

_, freqs_s, slope_s, intercept_s = chirp_ridge(signal_a, fs, nperseg=SHORT_WIN)
_, _, slope_m, intercept_m = chirp_ridge(signal_a, fs, nperseg=512)
times_l, freqs_l, slope_l, intercept_l = chirp_ridge(signal_a, fs, nperseg=LONG_WIN)
lobe_times, _, lobe_freq, burst_env = burst_lobes(signal_a, fs)
lobe_sep_us = np.diff(lobe_times).mean()


def lobe_width_us(t_center_us, frac=0.5):
    """-6 dB (frac = 0.5 in amplitude) width of one envelope lobe."""
    peak = int(round(t_center_us * 1e-6 * fs))
    threshold = frac * burst_env[peak]
    i = peak
    while i > 0 and burst_env[i] > threshold:
        i -= 1
    j = peak
    while j < burst_env.size - 1 and burst_env[j] > threshold:
        j += 1
    return (j - i) / fs * 1e6


print("--- Q2(a) component measurements (Signal A) ---")
print(f"fixed tones              : {tones[0]:.3f} + {tones[1]:.3f} MHz "
      f"(separation {tone_sep_mhz * 1e3:.0f} kHz)")
print(f"chirp ridge, long window : f = {intercept_l:.3f} + {slope_l:+.5f} t /us "
      f"({slope_l * 1e3:+.1f} GHz/s), tracked {times_l.min():.2f}-{times_l.max():.2f} us")
print(f"chirp ridge, 512 window  : f = {intercept_m:.3f} + {slope_m:+.5f} t /us "
      f"({slope_m * 1e3:+.1f} GHz/s); quoted slope ~ +30 GHz/s")
print(f"chirp ridge, short window: f = {intercept_s:.3f} + {slope_s:+.5f} t /us, "
      f"tracked {freqs_s.min():.3f}-{freqs_s.max():.3f} MHz")
for t_q in (5.0, 20.0, 45.0, 75.0):
    f_chirp = intercept_l + slope_l * t_q
    print(f"  t={t_q:5.1f} us: chirp {f_chirp:.3f} MHz, "
          f"separation from upper tone {f_chirp - tones[1]:.3f} MHz")
print(f"9 MHz pulse pair         : {lobe_times[0]:.2f} and {lobe_times[1]:.2f} us "
      f"(separated by {lobe_sep_us:.2f} us), f0 = {lobe_freq / 1e6:.3f} MHz")
print(f"9 MHz lobe widths (-6 dB): {lobe_width_us(lobe_times[0]):.2f} and "
      f"{lobe_width_us(lobe_times[1]):.2f} us")

# Q2(a) supporting figure: zoomed spectrograms and slices
band_lo, band_hi = 3.0, 8.0
slice_times = [20.0, 45.0]

fig = plt.figure(figsize=(12, 9), constrained_layout=True)
grid = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.0, 1.0])
for col, (fq, tt, db, label) in enumerate([
        (freq_s, t_s, db_short, "short window (128)"),
        (freq_l, t_l, db_long, "long window (1024)")]):
    ax = fig.add_subplot(grid[0, col])
    zoom = (fq >= band_lo * 1e6) & (fq <= band_hi * 1e6)
    mesh = ax.pcolormesh(tt * 1e6, fq[zoom] / 1e6, db[zoom], shading="auto",
                         cmap="viridis", vmin=VMIN, vmax=VMAX)
    for t_sel in slice_times:
        ax.axvline(t_sel, color="white", ls="--", lw=0.8)
    ax.set_xlim(0, 80)
    ax.set_title(f"zoom 3-8 MHz, {label}")
    ax.set_xlabel(r"time ($\mu$s)")
    ax.set_ylabel("frequency (MHz)")
fig.colorbar(mesh, ax=fig.axes[:2], label="magnitude (dB)", shrink=0.9)

for row, t_sel in enumerate(slice_times, start=1):
    ax = fig.add_subplot(grid[row, :])
    for fq, tt, db, style, colour in [(freq_s, t_s, db_short, ":", "tab:orange"),
                                      (freq_l, t_l, db_long, "-", "tab:blue")]:
        j = int(np.argmin(np.abs(tt * 1e6 - t_sel)))
        zoom = (fq >= band_lo * 1e6) & (fq <= band_hi * 1e6)
        label = "short 128" if style == ":" else "long 1024"
        ax.plot(fq[zoom] / 1e6, db[zoom, j], style, color=colour, lw=1.4, label=label)
    ax.set_xlim(band_lo, band_hi)
    ax.set_ylim(-80, 2)
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("magnitude (dB)")
    ax.set_title(f"STFT magnitude slice at t = {t_sel:.0f} us")
    ax.legend(fontsize=8)
save_fig(fig, "q2a_zoom_slices.png")

# Q2(c): complex Morlet CWT. cmor1.5-1.0 has B = 1.5, C = 1.0 (cycles/sample
# at scale 1), scale-to-frequency f = C*fs/a. PyWavelets returns L2-normalized
# coefficients (|W| ~ sqrt(a) for a tone), so each row is divided by sqrt(a)
# (per-scale L1) to give equal brightness for equal-amplitude components.
WAVELET = "cmor1.5-1.0"
F_MIN, F_MAX, N_FREQS = 2.5e6, 12e6, 400
freqs_target = np.geomspace(F_MIN, F_MAX, N_FREQS)
scales = 1.0 * fs / freqs_target

coefs, freqs = pywt.cwt(signal_a, scales, WAVELET, sampling_period=1.0 / fs)
coefs = coefs / np.sqrt(scales)[:, None]          # per-scale L1 normalization
cwt_db = to_db(np.abs(coefs), np.abs(coefs).max())

fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
mesh = ax.pcolormesh(time_us, freqs / 1e6, cwt_db, shading="auto",
                     cmap="viridis", vmin=VMIN, vmax=VMAX)
ax.contour(time_us, freqs / 1e6, cwt_db, levels=[-30.0], colors="white",
           linewidths=0.6, alpha=0.7)
ax.annotate("9 MHz pulse pair\n(two lobes resolved)", xy=(lobe_times[0], 9.0),
            xytext=(36, 10.2), color="white", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="white", lw=1.0))
ax.annotate("up-chirp", xy=(55, intercept_l + slope_l * 55), xytext=(62, 6.6),
            color="white", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="white", lw=1.0))
ax.text(8, 4.55, "fixed tones (3.8 + 4.2 MHz)", color="white", fontsize=8)
ax.set_xlim(0, 80)
ax.set_ylim(F_MIN / 1e6, 10)
ax.set_xlabel(r"time ($\mu$s)")
ax.set_ylabel("frequency (MHz)")
ax.set_title(f"Signal A CWT magnitude, {WAVELET} (white line: -30 dB contour)")
fig.colorbar(mesh, ax=ax, label="magnitude (dB, common reference)")
save_fig(fig, "q2c_cwt.png")

# Resolution summary.
bw_short = 4.0 * fs / SHORT_WIN
bw_long = 4.0 * fs / LONG_WIN
q_morlet = np.pi * np.sqrt(2 * 1.5) / 2.0
print("--- Q2 window / wavelet resolution ---")
print(f"short window: dt = {SHORT_WIN / fs * 1e6:.2f} us, Hann main lobe = {bw_short / 1e6:.3f} MHz")
print(f"long window : dt = {LONG_WIN / fs * 1e6:.2f} us, Hann main lobe = {bw_long / 1e6:.3f} MHz")
print(f"cmor1.5-1.0 : Q ~ {q_morlet:.2f}, bandwidth 4 MHz -> {4e6 / q_morlet / 1e6:.2f} MHz, "
      f"9 MHz -> {9e6 / q_morlet / 1e6:.2f} MHz")
