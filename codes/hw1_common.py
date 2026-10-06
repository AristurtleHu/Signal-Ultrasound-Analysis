"""Shared helpers for the EE180 HW1 analysis scripts.

Path resolution, the dB convention M_dB = 20*log10(|X|/ref + floor) with a
shared reference and a 1e-6 floor, signal measurements shared by Q1/Q2, and
figure saving. All paths are relative to the package root.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt, stft

# Package root = parent of the codes/ directory that contains this file.
ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "data" / "q1_signals.npz").is_file():
    raise FileNotFoundError("Could not locate the package root from hw1_common.py")

IMAGE_DIR = ROOT / "image"
IMAGE_DIR.mkdir(exist_ok=True)

# dB floor (see module docstring).
DB_REF_FLOOR = 1e-6


# Plotting / loading
def to_db(magnitude, reference, floor=DB_REF_FLOOR):
    """Convert a magnitude array to dB with a shared reference and positive floor."""
    magnitude = np.asarray(magnitude, dtype=float)
    return 20.0 * np.log10(magnitude / float(reference) + floor)


def save_fig(fig, filename, dpi=150):
    """Save a figure into image/ and close it."""
    path = IMAGE_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path.relative_to(ROOT)}")


def load_q1():
    """Load the Q1/Q2 signals (60 MHz, 4800 samples)."""
    with np.load(ROOT / "data" / "q1_signals.npz") as data:
        return (
            data["signal_A"].copy(),
            data["signal_B"].copy(),
            data["time_s"].copy(),
            float(data["sampling_rate_hz"]),
        )


def load_q3_set(folder, table_name):
    """Load one Q3 set; returns (metadata rows, signals of shape (n, 896))."""
    base = ROOT / "data" / "q3" / folder
    with (base / table_name).open(encoding="utf-8-sig", newline="") as f:
        metadata = list(csv.DictReader(f))
    signals = np.stack([
        np.loadtxt(base / row["signal_file"], delimiter=",", skiprows=1)[:, 1]
        for row in metadata
    ])
    if signals.shape != (len(metadata), 896):
        raise ValueError("Each Q3 recording must contain 896 samples.")
    return metadata, signals


# Q1/Q2 measurements (shared so both scripts quote the same numbers)
def single_sided_spectrum(x, fs, n_fft=None):
    """Single-sided amplitude spectrum, normalized by the record length.

    Positive interior bins are doubled (DC and Nyquist are not).
    Returns (frequencies_Hz, amplitudes).
    """
    n_fft = x.size if n_fft is None else n_fft
    spectrum = np.fft.rfft(x, n=n_fft)
    amplitude = 2.0 * np.abs(spectrum) / x.size
    amplitude[0] /= 2.0
    if x.size % 2 == 0:
        amplitude[-1] /= 2.0
    return np.fft.rfftfreq(n_fft, 1.0 / fs), amplitude


def tone_frequencies(x, fs, f_lo=3.5e6, f_hi=4.6e6, n_fft=2 ** 18, n_top=2):
    """Strongest local spectral peaks in a band, from the full record (Hann, zero-padded)."""
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size), n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    band = (freqs > f_lo) & (freqs < f_hi)
    fb, sb = freqs[band], spectrum[band]
    peaks = [(fb[i], sb[i]) for i in range(1, sb.size - 1)
             if sb[i] > sb[i - 1] and sb[i] > sb[i + 1] and sb[i] > 0.3 * sb.max()]
    peaks = sorted(peaks, key=lambda p: -p[1])[:n_top]
    return np.array(sorted(f for f, _ in peaks))


def chirp_ridge(x, fs, nperseg, hop=8, f_lo=4.6e6, f_hi=8.0e6, nfft=8192,
                threshold_db=-20.0):
    """Track the chirp ridge and fit f(t) = slope*t + intercept.

    Sub-bin frequencies come from a parabolic fit around each dB peak; frames
    below `threshold_db` relative to the STFT maximum are skipped.
    Returns (times_us, freqs_MHz, slope_MHz_per_us, intercept_MHz).
    """
    freq, t, z = stft(x, fs=fs, window="hann", nperseg=nperseg,
                      noverlap=nperseg - hop, nfft=nfft, detrend=False,
                      boundary=None, padded=False, scaling="spectrum")
    band = (freq >= f_lo) & (freq <= f_hi)
    magnitude = np.abs(z)
    mag_db = 20.0 * np.log10(magnitude / magnitude.max() + 1e-6)
    fb = freq[band]
    df = freq[1] - freq[0]
    times, freqs = [], []
    for j in range(t.size):
        column = mag_db[band, j]
        k = int(np.argmax(column))
        if column[k] <= threshold_db:
            continue
        delta = 0.0
        if 0 < k < column.size - 1:
            y0, y1, y2 = column[k - 1], column[k], column[k + 1]
            denom = y0 - 2 * y1 + y2
            if denom != 0:
                delta = 0.5 * (y0 - y2) / denom
        times.append(t[j] * 1e6)
        freqs.append((fb[k] + delta * df) / 1e6)
    times, freqs = np.array(times), np.array(freqs)
    slope, intercept = np.polyfit(times, freqs, 1)
    return times, freqs, slope, intercept


def burst_lobes(x, fs, f_lo=8.5e6, f_hi=9.5e6, height_frac=0.3, min_sep_us=1.0):
    """Band-passed 9 MHz burst: lobe times/amplitudes and center frequency.

    Returns (lobe_times_us, lobe_amplitudes, center_frequency_Hz, envelope).
    """
    sos = butter(4, [f_lo, f_hi], btype="band", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, x)
    envelope = np.abs(hilbert(filtered))
    peaks, _ = find_peaks(envelope, height=height_frac * envelope.max(),
                          distance=max(1, int(min_sep_us * 1e-6 * fs)))
    lobe_times = peaks / fs * 1e6
    lobe_amplitudes = envelope[peaks]
    spectrum = np.abs(np.fft.rfft(filtered, 2 ** 16))
    f_axis = np.fft.rfftfreq(2 ** 16, 1.0 / fs)
    band = (f_axis > f_lo) & (f_axis < f_hi)
    center_frequency = f_axis[band][np.argmax(spectrum[band])]
    return lobe_times, lobe_amplitudes, center_frequency, envelope
