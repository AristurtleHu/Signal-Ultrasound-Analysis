"""Optional HW1 features: each public function returns one float.

No feature bundles, classification, parameter fitting or amplitude normalization.
See ../FEATURE_GUIDE.md. Dependencies: numpy and scipy.
Underscore-prefixed functions are internal calculation helpers.
"""
import numpy as np
from scipy.signal import hilbert, correlate, windows, find_peaks
from scipy.ndimage import uniform_filter1d

def _signal(x):
    try:
        a = np.asarray(x)
    except (TypeError, ValueError) as exc:
        raise ValueError('x must be a real 1-D numeric signal') from exc
    if np.iscomplexobj(a) or a.ndim != 1 or a.size < 4:
        raise ValueError('x must be a real 1-D signal with at least 4 samples')
    try:
        a = a.astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('x must contain real numeric samples') from exc
    if not np.all(np.isfinite(a)):
        raise ValueError('x contains NaN or infinity')
    return a


def _fs(fs):
    if isinstance(fs, (bool, np.bool_)) or not isinstance(fs, (int, float, np.integer, np.floating)):
        raise ValueError('fs must be a positive finite real sampling frequency in Hz')
    try:
        value = float(fs)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('fs must be a positive finite real sampling frequency in Hz') from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError('fs must be a positive finite sampling frequency in Hz')
    return value


def _entropy(weights):
    total = np.sum(weights)
    if total == 0:
        return float('nan')
    p = weights / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(len(weights)))


def _amplitude_envelope(x, pad_samples=None):
    """Absolute analytic signal; reflect extension then crop to original length.

    Default padding is N-1 samples on each side. No detrending or taper is applied.
    Set pad_samples=0 to disable extension. Returns an N-element array.
    """
    x = _signal(x)
    if pad_samples is None:
        pad_samples = len(x) - 1
    if isinstance(pad_samples, bool) or not isinstance(pad_samples, (int, np.integer)) or pad_samples < 0:
        raise ValueError('pad_samples must be a nonnegative integer')
    extended = np.pad(x, (pad_samples, pad_samples), mode='reflect') if pad_samples else x
    a = np.abs(hilbert(extended))
    return a[pad_samples:pad_samples + len(x)].copy()


def _windowed_power_spectrum(x, fs):
    """Periodic Hann window, unpadded real FFT, one-sided squared magnitude.

    DC is retained. This is a relative power weight spectrum, not calibrated PSD.
    Returns (frequencies_Hz, weights). No mean subtraction or amplitude normalization.
    """
    x, fs = _signal(x), _fs(fs)
    y = np.fft.rfft(x * windows.hann(len(x), sym=False))
    power = np.abs(y) ** 2
    if len(x) % 2 == 0:
        power[1:-1] *= 2
    else:
        power[1:] *= 2
    return np.fft.rfftfreq(len(x), 1 / fs), power


def peak_abs(x):
    """Maximum absolute amplitude."""
    x = _signal(x)
    return float(np.max(np.abs(x)))


def peak_to_peak(x):
    """Maximum minus minimum amplitude."""
    x = _signal(x)
    return float(np.ptp(x))


def energy(x):
    """Sum of squared samples; not calibrated acoustic energy."""
    x = _signal(x)
    return float(np.sum(x ** 2))


def skewness(x):
    """Uncorrected skewness; NaN for constant input."""
    x = _signal(x)
    if np.all(x == x[0]):
        return float('nan')
    d = x - np.mean(x)
    v = np.mean(d ** 2)
    return float(np.mean(d ** 3) / v ** 1.5) if v > 0 else float("nan")


def kurtosis(x):
    """Pearson kurtosis (Gaussian population: 3); NaN for constant input."""
    x = _signal(x)
    if np.all(x == x[0]):
        return float('nan')
    d = x - np.mean(x)
    v = np.mean(d ** 2)
    return float(np.mean(d ** 4) / v ** 2) if v > 0 else float("nan")


def crest_factor(x):
    """Absolute peak divided by RMS; NaN for all-zero input."""
    x = _signal(x)
    r = np.sqrt(np.mean(x ** 2))
    return float(np.max(np.abs(x)) / r) if r > 0 else float("nan")


def envelope_mean(x, pad_samples=None):
    """Mean Hilbert-envelope amplitude. Reflect padding defaults to N-1 per side."""
    a = _amplitude_envelope(x, pad_samples)
    return float(np.mean(a))


def envelope_cv(x, pad_samples=None):
    """Population envelope standard deviation divided by envelope mean. Reflect padding defaults to N-1 per side."""
    a = _amplitude_envelope(x, pad_samples)
    return float(np.std(a) / np.mean(a)) if np.mean(a) > 0 else float("nan")


def envelope_peak_to_mean(x, pad_samples=None):
    """Envelope peak divided by envelope mean. Reflect padding defaults to N-1 per side."""
    a = _amplitude_envelope(x, pad_samples)
    return float(np.max(a) / np.mean(a)) if np.mean(a) > 0 else float("nan")


def envelope_energy_entropy(x, pad_samples=None):
    """Normalized entropy of squared-envelope weights. Reflect padding defaults to N-1 per side."""
    a = _amplitude_envelope(x, pad_samples)
    return float(_entropy(a ** 2))


def envelope_peak_spacing_us(x, fs):
    """Spacing (microseconds) of the two highest five-point-smoothed envelope peaks.

    Hilbert transform on the full input without padding, taper or mean subtraction.
    Smooth its magnitude using a centered 5-sample arithmetic mean with reflect
    boundaries. Find all local maxima, with no height, prominence or separation
    filter, and keep the highest two. Earlier sample wins equal-height ties.
    Plateau peaks use scipy.signal.find_peaks midpoint convention; endpoints
    are not local maxima. Return NaN if fewer than two peaks exist.
    """
    x, fs = _signal(x), _fs(fs)
    envelope = uniform_filter1d(np.abs(hilbert(x)), size=5, mode='reflect')
    peaks, _ = find_peaks(envelope)
    if len(peaks) < 2:
        return float('nan')
    first, second = sorted(peaks, key=lambda i: (-envelope[i], i))[:2]
    return float(abs(int(second) - int(first)) / fs * 1e6)


def spectral_centroid_hz(x, fs):
    """Hann-windowed spectral centroid in Hz. NaN for zero total spectral weight."""
    f, p = _windowed_power_spectrum(x, fs)
    if np.sum(p) == 0:
        return float("nan")
    w = p / np.sum(p)
    return float(np.sum(f * w))


def spectral_spread_hz(x, fs):
    """Hann-windowed spectral spread in Hz. NaN for zero total spectral weight."""
    f, p = _windowed_power_spectrum(x, fs)
    if np.sum(p) == 0:
        return float("nan")
    w = p / np.sum(p)
    return float(np.sqrt(np.sum(w * (f - np.sum(f * w)) ** 2)))


def spectral_peak_hz(x, fs):
    """Frequency of largest Hann spectral weight; lowest frequency wins ties. NaN for zero total spectral weight."""
    f, p = _windowed_power_spectrum(x, fs)
    if np.sum(p) == 0:
        return float("nan")
    w = p / np.sum(p)
    return float(f[np.argmax(p)])


def spectral_entropy(x, fs):
    """Normalized entropy of Hann-windowed spectral weights. NaN for zero total spectral weight."""
    f, p = _windowed_power_spectrum(x, fs)
    if np.sum(p) == 0:
        return float("nan")
    w = p / np.sum(p)
    return float(_entropy(p))


def spectral_flatness(x, fs):
    """Geometric-to-arithmetic mean of Hann-windowed spectral weights. NaN for zero total spectral weight."""
    f, p = _windowed_power_spectrum(x, fs)
    if np.sum(p) == 0:
        return float("nan")
    w = p / np.sum(p)
    return float(0.0 if np.any(p == 0) else np.exp(np.mean(np.log(p))) / np.mean(p))


def _autocorrelation(x, max_lag_samples):
    """Internal mean-centered linear correlation at lags 1 through L. No overlap correction."""
    x = _signal(x)
    if isinstance(max_lag_samples, bool) or not isinstance(max_lag_samples, (int, np.integer)) or not 1 <= max_lag_samples < len(x):
        raise ValueError('max_lag_samples must be an integer in [1, N-1]')
    if np.all(x == x[0]):
        return np.full(max_lag_samples, np.nan)
    y = x - np.mean(x)
    e = np.dot(y, y)
    if e == 0:
        return np.full(max_lag_samples, np.nan)
    return correlate(y, y, mode='full', method='direct')[len(y):len(y) + max_lag_samples] / e


def autocorr_mean_square(x, max_lag_samples):
    """Mean squared normalized correlation over lags 1..L; NaN for constant input."""
    r = _autocorrelation(x, max_lag_samples)
    return float(np.mean(r ** 2))


def autocorr_max(x, max_lag_samples):
    """Largest signed correlation over lags 1..L; not necessarily a periodicity peak."""
    r = _autocorrelation(x, max_lag_samples)
    return float(np.max(r))


def component_energy_fraction(components, component_index):
    """Energy fraction of ONE component (zero-based index). Input shape (components, samples)."""
    a = np.asarray(components)
    if a.ndim != 2 or len(a) < 1:
        raise ValueError('components must have shape (n_components, n_samples)')
    if isinstance(component_index, bool) or not isinstance(component_index, (int, np.integer)) or not 0 <= component_index < len(a):
        raise ValueError('component_index must be a valid zero-based component index')
    a = np.stack([_signal(row) for row in a])
    e = np.sum(a ** 2, axis=1)
    return float(e[component_index] / e.sum()) if e.sum() > 0 else float('nan')


__all__ = ['envelope_peak_spacing_us', 'peak_abs', 'peak_to_peak', 'energy', 'skewness', 'kurtosis', 'crest_factor', 'envelope_mean', 'envelope_cv', 'envelope_peak_to_mean', 'envelope_energy_entropy', 'spectral_centroid_hz', 'spectral_spread_hz', 'spectral_peak_hz', 'spectral_entropy', 'spectral_flatness', 'autocorr_mean_square', 'autocorr_max', 'component_energy_fraction']
