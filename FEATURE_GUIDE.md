# HW1: Optional Signal Features

This reference lists candidate features, not a required feature set or a classification solution. Choose features you can explain and support your choices using the labeled reference recordings. You do not need to use every feature or a machine-learning classifier. Use feature statistics and plots to choose explicit numerical decision rules, as described in the assignment.

## 1. Conventions

- Each supplied recording is a normalized, real, one-dimensional signal with 896 samples. The sampling frequency is `fs = 26.5216e6` Hz; this is not the probe center frequency.
- The helpers do not rescale individual signals or apply additional amplitude normalization. Report any preprocessing you add and use the same settings across recordings.
- Below, `x` denotes one signal or one component produced by your chosen decomposition. When describing a component feature, compute it on that component and label the feature accordingly. Corresponding feature-vector entries must have the same meaning across samples.
- Let $x_n$, $n=0,\ldots,N-1$, denote its samples; $\mu=N^{-1}\sum_n x_n$, $m_r=N^{-1}\sum_n(x_n-\mu)^r$. Statistical moments use denominator $N$ (`ddof=0`), without bias correction. These conventions describe the observed waveform.
- Amplitude units are normalized signal units. Squared-amplitude features are not calibrated acoustic power or energy. Entropies use natural logarithms and normalization to approximately $[0,1]$; terms with zero weight contribute zero.
- Undefined features return `NaN` (for example, skewness of a constant signal). Inspect and explain them; do not silently replace them with zero.

## 2. Basic amplitude and distribution features

Implement mean, variance, standard deviation, median, mean absolute amplitude, and RMS yourself using the formulas below and NumPy. These six features are not included in the helper script. For the remaining features, the identifiers in the table are the names of the supplied functions; each call returns one float.

| Feature / identifier | Meaning | Calculation |
|---|---|---|
| Mean / `mean` | Average signed amplitude, including any offset | $\mu$ |
| Variance / `variance` | Spread around the mean | $m_2$ |
| Standard deviation / `std` | Spread in amplitude units | $\sqrt{m_2}$ |
| Median / `median` | Middle value of the sorted samples | For even $N$, average the two middle values |
| Mean absolute amplitude / `mean_abs` | Average amplitude magnitude | $N^{-1}\sum_n\lvert x_n\rvert$ |
| Root mean square / `rms` | Overall amplitude magnitude | $\sqrt{N^{-1}\sum_n x_n^2}$ |
| Absolute peak / `peak_abs` | Largest amplitude magnitude | $\max_n\lvert x_n\rvert$ |
| Peak-to-peak amplitude / `peak_to_peak` | Range of signed amplitudes | $\max_n x_n-\min_n x_n$ |
| Discrete energy / `energy` | Sum of squared samples | $E=\sum_n x_n^2$ |
| Skewness / `skewness` | Asymmetry of the sample-amplitude distribution | $m_3/m_2^{3/2}$ |
| Pearson kurtosis / `kurtosis` | Weight of large deviations relative to variance | $m_4/m_2^2$; Gaussian population value is 3 |
| Crest factor / `crest_factor` | Peak relative to overall RMS | $\max_n\lvert x_n\rvert/\mathrm{RMS}$ |

Skewness and kurtosis describe the distribution of amplitude values, not the shape of a plotted curve alone. They are undefined when variance is zero. RMS and energy are redundant for a fixed length: $E=N\,\mathrm{RMS}^2$. Also, $\mathrm{RMS}^2=m_2+\mu^2$. Pearson kurtosis differs from *excess kurtosis*, which subtracts 3. The helper uses Pearson kurtosis.

## 3. Component energy fractions

For components $c_j[n]$, define $E_j=\sum_n c_j[n]^2$ and $q_j=E_j/\sum_i E_i$.

`component_energy_fraction(components, component_index)` returns a single fraction for the selected component. The index starts at zero; for example, index 0 selects the first component. Pass an array with shape `(n_components, n_samples)`. Fractions compare how squared amplitude is distributed across components. Their denominator is the sum of component energies, not the original signal energy. For nonorthogonal components, energies need not add to the original energy; these fractions are not automatically physical energy fractions. The result is `NaN` if every component is zero.

## 4. Envelope features (helper functions provided)

The analytic signal is $z_n=x_n+i\mathcal{H}\{x\}_n$, where $\mathcal{H}$ is the Hilbert transform. Its amplitude envelope is $a_n=\lvert z_n\rvert$. For an oscillatory narrowband component, this helps describe amplitude modulation; for a broad mixture of oscillations, interpretation requires care.

The four envelope summary functions in the first table below accept `(x, pad_samples=None)` and return one scalar. By default, the internal calculation reflects `N-1` samples at each end, applies the Hilbert transform, then crops back to the original length. `pad_samples=0` disables extension. It does not subtract the mean or apply a taper. This is an analytic-signal operation, separate from the windowed spectral estimate below: tapering here would change the envelope being measured. Reflection reduces the abrupt periodic seam but does not remove all edge effects; inspect the envelope and state any changed setting.

Call each of the following functions separately:

| Feature / identifier | Meaning | Calculation |
|---|---|---|
| Envelope mean / `envelope_mean` | Average envelope amplitude | $\bar a=N^{-1}\sum_n a_n$ |
| Envelope coefficient of variation / `envelope_cv` | Envelope variation relative to its mean | $\sqrt{N^{-1}\sum_n(a_n-\bar a)^2}/\bar a$ |
| Envelope peak-to-mean / `envelope_peak_to_mean` | Peak relative to average envelope | $\max_n a_n/\bar a$ |
| Envelope energy entropy / `envelope_energy_entropy` | Distribution of squared envelope across samples | $p_n=a_n^2/\sum_i a_i^2$, then $-\sum_n p_n\ln p_n/\ln N$ |

Lower entropy indicates concentration in fewer samples; higher entropy indicates a more even distribution. This is not a direct count of echoes. Ratios and entropy are undefined for an all-zero envelope.

### Spacing of the two highest envelope peaks

| Feature / identifier | Meaning | Calculation |
|---|---|---|
| Envelope peak spacing / `envelope_peak_spacing_us(x, fs)` | Time separation of the two highest local peaks of the smoothed envelope | $\Delta t=\lvert k_2-k_1\rvert\,10^6/f_s$, in microseconds |

For this feature, calculate the Hilbert envelope on the **full input without padding**, then apply a **centered five-point arithmetic mean**:

$$\widetilde a_n=\frac{a_{n-2}+a_{n-1}+a_n+a_{n+1}+a_{n+2}}{5}.$$

Use symmetric boundary extension for this averaging step (`scipy.ndimage.uniform_filter1d`, `size=5`, `mode='reflect'`). Find all local maxima of the smoothed envelope and retain the two with the greatest heights. **Do not add a height threshold, prominence threshold, minimum peak separation, or starting-time exclusion.** Equal heights are resolved by the earlier sample index. Flat peak plateaus follow `scipy.signal.find_peaks`: select the middle sample, rounded down for an even plateau; endpoints are not counted as peaks. Return `NaN` when fewer than two peaks are found.

This function deliberately has a different envelope convention from the four summary functions above: no extension before the Hilbert transform, followed by five-point smoothing. The output is in **microseconds**, using `fs = 26.5216e6` Hz. It measures two local envelope peaks, which may belong to the same echo group; it does not establish the identity of two physical interfaces. This is an optional candidate feature, not a supplied classification rule.

## 5. Spectral features (helper functions provided)

Each spectral feature function accepts `(x, fs)` and returns one scalar. Internally, spectral estimation first applies the periodic Hann window $w_n=0.5-0.5\cos(2\pi n/N)$, then computes an $N$-point real FFT. The weights are $P_k=\alpha_k\lvert\operatorname{FFT}(wx)_k\rvert^2$, where $\alpha_k=2$ for positive-frequency interior bins and 1 for DC and the Nyquist bin (when present). There is no zero padding, extension, or mean subtraction; DC is retained. These are relative spectral weights, not a calibrated PSD. This window is for spectral feature estimation; it does not specify how a decomposition must be implemented.

Let $K$ be the number of one-sided bins, $f_k=kf_s/N$, and $p_k=P_k/\sum_iP_i$. Call each of the following functions separately:

| Feature / identifier | Meaning | Calculation |
|---|---|---|
| Spectral centroid / `spectral_centroid_hz` | Weighted mean frequency | $f_c=\sum_k p_k f_k$ |
| Spectral spread / `spectral_spread_hz` | Spread around the centroid | $\sqrt{\sum_k p_k(f_k-f_c)^2}$ |
| Spectral peak / `spectral_peak_hz` | Frequency bin with greatest weight | $f_{\arg\max_k P_k}$; lowest bin wins a tie |
| Spectral entropy / `spectral_entropy` | Evenness of spectral weights | $-\sum_k p_k\ln p_k/\ln K$ |
| Spectral flatness / `spectral_flatness` | Geometric-to-arithmetic mean of weights | $\exp(K^{-1}\sum_k\ln P_k)/(K^{-1}\sum_kP_k)$ |

Flatness is zero if at least one bin is exactly zero and total weight is positive. All spectral features are `NaN` if all weights are zero. Frequency outputs are in Hz; divide by `1e6` for MHz. FFT-bin spacing is $f_s/N$, not a guarantee that two frequencies that close can be resolved. A spectral peak characterizes the received signal; it does not by itself identify probe specifications. Entropy and flatness depend on the frequency grid and window, so keep those settings consistent.

## 6. Autocorrelation features (helper function provided)

Autocorrelation compares a signal with shifted copies of itself. Set $y_n=x_n-\mu$ and

$$r_\ell=\frac{\sum_{n=0}^{N-1-\ell}y_n y_{n+\ell}}{\sum_{n=0}^{N-1}y_n^2},\quad \ell=1,\ldots,L.$$

Both `autocorr_mean_square(x, max_lag_samples=L)` and `autocorr_max(x, max_lag_samples=L)` return one scalar and use linear correlation (no circular wraparound), with no correction for the decreasing number of overlapping pairs. Choose and report the same maximum lag $L$ for all recordings. A lag of $L$ samples corresponds to $L/f_s$ seconds.

| Function name | Meaning | Calculation |
|---|---|---|
| `autocorr_mean_square` | Average squared similarity over the chosen lags | $L^{-1}\sum_{\ell=1}^L r_\ell^2$ |
| `autocorr_max` | Largest signed correlation over the chosen nonzero lags | $\max_{1\leq\ell\leq L}r_\ell$ |

The maximum may occur at lag 1 and does not necessarily represent a repeated echo or oscillation period. Both features are `NaN` for constant input. The helper does not automatically choose $L$.

## 7. Calling the helpers

The file `codes/feature_helpers.py` requires only NumPy and SciPy, already listed in the assignment requirements. From `HW1.ipynb` in the same `codes/` folder:

```python
import feature_helpers as fh  # already imported by the notebook setup cell

fs = 26.5216e6
# x: one loaded signal or one component; shape (N,)
envelope_cv_value = fh.envelope_cv(x)
centroid_hz = fh.spectral_centroid_hz(x, fs)
peak_spacing_us = fh.envelope_peak_spacing_us(x, fs)

# Optional, after YOU choose a maximum lag L:
# correlation_value = fh.autocorr_mean_square(x, max_lag_samples=L)

# If C has shape (n_components, N), select ONE component by its zero-based index:
# first_fraction = fh.component_energy_fraction(C, component_index=0)
```

Each public function computes one feature and returns one float, never a dictionary, tuple, or feature vector. Underscore-prefixed functions are internal helpers, not student-facing feature interfaces. This demonstrates calls, not a recommended feature vector. No function loads assignment data, chooses features, sets classification thresholds, or produces class labels. Signal inputs must be real, finite, one-dimensional arrays with at least 4 samples; the component-energy function instead accepts the two-dimensional component array described above. Invalid input raises `ValueError`.

## 8. What to explain in your report

Name the features you selected, their definitions, and whether each is computed from a signal or a particular component. State preprocessing and helper settings. Use reference-set statistics and plots to explain your numerical decision rules. Apply the method in both task stages and submit both prediction files as required. There is no requirement to include all features in this guide.

## References for implementation conventions

- [SciPy Hilbert transform and analytic signal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html)
- [SciPy kurtosis: Pearson versus Fisher definitions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html)
