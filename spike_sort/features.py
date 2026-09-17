"""Per-spike waveform measurements and shape-feature extraction for
clustering (PCA, wavelet coefficients, or t-SNE)."""
import numpy as np
import pywt
from scipy.fft import dct, idct
from scipy.stats import normaltest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def dct_highpass_waveforms(waveforms, n_low):
    """Zero the `n_low` lowest-order DCT coefficients of every extracted
    waveform, then transform back to the time domain.

    Each row (one spike snippet) is DCT-II transformed (orthonormal
    norm), coefficients 0 .. n_low-1 -- the DC term and the slowest
    half-cosine components, which carry residual baseline offset and slow
    drift across the extraction window rather than MUAP shape -- are set
    to zero, and the row is inverse-transformed. This is a smooth,
    ringing-free alternative to a time-domain high-pass for cleaning the
    per-trial waveforms before shape clustering / averaging.

    `n_low <= 0` returns the input unchanged. Works on a single waveform
    (1-D) or a stack (n_spikes x n_samples)."""
    if n_low <= 0:
        return waveforms
    wf = np.asarray(waveforms, dtype=float)
    axis = wf.ndim - 1
    coeffs = dct(wf, type=2, norm="ortho", axis=axis)
    n_low = min(n_low, coeffs.shape[axis])
    sl = [slice(None)] * wf.ndim
    sl[axis] = slice(0, n_low)
    coeffs[tuple(sl)] = 0.0
    return idct(coeffs, type=2, norm="ortho", axis=axis)


def onset_offset(wf, edge_frac=0.1, k=2.5):
    """Baseline-threshold onset/offset indices for one waveform (mean or
    single-spike): the first/last sample where |wf - baseline| exceeds
    k robust-sigma, using the window edges as the baseline estimate."""
    n = len(wf)
    edge_len = max(1, int(n * edge_frac))
    baseline = np.concatenate([wf[:edge_len], wf[-edge_len:]])
    baseline_mean, baseline_std = baseline.mean(), baseline.std()
    threshold = max(k * baseline_std, 1e-9)

    above = np.abs(wf - baseline_mean) > threshold
    if above.any():
        onset = int(np.argmax(above))
        offset = int(n - 1 - np.argmax(above[::-1]))
    else:
        onset, offset = 0, n - 1
    return onset, offset, baseline_mean


def spike_amplitude_duration(waveforms, fs, frac=0.5):
    """Per-spike peak-to-peak amplitude, full-width-at-fraction-max
    duration (ms), and polarity, used as explicit clustering features so
    units are grouped by these standard MUAP parameters and not just by
    overall waveform shape.

    Unlike `onset_offset` (which estimates a baseline noise level from a
    handful of edge samples -- too unstable on a single, non-averaged
    trial), width here is measured relative to the spike's own peak
    magnitude, which stays well-conditioned per spike.

    Amplitude (max - min) and duration are both mathematically identical
    for a waveform and its polarity-inverted mirror image -- e.g.
    duration is deliberately measured relative to |peak|, sign-agnostic,
    so a spike and its mirror get the same width. `polarities` (+1/-1,
    the sign of each spike's largest-magnitude deflection) is returned
    separately so callers can add it as its own feature and actually
    distinguish a waveform from its mirror, instead of relying on
    amplitude/duration to do so (they can't, by construction) or on the
    shape features alone to carry enough weight against amplitude/
    duration's up-weighting."""
    n_spikes, n = waveforms.shape
    amplitudes = np.empty(n_spikes)
    durations = np.empty(n_spikes)
    polarities = np.empty(n_spikes)
    for i, wf in enumerate(waveforms):
        amplitudes[i] = wf.max() - wf.min()
        peak_idx = np.argmax(np.abs(wf))
        peak_val = wf[peak_idx]
        sign = 1.0 if peak_val >= 0 else -1.0
        polarities[i] = sign
        level = frac * sign * peak_val  # = frac * |peak_val|, sign-agnostic

        left = peak_idx
        while left > 0 and sign * wf[left] >= level:
            left -= 1
        right = peak_idx
        while right < n - 1 and sign * wf[right] >= level:
            right += 1
        durations[i] = (right - left) / fs * 1000
    return amplitudes, durations, polarities


def shape_features_pca(waveforms, n_components, seed):
    n_components = min(n_components, waveforms.shape[0], waveforms.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    return pca.fit_transform(waveforms)


def shape_features_wavelet(waveforms, n_components, wavelet, level, clip_percentile=1.0):
    """Discrete wavelet transform shape features, WaveClus-style (Quiroga et
    al. 2004): decompose each waveform, then keep only the `n_components`
    coefficients whose distribution *across spikes* deviates most from
    Gaussian (D'Agostino-Pearson K^2 statistic). A coefficient that mixes
    two/more distinct MUAP shapes tends to be multimodal (non-Gaussian)
    across the spike population, while a coefficient dominated by noise
    looks Gaussian regardless of shape -- so this selects the coefficients
    that actually carry shape information worth clustering on, instead of
    using all of them (most of which are noise) or relying on PCA's
    global-variance criterion, which can be dominated by amplitude rather
    than shape differences.

    Each coefficient is winsorized to its [clip_percentile,
    100-clip_percentile] range (across spikes) before scoring and before
    being returned. The D'Agostino K^2 statistic used for scoring reacts
    to any heavy tail, including a single outlier waveform's stray
    coefficient -- so without clipping, one atypical spike can both hijack
    the "most discriminative" ranking (getting a genuinely uninformative
    coefficient selected) and, if that coefficient is returned unclipped,
    single-handedly dominate distance-based downstream clustering with one
    wildly out-of-scale coordinate. Set clip_percentile=0 to disable."""
    coeff_matrix = np.stack([
        np.concatenate(pywt.wavedec(wf, wavelet, level=level)) for wf in waveforms
    ])

    if clip_percentile > 0:
        lo = np.percentile(coeff_matrix, clip_percentile, axis=0)
        hi = np.percentile(coeff_matrix, 100 - clip_percentile, axis=0)
        coeff_matrix = np.clip(coeff_matrix, lo, hi)

    n_components = min(n_components, coeff_matrix.shape[1])

    scores = np.zeros(coeff_matrix.shape[1])
    for j in range(coeff_matrix.shape[1]):
        col = coeff_matrix[:, j]
        if np.ptp(col) == 0:
            continue
        scores[j], _ = normaltest(col)

    top_idx = np.argsort(scores)[::-1][:n_components]
    return coeff_matrix[:, top_idx]


def shape_features_tsne(waveforms, n_components, seed, perplexity=30.0):
    """Nonlinear shape-feature embedding via t-SNE (van der Maaten & Hinton
    2008): unlike PCA's linear projection onto directions of maximum global
    variance, t-SNE optimizes a low-dimensional layout that preserves each
    spike's *local* neighborhood structure -- so it can pull apart MUAP
    shapes that overlap heavily in their leading principal components but
    are still locally well-separated in the full waveform space. The cost
    is that t-SNE has no fit/predict split: it always jointly re-embeds
    every spike in the current analysis window, so features aren't
    directly comparable across separate runs/windows the way PCA
    components or wavelet coefficients are.

    Perplexity (roughly, the effective number of neighbors considered per
    point) must be less than the number of spikes; it's clamped down
    automatically for small spike counts rather than erroring. sklearn
    requires n_components <= 3 for the fast "barnes_hut" method, so higher
    values fall back to the slower "exact" method automatically."""
    n_spikes = waveforms.shape[0]
    effective_perplexity = min(perplexity, max(2.0, (n_spikes - 1) / 3.0))
    method = "barnes_hut" if n_components <= 3 else "exact"
    tsne = TSNE(n_components=n_components, perplexity=effective_perplexity,
                random_state=seed, method=method, init="pca")
    return tsne.fit_transform(waveforms)


def build_features(waveforms, fs, n_components, seed, amp_dur_weight,
                    method="pca", wavelet="sym4", wavelet_level=None, tsne_perplexity=30.0,
                    wavelet_clip_percentile=1.0, polarity_weight=1.0):
    """Shape + explicit peak-to-peak amplitude, duration, and polarity, all
    standardized to unit variance so no feature dominates by scale alone,
    then amplitude/duration are up-weighted by `amp_dur_weight` and
    polarity by `polarity_weight` so units end up homogeneous in the
    properties that matter clinically, not just in raw shape. `method`
    selects the shape representation: "pca" (top principal components of
    the raw waveform), "wavelet" (most discriminative DWT coefficients,
    winsorized against single-outlier domination -- see
    `shape_features_wavelet`), or "tsne" (nonlinear neighborhood-preserving
    embedding -- see `shape_features_tsne`).

    Polarity is included as its own explicit feature because amplitude and
    duration are both blind to it by construction (see
    `spike_amplitude_duration`) -- without it, a spike and its
    polarity-inverted mirror image can end up closer together in the
    up-weighted amplitude/duration dimensions than either is to a
    genuinely different spike, even though they should usually be treated
    as distinct. Set polarity_weight=0 to omit its influence."""
    if method == "wavelet":
        shape_features = shape_features_wavelet(waveforms, n_components, wavelet, wavelet_level,
                                                  clip_percentile=wavelet_clip_percentile)
    elif method == "tsne":
        shape_features = shape_features_tsne(waveforms, n_components, seed, tsne_perplexity)
    else:
        shape_features = shape_features_pca(waveforms, n_components, seed)

    amplitudes, durations, polarities = spike_amplitude_duration(waveforms, fs)
    features = np.column_stack([shape_features, amplitudes, durations, polarities])
    features = StandardScaler().fit_transform(features)
    features[:, -3:-1] *= amp_dur_weight   # up-weight amplitude & duration columns
    features[:, -1] *= polarity_weight     # up-weight polarity column
    return features
