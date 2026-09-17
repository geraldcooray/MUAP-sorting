"""Selection criteria for averaged (mean-template) MUAP waveforms -- a
secondary, post-hoc filter applied AFTER spike sorting, on the per-type
mean waveform each surviving cluster already has (not on individual
trials, and not on anything upstream of clustering). Two independent
shape checks, both of which a type must pass, plus a minimum-trial-count
floor:

    1. Low-frequency drift (evaluate_low_frequency_drift, meant to run
       FIRST as a cheap early filter): a DCT-II decomposition of the
       averaged waveform, checking what FRACTION of its total energy is
       carried by its lowest few (near-DC) terms (not their raw
       magnitude, which would scale with each type's own amplitude and
       so not be comparable across types). A real MUAP's *shape* is what
       carries the information; if the lowest terms dominate the energy,
       the average is mostly just a slow trend/offset -- e.g. misaligned
       detections averaged together smearing out into a slow drift --
       rather than a real, sharply-defined transient.

    2. High-frequency-power centering (evaluate_highfreq_centering): a
       real, cleanly-isolated MUAP is (by construction of the spike
       extraction step) cut around its own detected peak, and that peak
       is where the waveform's sharpest, fastest deflection -- its
       high-frequency content -- actually sits. So: high-pass filter the
       averaged waveform (>= HIGHPASS_HZ), square it to get an
       instantaneous power curve over time, and check where THAT curve
       peaks. For a genuine, well-aligned MUAP it should peak right at
       the window's center (t=0); if it peaks elsewhere, the average is
       probably not a real, cleanly-isolated single unit (misaligned
       detections averaged together, drift, a different unit's
       transient dominating).

Neither check alters or discards any underlying data -- they only flag
a type as not "survived".

This module also keeps a general CWT scalogram helper (compute_scalogram)
for visualizing the same averaged waveform's full time-frequency content
alongside the high-frequency power curve, even though the curve itself
(not the full 2D map) is what the second criterion actually uses.
"""
import numpy as np
import pywt
from scipy.fft import dct, idct
from scipy.signal import butter, filtfilt

# ---------------------------------------------------------------------------
TF_WAVELET = "cmor1.5-1.0"  # complex Morlet -- good time/frequency trade-off for a short,
                             # single-transient signal like a MUAP (real-valued wavelets like
                             # 'morl' give a much noisier magnitude map for the same signal)
TF_FMIN_HZ = 50.0
TF_FMAX_HZ = 5000.0         # upper end of the clinically relevant MUAP frequency content
TF_N_FREQS = 64

DCT_N_LOW_TERMS = 3              # how many of the lowest-frequency DCT-II coefficients to
                                   # check (includes the DC/mean term at index 0)
DCT_MAX_LOW_FREQ_POWER_FRAC = 0.7   # a type is rejected if the fraction of its total DCT
                                   # energy (Parseval's theorem: sum of squared coefficients =
                                   # sum of squared samples) carried by those lowest n_terms
                                   # coefficients exceeds this -- a fraction, not a raw
                                   # magnitude, so it's scale-invariant: comparable across types
                                   # regardless of their own amplitude

HIGHPASS_HZ = 500.0         # cutoff for the high-frequency power curve the criterion is based on
CENTER_TOL_MS = 1.0         # the high-frequency power curve's peak must fall within +/- this
                             # many ms of window center (t=0) to survive

MIN_TRIALS = 10             # a type's average must be built from at least this many individual
                             # spikes to survive -- an average of very few trials is a noisy,
                             # unreliable estimate of the true MUAP shape regardless of how well
                             # it does on the shape checks above


def low_frequency_dct_terms(waveform, n_terms=DCT_N_LOW_TERMS):
    """The lowest n_terms coefficients of a DCT-II (norm="ortho")
    decomposition of `waveform` -- its near-DC / slowest-varying
    content, index 0 being the (scaled) mean."""
    coeffs = dct(np.asarray(waveform, dtype=float), type=2, norm="ortho")
    return coeffs[:n_terms]


def low_frequency_reconstruction(waveform, n_terms=DCT_N_LOW_TERMS):
    """Inverse DCT-II of `waveform` using only its lowest n_terms
    coefficients (everything else zeroed) -- i.e. the slow trend/offset
    the low-frequency-drift criterion is actually measuring, rendered
    back in the original waveform's own units/timebase so it can be
    plotted directly against it."""
    waveform = np.asarray(waveform, dtype=float)
    coeffs = dct(waveform, type=2, norm="ortho")
    truncated = np.zeros_like(coeffs)
    truncated[:n_terms] = coeffs[:n_terms]
    return idct(truncated, type=2, norm="ortho")


def evaluate_low_frequency_drift(mean_wf, n_terms=DCT_N_LOW_TERMS,
                                  max_power_frac=DCT_MAX_LOW_FREQ_POWER_FRAC):
    """Score one averaged MUAP waveform against the low-frequency-drift
    criterion (see module docstring): the fraction of its total DCT
    energy carried by the lowest n_terms coefficients must not exceed
    max_power_frac.

    Returns a dict with `survived` (bool) plus the diagnostics behind it
    (`low_terms`, `low_freq_power_frac`) so callers can report/plot why a
    type passed or failed, not just the verdict."""
    mean_wf = np.asarray(mean_wf, dtype=float)
    all_terms = dct(mean_wf, type=2, norm="ortho")
    low_terms = all_terms[:n_terms]
    total_energy = float(np.sum(all_terms ** 2))
    low_freq_power_frac = (float(np.sum(low_terms ** 2)) / total_energy) if total_energy > 0 \
        else 1.0
    survived = bool(low_freq_power_frac <= max_power_frac)
    return dict(survived=survived, low_terms=low_terms, low_freq_power_frac=low_freq_power_frac)


def compute_scalogram(waveform, fs, wavelet=TF_WAVELET, fmin=TF_FMIN_HZ, fmax=TF_FMAX_HZ,
                       n_freqs=TF_N_FREQS):
    """Continuous-wavelet-transform time-frequency power map of one
    waveform: log-spaced frequencies from fmin to min(fmax, fs/2), a
    complex Morlet wavelet. Purely for visualization context -- see
    module docstring for why the selection criterion itself uses
    highpass_power_curve instead.

    Returns (power, freqs) where power has shape (n_freqs, len(waveform))
    and freqs is in Hz, ascending."""
    fmax = min(fmax, fs / 2)
    freqs = np.geomspace(fmin, fmax, n_freqs)
    dt = 1.0 / fs
    scales = pywt.central_frequency(wavelet) / (freqs * dt)
    coeffs, freqs_out = pywt.cwt(waveform, scales, wavelet, sampling_period=dt)
    power = np.abs(coeffs) ** 2
    return power, freqs_out


def highpass_power_curve(waveform, fs, highpass_hz=HIGHPASS_HZ):
    """Zero-phase Butterworth high-pass filtered waveform, squared -- an
    instantaneous power curve over time restricted to content at or above
    highpass_hz."""
    waveform = np.asarray(waveform, dtype=float)
    nyq = fs / 2
    cutoff = min(highpass_hz, nyq * 0.99)
    b, a = butter(4, cutoff / nyq, btype="highpass")
    filtered = filtfilt(b, a, waveform)
    return filtered ** 2


def evaluate_highfreq_centering(mean_wf, t_wave_ms, fs, n_spikes=None, highpass_hz=HIGHPASS_HZ,
                                 center_tol_ms=CENTER_TOL_MS, min_trials=MIN_TRIALS):
    """Score one averaged MUAP waveform against the high-frequency-power
    centering criterion (see module docstring): the high-pass (>=
    highpass_hz) power curve's peak must fall within center_tol_ms of the
    window's center (t=0). If `n_spikes` (the number of individual trials
    the average was built from) is given, the average must also be built
    from at least `min_trials` spikes.

    Returns a dict with `survived` (bool) plus the diagnostics behind it
    (`hf_power`, `peak_t_ms`, `centered`, `enough_trials`) so callers can
    report/plot why a type passed or failed, not just the verdict."""
    mean_wf = np.asarray(mean_wf, dtype=float)
    t_wave_ms = np.asarray(t_wave_ms, dtype=float)

    hf_power = highpass_power_curve(mean_wf, fs, highpass_hz=highpass_hz)
    peak_idx = int(np.argmax(hf_power))
    peak_t_ms = float(t_wave_ms[peak_idx])

    centered = abs(peak_t_ms) <= center_tol_ms
    enough_trials = True if n_spikes is None else (n_spikes >= min_trials)
    survived = bool(centered and enough_trials)

    return dict(survived=survived, hf_power=hf_power, peak_t_ms=peak_t_ms,
                centered=centered, enough_trials=enough_trials)
