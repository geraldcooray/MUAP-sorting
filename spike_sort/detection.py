"""Spike detection and waveform extraction."""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def bandpass(signal, fs, low_hz, high_hz):
    nyq = fs / 2
    high_hz = min(high_hz, nyq * 0.99)
    b, a = butter(4, [low_hz / nyq, high_hz / nyq], btype="bandpass")
    return filtfilt(b, a, signal)


def detect_spikes(signal, fs, highpass_hz, threshold_mad, refractory_ms,
                   smoothing_ms, noise_baseline_samples=None):
    """Detect one event per MUAP rather than one per phase/turn.

    A multiphasic MUAP crosses a raw |signal| threshold once per lobe, so
    naive peak-picking on |filtered signal| fragments a single MUAP into
    several "spikes" with different shapes (the bug this fixes). Instead:
    rectify + smooth over ~1 MUAP-phase width to collapse each MUAP's
    lobes into one broad envelope bump, detect peaks on that envelope,
    then snap each detection back to the true local extreme of the
    filtered signal for precise alignment.

    Minimum spacing between detections is governed purely by
    `refractory_ms`, independent of the waveform-extraction window used
    downstream -- so two spikes closer together than the extraction window
    can still both be detected. `extract_waveforms` allows their snippets
    to overlap rather than dropping one to preserve non-overlap.

    The noise-sigma estimate is normally the median envelope over the
    WHOLE signal (assumes spikes are a minority of samples, so the median
    tracks quiet baseline). If `noise_baseline_samples` is given, only the
    signal's first that-many samples are used for this estimate instead --
    meant for a known spike-free lead-in segment, so busy recordings (many
    concurrently firing units, little quiet baseline overall) don't
    inflate the median and silently raise the effective threshold.
    """
    nyq = fs / 2
    b, a = butter(4, highpass_hz / nyq, btype="highpass")
    filtered = filtfilt(b, a, signal)
    rectified = np.abs(filtered)

    smooth_samples = max(1, int(round(smoothing_ms * 1e-3 * fs)))
    kernel = np.ones(smooth_samples) / smooth_samples
    envelope = np.convolve(rectified, kernel, mode="same")

    if noise_baseline_samples is not None:
        noise_segment = envelope[:min(noise_baseline_samples, len(envelope))]
    else:
        noise_segment = envelope
    sigma = np.median(noise_segment) / 0.6745
    threshold = threshold_mad * sigma
    refractory_samples = max(1, int(round(refractory_ms * 1e-3 * fs)))

    env_peaks, _ = find_peaks(envelope, height=threshold, distance=refractory_samples)

    peaks = []
    for p in env_peaks:
        lo, hi = max(0, p - smooth_samples), min(len(filtered), p + smooth_samples + 1)
        peaks.append(lo + int(np.argmax(rectified[lo:hi])))
    peaks = np.array(sorted(set(peaks)))

    return peaks, threshold, filtered, envelope


def extract_waveforms(measured, peaks, fs, window_ms, baseline_window_ms=None):
    """Cut a +/-window_ms snippet around each peak, baselined against a
    (typically wider) +/-baseline_window_ms window rather than its own
    span.

    Subtracting a snippet's own mean forces every extracted waveform to
    have ~zero mean by construction (the row-sum is a fixed constraint of
    the subtraction itself), which erases genuine non-zero DC offsets
    between MUAPs -- e.g. from needle position/orientation relative to the
    source -- that are real, clinically useful shape information, not
    baseline drift. Estimating the baseline from a separate, wider window
    instead only removes slow drift local to that spike's neighbourhood;
    it doesn't force the narrower returned snippet's own mean to be zero,
    so a real DC offset specific to that MUAP survives into the waveform
    used for PCA/clustering.

    `baseline_window_ms` defaults to `window_ms` (previous behaviour:
    baseline = the snippet's own mean) when not given."""
    half_win = int(round(window_ms * 1e-3 * fs))
    baseline_half_win = int(round(
        (baseline_window_ms if baseline_window_ms is not None else window_ms) * 1e-3 * fs))

    max_half = max(half_win, baseline_half_win)
    valid = (peaks - max_half >= 0) & (peaks + max_half < len(measured))
    peaks = peaks[valid]

    waveforms = np.stack([
        measured[p - half_win:p + half_win]
        - measured[p - baseline_half_win:p + baseline_half_win].mean()
        for p in peaks
    ])
    return peaks, waveforms, half_win
