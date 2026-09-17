"""Post-clustering cleanup: prune biologically implausible short
inter-spike intervals within a single sorted spike type."""
import numpy as np


def enforce_min_isi_per_type(peaks_t, labels, peak_amps, t_min_s):
    """For each spike type (unique label), remove spikes that fall within
    `t_min_s` seconds of a higher-amplitude spike of the *same* type,
    keeping the larger-amplitude spike of each too-close group.

    A single motor unit cannot re-fire faster than its refractory period --
    t_min_s = 0.010 s (100 Hz) is already far above physiologically
    plausible voluntary firing rates -- so an ISI shorter than that within
    one sorted type is much more likely a detection/clustering artifact
    (e.g. the same discharge picked up twice, or a neighbouring unit's
    spike misclassified into this type) than a genuine double-fire.
    Keeping the larger-|amplitude| spike of each conflicting pair is a
    simple, consistent tie-break: it's usually the cleaner detection
    (closer to the true peak), and this rule picks the same physical event
    out of a whole chain of near-simultaneous detections rather than an
    arbitrary one.

    This uses the same greedy, amplitude-priority suppression
    `scipy.signal.find_peaks` uses for its `distance` parameter: visit
    candidates largest-|amplitude|-first, keep one only if it's far enough
    in time from every *already-kept* spike of that type. Judging
    acceptance against the growing kept set (not just the nearest
    neighbor) is what makes this correct for chains of 3+ mutually-close
    spikes, where naive pairwise comparison can accept two spikes that are
    each individually far from one neighbor but still too close to each
    other once one of them is removed from between them.

    Returns a boolean `keep` mask, same length/order as the inputs --
    apply it (e.g. `peaks_t[keep]`) to every per-spike array (peaks_t,
    labels, peak_amps, waveforms, features, ...) to get the cleaned-up
    spike list. Every type that starts with >=1 spike keeps at least one
    (its largest-amplitude spike is always accepted first), so this never
    empties out a type entirely.
    """
    n = len(peaks_t)
    keep = np.ones(n, dtype=bool)

    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        if idx.size < 2:
            continue

        order = idx[np.argsort(-np.abs(peak_amps[idx]))]  # largest |amplitude| first
        kept_times = []  # kept spikes of this type, in ascending time order

        for i in order:
            ti = peaks_t[i]
            pos = np.searchsorted(kept_times, ti)
            too_close = (
                (pos > 0 and ti - kept_times[pos - 1] < t_min_s)
                or (pos < len(kept_times) and kept_times[pos] - ti < t_min_s)
            )
            if too_close:
                keep[i] = False
            else:
                kept_times.insert(pos, ti)

    return keep
