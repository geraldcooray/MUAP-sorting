"""Per-unit firing-regularity and standard MUAP parameters."""
import numpy as np

from spike_sort.features import onset_offset


def isi_cv(times):
    """Coefficient of variation of inter-spike intervals -- a standard
    firing-regularity index; lower = more temporally regular."""
    if len(times) < 3:
        return float("nan")
    isis = np.diff(np.sort(times))
    return isis.std() / isis.mean() if isis.mean() > 0 else float("nan")


def muap_metrics(mean_wf, fs, k=2.5, turn_thresh_frac=0.05):
    """Standard MUAP parameters from a mean waveform: onset/offset
    (baseline-threshold crossing), duration, peak-to-peak amplitude,
    phases (baseline crossings + 1) and turns (direction reversals
    exceeding a small fraction of the amplitude)."""
    n = len(mean_wf)
    onset, offset, baseline_mean = onset_offset(mean_wf, k=k)
    segment = mean_wf[onset:offset + 1]
    amp_max, amp_min = segment.max(), segment.min()
    amplitude = amp_max - amp_min
    duration_ms = (offset - onset) / fs * 1000

    zero = baseline_mean
    signs = np.sign(segment - zero)
    signs[signs == 0] = 1
    phases = int(np.sum(np.diff(signs) != 0)) + 1

    turn_thresh = max(turn_thresh_frac * amplitude, 1e-9)
    turns = 0
    last_extreme = segment[0]
    direction = 0
    for v in segment[1:]:
        d = v - last_extreme
        if direction == 0 and abs(d) > turn_thresh:
            direction = 1 if d > 0 else -1
            last_extreme = v
        elif direction != 0:
            if direction > 0 and d < -turn_thresh:
                turns += 1
                direction = -1
                last_extreme = v
            elif direction < 0 and d > turn_thresh:
                turns += 1
                direction = 1
                last_extreme = v
            elif (direction > 0 and v > last_extreme) or (direction < 0 and v < last_extreme):
                last_extreme = v

    return {
        "onset": onset, "offset": offset, "duration_ms": duration_ms,
        "amplitude": amplitude, "amp_max": amp_max, "amp_min": amp_min,
        "phases": phases, "turns": turns,
        "area": np.trapezoid(np.abs(segment), dx=1 / fs * 1000),
    }
