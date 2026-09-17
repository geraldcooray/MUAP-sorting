#!/usr/bin/env python3
"""Run the real spike-sorting pipeline on each simulated_traces/
simulated_trace_n<K>.npz (see simulate_emg_traces.py) and compare its
ESTIMATED spike types against the simulation's own TRUE spike types
(known exactly, since we generated them).

For each trace:
    1. The raw simulated signal is written out as a one-column EMGRUN
       file (+ matching Header.txt row) so the exact same pipeline used
       on real recordings (call_spike_sort_emg.run_pipeline: detect_spikes
       -> extract_waveforms -> t-SNE features -> k-means, with the usual
       silhouette-sweep cluster-count selection) can be run on it
       unmodified.
    2. Every estimated spike is matched to the nearest TRUE spike (from
       any true unit) within MATCH_TOL_MS; unmatched estimated spikes
       (too far from any true spike -- a false-positive detection, e.g.
       triggered by noise) get no true-unit label.
    3. Each estimated cluster is assigned the true unit its members match
       most often (majority vote) and a "purity" = that majority fraction
       -- 1.0 means every one of that cluster's matched spikes came from
       the same true unit.

Three figures are generated per trace (detection-only and
true-vs-detected-window figures are still implemented below --
plot_detected_unclustered, plot_true_vs_detected_window -- just not
called from main(), so they're easy to re-enable if needed):

    spikesort_muap_shapes_n<K>.png -- three panels: the SIMULATED (input)
                                  MUAP templates this trace was built
                                  from; the ESTIMATED (recovered) mean-
                                  template shapes spike-sorting found,
                                  same grey-trials/black-mean panel style,
                                  colored/labeled by true<->estimated
                                  matching; and a narrow panel plotting
                                  every pairwise L2 distance between the
                                  estimated types' mean waveforms, sorted
                                  ascending, marking l2_threshold_used (if
                                  a merge ran) and the 20th-percentile
                                  distance for reference either way.
    spikesort_true_vs_est_n<K>.png -- ACTUAL vs ESTIMATED spikes: the TRUE
                                  spike-type raster (one row per simulated
                                  unit) above the ESTIMATED spike-type
                                  raster (one row per recovered cluster,
                                  colored by its best-matching true unit),
                                  plus a matched-spike-count confusion
                                  matrix.
    spikesort_muap_shapes_superposition_n<K>.png -- the SIMULATED and
                                  ESTIMATED MUAP galleries again, plus a
                                  third gallery with every estimated type
                                  that spike_sort.superposition flagged as
                                  really the sum of two other estimated
                                  types (spikes consistently superimposing)
                                  removed.
    spikesort_compare_manifest.csv -- one row per estimated type, across
                                  every trace: cluster method, matched true
                                  unit, purity, spike counts, and whether
                                  it was flagged as a superposition (of
                                  which pair, at what residual).

Non-kmeans --cluster-method and non-default --feature-method/--n-components
runs write to '_<suffix>'-tagged filenames so they don't clobber each other.

Usage:
    python spikesort_simulated_traces.py [--feature-method tsne|pca|wavelet] [--n-components N]
        [--cluster-method kmeans|gmm|tmixture|hdbscan|spc]
        [--l2-merge-threshold D | --l2-merge-percentile P]
"""
import argparse
import copy
import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np

import call_spike_sort_emg
from call_spike_sort_emg import BASE_PARAMS, run_pipeline
from simulate_emg_traces import LIBRARY_PATH, N_MUAPS_LIST, OUT_DIR, load_library
from spike_sort.metrics import muap_metrics
from spike_sort.plotting import plot_muap_panel
from spike_sort.superposition import find_superposition_candidates, format_superposition_report

SPIKESORT_INPUT_DIR = os.path.join(OUT_DIR, "spikesort_input")
MATCH_TOL_MS = 1.5  # an estimated spike within this many ms of a true spike counts as a match
SUPERPOSITION_MAX_SHIFT_MS = 3.0  # widest relative discharge-timing offset (ms) two independently
                                   # firing units can have while still landing in one detection
                                   # window -- the +/- shift range find_superposition_candidates
                                   # searches when testing whether an estimated type is really the
                                   # sum of two others (see spike_sort.superposition)
DETECTION_THRESHOLD_MAD = 20.0  # MAD multiplier for the spike-detection threshold, applied
                                 # against the sigma estimated from each trace's spike-free
                                 # baseline segment (see simulate_emg_traces.py's BASELINE_S and
                                 # detect_spikes' noise_baseline_samples) rather than the whole
                                 # record. That baseline-only sigma is much smaller than the
                                 # old whole-signal-median estimate, so it needs a correspondingly
                                 # larger multiplier -- otherwise it sits low enough to catch
                                 # secondary lobes/turns of each multiphasic MUAP as spurious
                                 # extra detections. A sweep (n=3/5/10/20) found 20 the best
                                 # recall-vs-false-positive balance; 10 trades more false
                                 # positives (more side-lobe detections) for higher recall.

SIM_COLORS = plt.get_cmap("tab20").colors
UNMATCHED_COLOR = "0.75"
GALLERY_MAX_COLS = 6
TRUE_VS_DETECTED_WINDOW_S = 5.0  # short enough to actually see individual spikes by eye
TRUE_VS_DETECTED_WINDOW_S_SHORT = 1.0  # an even tighter zoom, for eyeballing individual spikes

_UNIT_NAME_RE = re.compile(r"^unit\d+_(.*)_type(\d+)$")

# Give the silhouette sweep enough headroom to find as many clusters as the largest
# simulated trace actually has true units for (the module default, 20, is exactly the
# n=20 case's true count with zero margin for any spurious extra cluster).
call_spike_sort_emg.SILHOUETTE_MAX_N = max(N_MUAPS_LIST) + 5


def load_library_lookup():
    """{(subject, type_idx): entry} for every muap_library.py template, so
    a simulated trace's ground-truth units (which only remember which
    template they came from, not the waveform itself) can look their own
    input MUAP shape back up."""
    return {(e["subject"], e["type_idx"]): e for e in load_library(LIBRARY_PATH)}


def load_simulated_trace(npz_path, library_lookup):
    """Parse simulate_emg_traces.py's flat unit<i>_<subject>_type<c>_*
    keys back into ground-truth per-unit spike times, with each unit's
    own input MUAP template (waveform, trials, t_wave_ms) attached from
    the library it was drawn from. Also returns the trace's spike-free
    baseline_s lead-in (0.0 for older traces saved before it existed)."""
    data = np.load(npz_path, allow_pickle=True)
    trace, fs = data["trace"], float(data["fs"])
    baseline_s = float(data["baseline_s"]) if "baseline_s" in data.files else 0.0

    st_suffix = "_spike_times_s"
    true_units = []
    for key in data.files:
        if not (key.startswith("unit") and key.endswith(st_suffix)):
            continue
        name = key[: -len(st_suffix)]
        m = _UNIT_NAME_RE.match(name)
        subject, type_idx = m.group(1), int(m.group(2))
        e = library_lookup[(subject, type_idx)]
        true_units.append(dict(
            name=name, subject=subject, type_idx=type_idx,
            spike_times=np.atleast_1d(data[key]), template=e["mean_wf"],
            trial_waveforms=e["trial_waveforms"], t_wave_ms=e["t_wave_ms"], fs=e["fs"],
        ))
    return trace, fs, true_units, baseline_s


def write_emgrun_file(trace, fs, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    np.savetxt(path, trace, fmt="%.6f")
    header_path = os.path.join(out_dir, "Header.txt")
    is_new = not os.path.isfile(header_path)
    with open(header_path, "a") as f:
        if is_new:
            f.write("Type\tSamples\tSmplFrq\tFile\n")
        f.write(f"EMGRUN\t{len(trace)}\t{int(fs)}\t{filename}\n")
    return path


def sort_simulated_trace(trace, fs, filename, label, feature_method="tsne", n_components=2,
                          cluster_method="kmeans",
                          threshold_mad=DETECTION_THRESHOLD_MAD, noise_baseline_samples=None,
                          tsne_perplexity=None, l2_merge_threshold=None, l2_merge_percentile=None,
                          l2_merge_elbow=False, l2_merge_outlier_fence_k=None,
                          l2_merge_normalize_area=False, param_overrides=None):
    params = copy.deepcopy(BASE_PARAMS)
    params["filename"] = filename
    params["fs"] = fs
    params["label"] = label
    params["feature_method"] = feature_method
    params["n_components"] = n_components
    params["cluster_method"] = cluster_method
    if tsne_perplexity is not None:
        params["tsne_perplexity"] = tsne_perplexity
    params["l2_merge_threshold"] = l2_merge_threshold
    params["l2_merge_percentile"] = l2_merge_percentile
    params["l2_merge_elbow"] = l2_merge_elbow
    params["l2_merge_outlier_fence_k"] = l2_merge_outlier_fence_k
    params["l2_merge_normalize_area"] = l2_merge_normalize_area
    # The library templates were built with only a very mild DCT high-pass (the 2 lowest
    # coefficients dropped -- see build_muap_library.py / dct_muap_compare.py), so they keep
    # much more low-frequency content than a typical real needle-EMG MUAP. BASE_PARAMS'
    # default 500 Hz detection highpass strips nearly all of that away (confirmed: it leaves
    # only 3 detectable peaks in a trace with ~285 true spikes); 100 Hz keeps enough of the
    # template's own energy for the usual MAD-threshold detector to actually find them.
    params["highpass"] = 100.0
    params["threshold_mad"] = threshold_mad
    # Estimate the noise floor from the trace's known spike-free baseline segment (see
    # simulate_emg_traces.py's BASELINE_S) rather than the whole record -- with many
    # concurrently firing units and a clean SNR, spike energy can dominate enough of the
    # record that the usual whole-signal median inflates the noise estimate and silently
    # raises the effective threshold (observed: n=10/20 traces collapsing to near-zero
    # detections at a fixed threshold_mad once the trace got quiet enough).
    params["noise_baseline_samples"] = noise_baseline_samples
    # Density/superparamagnetic methods (hdbscan, spc) need their own knobs tuned per feature
    # space -- passed straight through here so callers can sweep them without a new kwarg each.
    for key, value in (param_overrides or {}).items():
        params[key] = value
    return run_pipeline(params)


def match_to_ground_truth(est_peaks_t, true_units, tol_s):
    """For each estimated spike time, the index of the true unit with the
    nearest true spike within tol_s (else -1 = unmatched)."""
    all_true_t, all_true_unit = [], []
    for i, u in enumerate(true_units):
        all_true_t.append(u["spike_times"])
        all_true_unit.append(np.full(len(u["spike_times"]), i))
    all_true_t = np.concatenate(all_true_t) if all_true_t else np.array([])
    all_true_unit = np.concatenate(all_true_unit) if all_true_unit else np.array([], dtype=int)

    order = np.argsort(all_true_t)
    sorted_t, sorted_unit = all_true_t[order], all_true_unit[order]

    matched_unit = np.full(len(est_peaks_t), -1)
    for i, pt in enumerate(est_peaks_t):
        if sorted_t.size == 0:
            break
        j = np.searchsorted(sorted_t, pt)
        candidates = [k for k in (j - 1, j) if 0 <= k < sorted_t.size]
        if not candidates:
            continue
        best = min(candidates, key=lambda k: abs(sorted_t[k] - pt))
        if abs(sorted_t[best] - pt) <= tol_s:
            matched_unit[i] = sorted_unit[best]
    return matched_unit


def compute_match_purity(true_units, est_labels, matched_unit):
    """Majority-vote match + purity for each estimated cluster: which
    true unit its members match most often, and what fraction of them
    agree. Split out from plot_comparison so callers that only want the
    MUAP gallery (which colors/labels panels by this) don't have to
    render/save the raster+confusion-matrix figure just to get it."""
    n_true = len(true_units)
    n_est = int(est_labels.max()) + 1 if est_labels.size else 0
    est_match, est_purity = [], []
    for c in range(n_est):
        mu = matched_unit[est_labels == c]
        mu_valid = mu[mu >= 0]
        if mu_valid.size == 0:
            est_match.append(-1)
            est_purity.append(0.0)
            continue
        counts = np.bincount(mu_valid, minlength=n_true)
        best = int(np.argmax(counts))
        est_match.append(best)
        est_purity.append(float(counts[best] / mu.size))
    return est_match, est_purity


def plot_comparison(true_units, est_labels, est_peaks_t, matched_unit, n_muaps, out_path):
    n_true = len(true_units)
    n_est = int(est_labels.max()) + 1 if est_labels.size else 0
    est_match, est_purity = compute_match_purity(true_units, est_labels, matched_unit)

    fig, (ax_true, ax_est, ax_conf) = plt.subplots(
        3, 1, figsize=(14, 2.0 * n_true + 2.0 * n_est + 4.5),
        gridspec_kw=dict(height_ratios=[max(n_true, 1), max(n_est, 1), max(n_true, 1) * 0.5 + 2]))

    for i, u in enumerate(true_units):
        color = SIM_COLORS[i % len(SIM_COLORS)]
        ax_true.vlines(u["spike_times"], i - 0.4, i + 0.4, color=color, linewidth=1)
    ax_true.set_ylim(-0.5, max(n_true - 0.5, 0.5))
    ax_true.set_yticks(range(n_true))
    ax_true.set_yticklabels([f"true {i}: {u['name']}" for i, u in enumerate(true_units)],
                             fontsize=7)
    ax_true.set_title(f"TRUE spike types (n={n_true} units)", fontsize=10)

    for c in range(n_est):
        mask = est_labels == c
        color = (SIM_COLORS[est_match[c] % len(SIM_COLORS)] if est_match[c] >= 0
                 else UNMATCHED_COLOR)
        ax_est.vlines(est_peaks_t[mask], c - 0.4, c + 0.4, color=color, linewidth=1)
    ax_est.set_ylim(-0.5, max(n_est - 0.5, 0.5))
    ax_est.set_yticks(range(n_est))
    ax_est.set_yticklabels(
        [f"est {c} -> true {est_match[c]} ({est_purity[c]:.0%})" if est_match[c] >= 0
         else f"est {c} -> unmatched" for c in range(n_est)], fontsize=7)
    ax_est.set_xlabel("Time (s)")
    ax_est.set_title(f"ESTIMATED spike types (n={n_est} clusters found) -- color = best-"
                      f"matching true unit", fontsize=10)

    conf = np.zeros((max(n_true, 1), max(n_est, 1)))
    for c in range(n_est):
        mu = matched_unit[est_labels == c]
        for i in range(n_true):
            conf[i, c] = int((mu == i).sum())
    im = ax_conf.imshow(conf, aspect="auto", cmap="viridis")
    ax_conf.set_xlabel("Estimated type")
    ax_conf.set_ylabel("True unit")
    ax_conf.set_xticks(range(n_est))
    ax_conf.set_yticks(range(n_true))
    ax_conf.set_title("Confusion matrix (matched spike counts)", fontsize=10)
    fig.colorbar(im, ax=ax_conf, fraction=0.03, pad=0.02)

    fig.suptitle(f"Simulated trace, n={n_muaps} true MUAPs — spike-sort recovery", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    print(f"Saved comparison figure to {out_path}")
    return est_match, est_purity


def plot_true_vs_detected_window(trace, fs, true_units, peaks_t, peak_amps, n_muaps, window_s,
                                  out_path, start_s=0.0):
    """A short (window_s), zoomed-in two-panel figure: top -- the raw
    trace with every TRUE spike marked, color-coded by true unit; bottom
    -- the same stretch of trace with every DETECTED spike marked
    (unclustered, one color). Meant to be eyeballed directly: which true
    spikes visually stick out of the noise/interference enough that the
    detector actually catches them, and which don't. start_s skips past
    any known spike-free lead-in (see simulate_emg_traces.py's
    BASELINE_S) so the window shows actual spiking activity."""
    t = np.arange(len(trace)) / fs
    end_s = start_s + window_s
    win_mask = (t >= start_s) & (t <= end_s)

    fig, (ax_true, ax_det) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax_true.plot(t[win_mask], trace[win_mask], linewidth=0.5, color="0.4", zorder=1)
    for i, u in enumerate(true_units):
        color = SIM_COLORS[i % len(SIM_COLORS)]
        st = u["spike_times"]
        st = st[(st >= start_s) & (st <= end_s)]
        amps = np.interp(st, t, trace)
        ax_true.scatter(st, amps, s=22, color=color, zorder=2,
                         label=f"{u['subject']} type{u['type_idx']}")
    ax_true.invert_yaxis()
    ax_true.set_ylabel("Amplitude")
    ax_true.set_title(f"TRUE spike locations (n={len(true_units)} units)", fontsize=10)
    ax_true.legend(loc="upper right", fontsize=6, ncol=min(len(true_units), 5))

    det_mask = (peaks_t >= start_s) & (peaks_t <= end_s)
    ax_det.plot(t[win_mask], trace[win_mask], linewidth=0.5, color="0.4", zorder=1)
    ax_det.scatter(peaks_t[det_mask], peak_amps[det_mask], s=22, color="crimson", zorder=2,
                   label=f"{int(det_mask.sum())} detected")
    ax_det.invert_yaxis()
    ax_det.set_xlabel("Time (s)")
    ax_det.set_ylabel("Amplitude")
    ax_det.set_title("DETECTED spikes (unclustered)", fontsize=10)
    ax_det.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Simulated trace, n={n_muaps} true MUAPs — true vs. detected spikes "
                 f"({start_s:.0f}-{end_s:.0f} s)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    print(f"Saved true-vs-detected window plot to {out_path}")


def plot_detected_unclustered(trace, fs, peaks_t, peak_amps, n_muaps, out_path, baseline_s=0.0):
    """The raw simulated trace with every DETECTED spike marked in one
    uniform color -- i.e. what the detector alone found, before any
    clustering assigns those detections to types. (Detection happens
    upstream of feature extraction/clustering, so this is the same
    regardless of --feature-method; only generated for the tsne/default
    run to avoid an identical duplicate per method.)"""
    t = np.arange(len(trace)) / fs
    fig, ax = plt.subplots(figsize=(14, 4))
    if baseline_s > 0:
        ax.axvspan(0, baseline_s, color="0.9", zorder=0, label="baseline (spike-free)")
    ax.plot(t, trace, linewidth=0.4, color="0.4", zorder=1)
    ax.scatter(peaks_t, peak_amps, s=14, color="crimson", zorder=2,
               label=f"{len(peaks_t)} detected (unclustered)")
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Simulated trace, n={n_muaps} true MUAPs — detected spikes before "
                 f"clustering", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved detected-unclustered plot to {out_path}")


def square_grid_shape(n):
    n_side = max(1, int(np.ceil(np.sqrt(max(n, 1)))))
    return n_side, n_side


def plot_muap_galleries(true_units, r, est_match, est_purity, n_muaps, out_path,
                         l2_distances=None, l2_threshold_used=None, l2_percentile_marker=None,
                         l2_normalized=False):
    """Two galleries side by side: the SIMULATED (input) MUAP templates
    actually used to build this trace, and the ESTIMATED (recovered)
    MUAP templates spike-sorting found in it -- same grey-trials/black-
    mean panel style as build_muap_library.py's gallery, so the two are
    directly comparable by eye. A third, narrow panel plots every
    pairwise L2 distance between the estimated types' mean waveforms,
    sorted ascending -- a quick visual read on where a merge cutoff
    would fall (see spike_sort.l2_merge); l2_threshold_used marks the
    cutoff actually applied this run (if any), l2_percentile_marker
    optionally marks a candidate percentile (e.g. 20) even when it
    wasn't applied, for comparison."""
    n_true = len(true_units)
    n_est = int(r["labels"].max()) + 1 if r["labels"].size else 0
    t_est_ms = (np.arange(-r["half_win"], r["half_win"]) / r["fs"]) * 1000
    rng = np.random.default_rng(0)

    n_cols_l, _ = square_grid_shape(n_true)
    n_rows_l = int(np.ceil(n_true / n_cols_l)) if n_true else 1
    n_cols_r, _ = square_grid_shape(n_est)
    n_rows_r = int(np.ceil(n_est / n_cols_r)) if n_est else 1
    n_rows = max(n_rows_l, n_rows_r)

    l2_col_width = 1.6
    fig = plt.figure(figsize=(3.2 * (n_cols_l + n_cols_r) + l2_col_width + 1, 3.0 * n_rows + 1))
    subfigs = fig.subfigures(1, 3, wspace=0.05,
                              width_ratios=[n_cols_l, n_cols_r, l2_col_width])

    axes_l = subfigs[0].subplots(n_rows, n_cols_l, squeeze=False,
                                  gridspec_kw=dict(hspace=0.7, wspace=0.35))
    for idx in range(n_rows * n_cols_l):
        ax = axes_l[idx // n_cols_l][idx % n_cols_l]
        if idx >= n_true:
            ax.axis("off")
            continue
        u = true_units[idx]
        color = SIM_COLORS[idx % len(SIM_COLORS)]
        m = muap_metrics(u["template"], u["fs"])
        title = f"true {idx}: {u['subject']} type{u['type_idx']}\nn={len(u['spike_times'])}"
        plot_muap_panel(ax, u["trial_waveforms"], u["template"], m, u["t_wave_ms"], color, title,
                         muap_ylim=None, negative_up=True, rng=rng, mean_color=color,
                         ylabel="Amplitude" if idx % n_cols_l == 0 else None)
    subfigs[0].suptitle(f"SIMULATED (input) MUAPs — n={n_true}", fontsize=11)

    axes_r = subfigs[1].subplots(n_rows, n_cols_r, squeeze=False,
                                  gridspec_kw=dict(hspace=0.7, wspace=0.35))
    for idx in range(n_rows * n_cols_r):
        ax = axes_r[idx // n_cols_r][idx % n_cols_r]
        if idx >= n_est:
            ax.axis("off")
            continue
        mask = r["labels"] == idx
        trials = r["waveforms"][mask]
        mean_wf = trials.mean(axis=0)
        m = muap_metrics(mean_wf, r["fs"])
        match = est_match[idx]
        color = SIM_COLORS[match % len(SIM_COLORS)] if match >= 0 else UNMATCHED_COLOR
        title = (f"est {idx} -> true {match} ({est_purity[idx]:.0%})" if match >= 0
                 else f"est {idx} -> unmatched") + f"\nn={int(mask.sum())}"
        plot_muap_panel(ax, trials, mean_wf, m, t_est_ms, color, title,
                         muap_ylim=None, negative_up=True, rng=rng, mean_color=color,
                         ylabel="Amplitude" if idx % n_cols_r == 0 else None)
    subfigs[1].suptitle(f"ESTIMATED (recovered) MUAPs — n={n_est}", fontsize=11)

    ax_l2 = subfigs[2].subplots(1, 1)
    if l2_distances:
        dist_sorted = np.sort(np.fromiter(l2_distances.values(), dtype=float))
        rank = np.arange(1, len(dist_sorted) + 1)
        ax_l2.plot(rank, dist_sorted, marker="o", markersize=3, linewidth=1, color="0.2")
        if l2_threshold_used:
            ax_l2.axhline(l2_threshold_used, color="crimson", linewidth=1, linestyle="-",
                           label=f"used: {l2_threshold_used:.0f}")
        if l2_percentile_marker is not None:
            marker_val = np.percentile(dist_sorted, l2_percentile_marker)
            ax_l2.axhline(marker_val, color="steelblue", linewidth=1, linestyle="--",
                           label=f"{l2_percentile_marker:g}th pct: {marker_val:.0f}")
        if l2_threshold_used or l2_percentile_marker is not None:
            ax_l2.legend(fontsize=6, loc="upper left")
        ax_l2.set_xlabel("pair rank", fontsize=8)
        ax_l2.set_ylabel("L2 distance (area-normalized)" if l2_normalized
                          else "L2 distance (raw units)", fontsize=8)
        ax_l2.tick_params(labelsize=7)
    else:
        ax_l2.axis("off")
        ax_l2.text(0.5, 0.5, "<2 estimated\ntypes", ha="center", va="center", fontsize=8,
                    transform=ax_l2.transAxes)
    subfigs[2].suptitle("sorted type-pair\nL2 distances", fontsize=9)

    fig.suptitle(f"Simulated trace, n={n_muaps} true MUAPs — input vs. recovered MUAP shapes",
                 fontsize=13, y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved MUAP shape galleries to {out_path}")


def _draw_estimated_gallery(subfig, r, est_match, est_purity, t_est_ms, rng, type_ids, title):
    """One grid of MUAP panels, one per id in `type_ids` (original
    estimated-type indices, not necessarily 0..k-1 or contiguous) --
    factored out of plot_muap_galleries so the same drawing code can
    render both "every estimated type" and a filtered subset (e.g. with
    flagged superposition-composite types removed) identically."""
    n = len(type_ids)
    n_cols, _ = square_grid_shape(n)
    n_rows = int(np.ceil(n / n_cols)) if n else 1
    axes = subfig.subplots(n_rows, n_cols, squeeze=False,
                            gridspec_kw=dict(hspace=0.7, wspace=0.35))
    for pos in range(n_rows * n_cols):
        ax = axes[pos // n_cols][pos % n_cols]
        if pos >= n:
            ax.axis("off")
            continue
        idx = type_ids[pos]
        mask = r["labels"] == idx
        trials = r["waveforms"][mask]
        mean_wf = trials.mean(axis=0)
        m = muap_metrics(mean_wf, r["fs"])
        match = est_match[idx]
        color = SIM_COLORS[match % len(SIM_COLORS)] if match >= 0 else UNMATCHED_COLOR
        panel_title = (f"est {idx} -> true {match} ({est_purity[idx]:.0%})" if match >= 0
                       else f"est {idx} -> unmatched") + f"\nn={int(mask.sum())}"
        plot_muap_panel(ax, trials, mean_wf, m, t_est_ms, color, panel_title,
                         muap_ylim=None, negative_up=True, rng=rng, mean_color=color,
                         ylabel="Amplitude" if pos % n_cols == 0 else None)
    subfig.suptitle(title, fontsize=11)
    return n_cols, n_rows


def plot_muap_galleries_superposition(true_units, r, est_match, est_purity, n_muaps, out_path,
                                       superposition_type_ids):
    """Three galleries side by side: SIMULATED (input) MUAP templates,
    ESTIMATED (every recovered type), and ESTIMATED WITH SUPERPOSITIONS
    REMOVED -- the same estimated types minus whichever ones
    spike_sort.superposition.find_superposition_candidates flagged as
    likely sums of two other estimated types rather than a genuinely
    distinct unit (superposition_type_ids, their type_c values)."""
    n_true = len(true_units)
    n_est = int(r["labels"].max()) + 1 if r["labels"].size else 0
    all_ids = list(range(n_est))
    kept_ids = [c for c in all_ids if c not in superposition_type_ids]
    t_est_ms = (np.arange(-r["half_win"], r["half_win"]) / r["fs"]) * 1000
    rng = np.random.default_rng(0)

    n_cols_l, _ = square_grid_shape(n_true)
    n_rows_l = int(np.ceil(n_true / n_cols_l)) if n_true else 1
    n_cols_e, _ = square_grid_shape(n_est)
    n_rows_e = int(np.ceil(n_est / n_cols_e)) if n_est else 1
    n_cols_k, _ = square_grid_shape(len(kept_ids))
    n_rows_k = int(np.ceil(len(kept_ids) / n_cols_k)) if kept_ids else 1
    n_rows = max(n_rows_l, n_rows_e, n_rows_k)

    fig = plt.figure(figsize=(3.2 * (n_cols_l + n_cols_e + n_cols_k) + 1, 3.0 * n_rows + 1))
    subfigs = fig.subfigures(1, 3, wspace=0.04,
                              width_ratios=[n_cols_l, n_cols_e, max(n_cols_k, 1)])

    axes_l = subfigs[0].subplots(n_rows_l, n_cols_l, squeeze=False,
                                  gridspec_kw=dict(hspace=0.7, wspace=0.35))
    for idx in range(n_rows_l * n_cols_l):
        ax = axes_l[idx // n_cols_l][idx % n_cols_l]
        if idx >= n_true:
            ax.axis("off")
            continue
        u = true_units[idx]
        color = SIM_COLORS[idx % len(SIM_COLORS)]
        m = muap_metrics(u["template"], u["fs"])
        title = f"true {idx}: {u['subject']} type{u['type_idx']}\nn={len(u['spike_times'])}"
        plot_muap_panel(ax, u["trial_waveforms"], u["template"], m, u["t_wave_ms"], color, title,
                         muap_ylim=None, negative_up=True, rng=rng, mean_color=color,
                         ylabel="Amplitude" if idx % n_cols_l == 0 else None)
    subfigs[0].suptitle(f"SIMULATED (input) MUAPs — n={n_true}", fontsize=11)

    _draw_estimated_gallery(subfigs[1], r, est_match, est_purity, t_est_ms, rng, all_ids,
                             f"ESTIMATED (all recovered) — n={n_est}")
    _draw_estimated_gallery(subfigs[2], r, est_match, est_purity, t_est_ms, rng, kept_ids,
                             f"ESTIMATED, superpositions removed — n={len(kept_ids)} "
                             f"({len(superposition_type_ids)} removed)")

    fig.suptitle(f"Simulated trace, n={n_muaps} true MUAPs — simulated vs. estimated vs. "
                 f"superposition-filtered", fontsize=13, y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved simulated/estimated/superposition-removed galleries to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feature-method", default="tsne", choices=["tsne", "pca", "wavelet"],
                         help="Shape-feature method fed to the clusterer")
    parser.add_argument("--n-components", type=int, default=2,
                         help="Number of shape-feature dimensions (PCA components / t-SNE "
                              "embedding dims / top wavelet coefficients)")
    parser.add_argument("--cluster-method", default="kmeans",
                         choices=["kmeans", "gmm", "tmixture", "hdbscan", "spc"],
                         help="Clustering algorithm run on the shape features: kmeans, gmm "
                              "(Gaussian mixture), tmixture (Student's-t mixture -- heavier "
                              "tails, more robust to superimposed/outlier waveforms), hdbscan "
                              "(density-based) or spc (superparamagnetic). kmeans/gmm/tmixture "
                              "take their cluster count from the silhouette sweep; hdbscan/spc "
                              "pick their own count and may label some spikes as noise (dropped "
                              "before matching). Non-kmeans runs get a '_<method>' filename "
                              "suffix.")
    parser.add_argument("--only-n", type=int, default=None,
                         help="Only process the simulated trace with this many true MUAPs (one "
                              "of N_MUAPS_LIST) instead of all of them -- handy for tuning "
                              "hdbscan/spc params quickly.")
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=None,
                         help="hdbscan only: smallest group of spikes that counts as its own "
                              "cluster (BASE_PARAMS default 5). On a dense t-SNE cloud the "
                              "default over-fragments badly -- raise it (e.g. 50-150) to get "
                              "MUAP-count-scale clusters.")
    parser.add_argument("--hdbscan-min-samples", type=int, default=None,
                         help="hdbscan only: core-point neighbourhood size / noise aggressiveness "
                              "(BASE_PARAMS default = min_cluster_size). Higher = more points "
                              "called noise, fewer/tighter clusters.")
    parser.add_argument("--spc-t-min", type=float, default=None,
                         help="spc only: lowest temperature in the sweep (BASE_PARAMS default "
                              "0.05).")
    parser.add_argument("--spc-t-max", type=float, default=None,
                         help="spc only: highest temperature in the sweep (BASE_PARAMS default "
                              "1.5). If no superparamagnetic transition is found in "
                              "[t_min, t_max] every spike ends up noise and the run aborts -- "
                              "lower t_max and/or add temps.")
    parser.add_argument("--spc-n-temps", type=int, default=None,
                         help="spc only: number of temperatures sampled between t_min and t_max "
                              "(BASE_PARAMS default 20).")
    parser.add_argument("--spc-min-clus-frac", type=float, default=None,
                         help="spc only: the second-largest cluster must reach this fraction of "
                              "all spikes for a temperature to count as the transition "
                              "(BASE_PARAMS default 0.1). Lower it to accept a finer split.")
    parser.add_argument("--spc-knn", type=int, default=None,
                         help="spc only: nearest-neighbour count for the Potts interaction graph "
                              "(BASE_PARAMS default 10).")
    parser.add_argument("--spc-mc-steps", type=int, default=None,
                         help="spc only: Monte-Carlo (Wolff) steps per temperature (BASE_PARAMS "
                              "default 100). Fewer = faster but noisier correlation estimates.")
    parser.add_argument("--threshold-mad", type=float, default=DETECTION_THRESHOLD_MAD,
                         help="MAD multiplier for the spike-detection threshold -- lower "
                              "catches smaller/near-noise spikes too (default: "
                              f"{DETECTION_THRESHOLD_MAD})")
    parser.add_argument("--tsne-perplexity", type=float, default=None,
                         help="t-SNE perplexity (roughly, effective neighbors per point) -- "
                              "only used when --feature-method=tsne; lower finds tighter/more "
                              "local sub-clusters (more fragmentation risk), higher smooths "
                              "toward global structure (default: BASE_PARAMS' 30.0)")
    parser.add_argument("--l2-merge-threshold", type=float, default=None,
                         help="Post-clustering shape merge: types whose averaged (mean) "
                              "waveforms are within this raw, unnormalized L2 distance of each "
                              "other are forced together (see spike_sort.l2_merge). No default "
                              "-- disabled unless given; inspect "
                              "spike_sort.l2_merge.pairwise_l2_distances on a run first to pick "
                              "a sensible value for this trace's amplitude scale.")
    parser.add_argument("--l2-merge-percentile", type=float, default=None,
                         help="Alternative to --l2-merge-threshold: merge the closest this-%% "
                              "of type pairs by L2 distance (e.g. 20 -> merge the smallest 20%% "
                              "of pairwise distances), recomputed per trace from its own "
                              "distance distribution. Ignored if --l2-merge-threshold is given. "
                              "CAUTION: validated against ground truth and found to over-merge "
                              "badly as type count grows -- prefer --l2-merge-outlier-fence-k.")
    parser.add_argument("--l2-merge-elbow", action="store_true",
                         help="Alternative to both of the above: pick the threshold "
                              "automatically at the largest gap in this trace's own sorted "
                              "pairwise-distance curve (see spike_sort.l2_merge.elbow_threshold). "
                              "CAUTION: validated against ground truth and found to fail badly "
                              "(collapsed almost everything into one type on 3 of 4 test traces) "
                              "-- prefer --l2-merge-outlier-fence-k. Ignored if "
                              "--l2-merge-threshold or --l2-merge-percentile is given.")
    parser.add_argument("--l2-merge-outlier-fence-k", type=float, default=None,
                         help="Best-validated automatic option: median(distances) - k * "
                              "MAD(distances), a robust low-outlier fence (see "
                              "spike_sort.l2_merge.outlier_fence_threshold) -- matches this "
                              "problem's actual shape (a few genuine duplicates as low outliers, "
                              "not a bimodal split) much better than the two alternatives above. "
                              "k=2.0 validated well. Ignored if --l2-merge-threshold, "
                              "--l2-merge-percentile, or --l2-merge-elbow is given.")
    parser.add_argument("--l2-merge-normalize-area", action="store_true",
                         help="Scale each type's mean waveform to unit area (sum(|waveform|) = "
                              "1) before computing L2 distance, instead of raw units (see "
                              "spike_sort.l2_merge.normalize_area) -- compares SHAPE only, "
                              "blind to amplitude. Applies to whichever threshold method above "
                              "is selected. EXPLORATORY: mixed result on the simulated traces "
                              "(roughly equivalent at low unit counts, more aggressive but "
                              "roughly purity-neutral at higher counts) -- not a validated "
                              "improvement, try deliberately.")
    args = parser.parse_args()
    # Keep the original (tsne, default) output filenames stable for backward compatibility;
    # any other configuration gets its own suffixed filenames so runs don't clobber each other.
    suffix = "" if (args.feature_method, args.n_components) == ("tsne", 2) \
        else f"_{args.feature_method}{args.n_components}"
    if args.cluster_method != "kmeans":
        suffix += f"_{args.cluster_method}"
    if args.tsne_perplexity is not None:
        suffix += f"_perp{args.tsne_perplexity:g}"
    if args.l2_merge_threshold is not None:
        suffix += f"_l2m{args.l2_merge_threshold:g}"
    elif args.l2_merge_percentile is not None:
        suffix += f"_l2p{args.l2_merge_percentile:g}"
    elif args.l2_merge_elbow:
        suffix += "_l2elbow"
    elif args.l2_merge_outlier_fence_k is not None:
        suffix += f"_l2fence{args.l2_merge_outlier_fence_k:g}"
    if args.l2_merge_normalize_area:
        suffix += "_areanorm"

    # hdbscan/spc knob overrides -> {param_name: value}, plus a filename tag for any that differ
    # from BASE_PARAMS.
    param_overrides = {}
    for arg_name, param_name, tag in [
        ("hdbscan_min_cluster_size", "hdbscan_min_cluster_size", "mcs"),
        ("hdbscan_min_samples", "hdbscan_min_samples", "ms"),
        ("spc_t_min", "spc_t_min", "tmin"),
        ("spc_t_max", "spc_t_max", "tmax"),
        ("spc_n_temps", "spc_n_temps", "nt"),
        ("spc_min_clus_frac", "spc_min_clus_frac", "mcf"),
        ("spc_knn", "spc_knn", "knn"),
        ("spc_mc_steps", "spc_mc_steps", "mc"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            param_overrides[param_name] = value
            suffix += f"_{tag}{value:g}"

    n_muaps_list = ([args.only_n] if args.only_n is not None else N_MUAPS_LIST)

    prev_cwd = os.getcwd()
    manifest_rows = []
    library_lookup = load_library_lookup()

    for n_muaps in n_muaps_list:
        npz_path = os.path.join(OUT_DIR, f"simulated_trace_n{n_muaps}.npz")
        trace, fs, true_units, baseline_s = load_simulated_trace(npz_path, library_lookup)
        print(f"n={n_muaps}: loaded {len(true_units)} true unit(s) from {npz_path} "
              f"(baseline={baseline_s:.1f}s)")

        filename = f"EMGRUN__simN{n_muaps}.txt"
        write_emgrun_file(trace, fs, SPIKESORT_INPUT_DIR, filename)

        noise_baseline_samples = int(round(baseline_s * fs)) if baseline_s > 0 else None
        os.chdir(SPIKESORT_INPUT_DIR)
        try:
            r = sort_simulated_trace(trace, fs, filename, label=f"sim_n{n_muaps}",
                                      feature_method=args.feature_method,
                                      n_components=args.n_components,
                                      cluster_method=args.cluster_method,
                                      threshold_mad=args.threshold_mad,
                                      noise_baseline_samples=noise_baseline_samples,
                                      tsne_perplexity=args.tsne_perplexity,
                                      l2_merge_threshold=args.l2_merge_threshold,
                                      l2_merge_percentile=args.l2_merge_percentile,
                                      l2_merge_elbow=args.l2_merge_elbow,
                                      l2_merge_outlier_fence_k=args.l2_merge_outlier_fence_k,
                                      l2_merge_normalize_area=args.l2_merge_normalize_area,
                                      param_overrides=param_overrides)
        finally:
            os.chdir(prev_cwd)
        print(f"  spike-sort ({args.feature_method}, {args.n_components} comp, "
              f"{args.cluster_method}): {r['n_spikes']} spikes -> {r['n_types']} estimated type(s)"
              + (f" ({r['n_types_l2_merged']} types L2-merged)" if r["n_types_l2_merged"] else ""))

        matched_unit = match_to_ground_truth(r["peaks_t"], true_units, MATCH_TOL_MS / 1000)
        est_match, est_purity = compute_match_purity(true_units, r["labels"], matched_unit)

        # Superposition cleanup: flag estimated types that are really the sum of two other
        # estimated types (their spikes consistently superimposing) rather than a distinct unit.
        n_est = int(r["labels"].max()) + 1 if r["labels"].size else 0
        templates = np.stack([r["waveforms"][r["labels"] == c].mean(axis=0)
                               for c in range(n_est)]) if n_est else np.empty((0, 0))
        max_shift = int(round(SUPERPOSITION_MAX_SHIFT_MS * 1e-3 * fs))
        superposition_candidates = (find_superposition_candidates(templates, max_shift)
                                     if n_est >= 3 else [])
        superposition_type_ids = {cand["type_c"] for cand in superposition_candidates}
        print("  superposition check: " + (format_superposition_report(
            superposition_candidates, fs=fs).replace("\n", "\n    ")
            if superposition_candidates else "no candidates"))

        gallery_out_path = os.path.join(OUT_DIR, f"spikesort_muap_shapes_n{n_muaps}{suffix}.png")
        plot_muap_galleries(true_units, r, est_match, est_purity, n_muaps, gallery_out_path,
                             l2_distances=r["l2_distances"], l2_threshold_used=r["l2_threshold_used"],
                             l2_percentile_marker=20, l2_normalized=args.l2_merge_normalize_area)

        # "Actual spikes vs estimated spikes": TRUE spike-type raster over ESTIMATED spike-type
        # raster + matched-count confusion matrix.
        compare_out_path = os.path.join(OUT_DIR, f"spikesort_true_vs_est_n{n_muaps}{suffix}.png")
        plot_comparison(true_units, r["labels"], r["peaks_t"], matched_unit, n_muaps,
                         compare_out_path)

        # Same MUAP galleries, but with the flagged superposition-composite types removed.
        superposition_out_path = os.path.join(
            OUT_DIR, f"spikesort_muap_shapes_superposition_n{n_muaps}{suffix}.png")
        plot_muap_galleries_superposition(true_units, r, est_match, est_purity, n_muaps,
                                           superposition_out_path, superposition_type_ids)
        plt.show()

        for c in range(r["n_types"]):
            n_c = int((r["labels"] == c).sum())
            cand = next((x for x in superposition_candidates if x["type_c"] == c), None)
            manifest_rows.append(dict(
                n_muaps_in_trace=n_muaps, feature_method=args.feature_method,
                n_components=args.n_components, cluster_method=args.cluster_method,
                estimated_type=c, n_spikes=n_c,
                matched_true_unit=est_match[c], purity=round(est_purity[c], 3),
                is_superposition=cand is not None,
                superposition_of=(f"{cand['type_i']}+{cand['type_j']}" if cand else ""),
                superposition_pair_residual=(round(cand["pair_residual_frac"], 3) if cand else ""),
            ))

    if not manifest_rows:
        print("No estimated types on any trace -- no manifest written.")
        return
    manifest_path = os.path.join(OUT_DIR, f"spikesort_compare_manifest{suffix}.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
