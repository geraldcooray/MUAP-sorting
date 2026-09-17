#!/usr/bin/env python3
"""Compare MUAP sorting with per-run vs all-runs-together t-SNE, after a
DCT high-pass of every extracted waveform.

Pipeline (shared prefix)
------------------------
1. DETECTION + WAVEFORM EXTRACTION -- per run, unchanged
   (spike_sort.detection.detect_spikes / extract_waveforms).
2. DCT FILTER -- every extracted waveform is DCT-II transformed, its
   `DCT_N_LOW` lowest coefficients (DC + slowest cosines = residual
   baseline offset / slow drift) are zeroed, and it is inverse
   transformed (spike_sort.features.dct_highpass_waveforms). All
   clustering AND mean-template averaging below use these filtered
   waveforms.

Then the same DCT-filtered waveforms are sorted two ways:

  A. PER-RUN t-SNE   -- one t-SNE embedding + one k-means labelling per
                        run, in isolation (like multi_run_spike_sort.py).
  B. JOINT t-SNE     -- every run's waveforms pooled, ONE t-SNE
                        embedding + ONE k-means labelling (like
                        joint_run_spike_sort.py).

Outputs
-------
  dct_muap_per_run_tsne.png  -- MUAP panel per (run, per-run type)
  dct_muap_joint_tsne.png    -- MUAP panel per joint (global) type, trials
                                pooled across runs
  dct_muap_tsne_compare.png  -- the two galleries side by side, one figure

Run from a directory holding EMGRUN__*.txt + Header.txt:
    python /path/to/python/dct_muap_compare.py
Edit the config constants below directly -- harness script, not a CLI.
"""
import copy
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score

from call_spike_sort_emg import BASE_PARAMS
from joint_run_spike_sort import run_detection
from spike_sort.clustering import cluster_spikes, relabel_dense
from spike_sort.features import build_features, dct_highpass_waveforms
from spike_sort.metrics import muap_metrics
from spike_sort.plotting import CLUSTER_COLORS, plot_muap_panel

# ---------------------------------------------------------------------------
RUN_FILES = sorted(glob.glob("EMGRUN__*.txt"))

DCT_N_LOW = 3                      # lowest DCT coefficients removed from each waveform

PARAM_OVERRIDES = dict(
    threshold_mad=2.0,
    feature_method="tsne",
    cluster_method="kmeans",
    isi_weight=0.0,
    polarity_weight=0.0,
    amp_dur_weight=0.0,
    n_components=2,
)

MAX_RATE_HZ = 10.0                 # firing-rate floor -> minimum cluster count n0
SIL_SWEEP = True                   # silhouette-sweep the cluster count
SIL_MAX_N = 12
SIL_SAMPLE = 4000                  # silhouette is O(n^2); score on at most this many spikes
MIN_SPIKES_PER_TYPE = 50           # drop types below this many (pooled, for joint) detections

MUAP_YLIM = None                   # None = autoscale each panel to its mean template
NEGATIVE_UP = True

OUT_PER_RUN = "dct_muap_per_run_tsne.png"
OUT_JOINT = "dct_muap_joint_tsne.png"
OUT_COMPARE = "dct_muap_tsne_compare.png"
# ---------------------------------------------------------------------------


def choose_k(features, n0, seed, method):
    """Silhouette-maximizing cluster count in [max(n0,2), SIL_MAX_N],
    scored on a random subsample. Falls back to max(n0,1)."""
    if not SIL_SWEEP:
        return max(n0, 1)
    n = features.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, SIL_SAMPLE, replace=False) if n > SIL_SAMPLE else np.arange(n)
    best_k, best_s = max(n0, 1), -np.inf
    for k in range(max(n0, 2), min(SIL_MAX_N, n - 1) + 1):
        labels = cluster_spikes(features, k, seed, method=method)
        sub = labels[idx]
        if len(set(sub.tolist())) < 2:
            continue
        s = silhouette_score(features[idx], sub)
        if s > best_s:
            best_k, best_s = k, s
    return best_k


def sort_one(waveforms, fs, duration, params, seed):
    """t-SNE features + k-means on one waveform pool (already DCT-filtered).
    Returns dense labels (0..k-1) and the cluster count used."""
    features = build_features(
        waveforms, fs, params["n_components"], seed, params["amp_dur_weight"],
        method=params["feature_method"], wavelet=params["wavelet"],
        wavelet_level=params["wavelet_level"], tsne_perplexity=params["tsne_perplexity"],
        wavelet_clip_percentile=params["wavelet_clip_percentile"],
        polarity_weight=params["polarity_weight"])
    n0 = max(1, int(np.ceil(waveforms.shape[0] / (MAX_RATE_HZ * duration))))
    k = choose_k(features, n0, seed, params["cluster_method"])
    labels = cluster_spikes(features, k, seed, method=params["cluster_method"])
    labels, n_types = relabel_dense(labels)
    return labels, n_types, n0, k


def keep_big_types(labels, min_spikes):
    """Relabel types with < min_spikes members to -1 and renumber the
    survivors 0..k-1. Returns (labels, n_types)."""
    counts = np.bincount(labels[labels >= 0], minlength=labels.max() + 1) \
        if labels.max() >= 0 else np.array([])
    keep = [c for c in range(len(counts)) if counts[c] >= min_spikes]
    remap = {old: new for new, old in enumerate(keep)}
    return np.array([remap.get(l, -1) for l in labels]), len(keep)


def square_grid(n):
    s = max(1, int(np.ceil(np.sqrt(max(n, 1)))))
    return s, s


def grid_for(n, ncols):
    """(nrows, ncols) holding n panels at a fixed column count."""
    return max(1, int(np.ceil(n / ncols))), ncols


# ---------------------------------------------------------------------------
# A. per-run t-SNE
# ---------------------------------------------------------------------------
def sort_per_run(runs, params):
    out = []
    for r in runs:
        wf = r["waveforms"]
        if wf.shape[0] < 3:
            print(f"  {r['label']}: only {wf.shape[0]} spikes -- skipped")
            out.append(dict(r, labels=np.full(wf.shape[0], -1), n_types=0))
            continue
        labels, n_types, n0, k = sort_one(wf, r["fs"], r["analysis_duration"],
                                          params, params["seed"])
        labels, n_types = keep_big_types(labels, MIN_SPIKES_PER_TYPE)
        print(f"  {r['label']}: {wf.shape[0]} spikes -> {n_types} type(s) kept "
              f"(n0={n0}, k={k})")
        out.append(dict(r, labels=labels, n_types=n_types))
    return out


# ---------------------------------------------------------------------------
# B. joint t-SNE
# ---------------------------------------------------------------------------
def sort_joint(runs, params):
    wf = np.vstack([r["waveforms"] for r in runs])
    run_id = np.concatenate([np.full(r["waveforms"].shape[0], i) for i, r in enumerate(runs)])
    peaks_t = np.concatenate([r["peaks_t"] for r in runs])
    total_dur = float(sum(r["analysis_duration"] for r in runs))
    print(f"  pooled {wf.shape[0]} spikes from {len(runs)} runs -- one t-SNE embedding...")
    labels, n_types, n0, k = sort_one(wf, runs[0]["fs"], total_dur, params, params["seed"])
    labels, n_types = keep_big_types(labels, MIN_SPIKES_PER_TYPE)
    print(f"  joint: {wf.shape[0]} spikes -> {n_types} global type(s) kept (n0={n0}, k={k})")
    return dict(waveforms=wf, run_id=run_id, peaks_t=peaks_t, labels=labels,
               n_types=n_types, fs=runs[0]["fs"], half_win=runs[0]["half_win"],
               total_dur=total_dur, run_labels=[r["label"] for r in runs])


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def per_run_panels(per_run):
    return [(r, c) for r in per_run for c in range(r["n_types"])]


def draw_per_run_gallery(per_run, axes, nrows, ncols):
    panels = per_run_panels(per_run)
    rng = np.random.default_rng(0)
    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        if idx >= len(panels):
            ax.axis("off")
            continue
        r, c = panels[idx]
        mask = r["labels"] == c
        wf = r["waveforms"][mask]
        mean_wf = wf.mean(axis=0)
        m = muap_metrics(mean_wf, r["fs"])
        t_ms = (np.arange(-r["half_win"], r["half_win"]) / r["fs"]) * 1000
        fr = mask.sum() / r["analysis_duration"]
        plot_muap_panel(ax, wf, mean_wf, m, t_ms,
                        CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                        f"{r['label']} type {c}\n(n={int(mask.sum())}, {fr:.1f} Hz, "
                        f"dur {m['duration_ms']:.2f} ms)",
                        muap_ylim=MUAP_YLIM, negative_up=NEGATIVE_UP, rng=rng,
                        ylabel="Amplitude" if idx % ncols == 0 else None)


def draw_joint_gallery(joint, axes, nrows, ncols):
    rng = np.random.default_rng(0)
    t_ms = (np.arange(-joint["half_win"], joint["half_win"]) / joint["fs"]) * 1000
    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        if idx >= joint["n_types"]:
            ax.axis("off")
            continue
        mask = joint["labels"] == idx
        wf = joint["waveforms"][mask]
        mean_wf = wf.mean(axis=0)
        m = muap_metrics(mean_wf, joint["fs"])
        n_runs_present = len(np.unique(joint["run_id"][mask]))
        fr = mask.sum() / joint["total_dur"]
        plot_muap_panel(ax, wf, mean_wf, m, t_ms,
                        CLUSTER_COLORS[idx % len(CLUSTER_COLORS)],
                        f"type {idx} (n={int(mask.sum())} pooled, {n_runs_present} run(s), "
                        f"{fr:.1f} Hz)\ndur {m['duration_ms']:.2f} ms, amp {m['amplitude']:.0f}",
                        muap_ylim=MUAP_YLIM, negative_up=NEGATIVE_UP, rng=rng,
                        ylabel="Amplitude" if idx % ncols == 0 else None)


def save_per_run_gallery(per_run, out_path):
    n = len(per_run_panels(per_run))
    nrows, ncols = square_grid(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows),
                             squeeze=False)
    draw_per_run_gallery(per_run, axes, nrows, ncols)
    fig.suptitle(f"Per-run t-SNE on DCT-filtered waveforms (drop {DCT_N_LOW} lowest DCT "
                 f"coeffs) — each run embedded/clustered alone; types NOT matched across runs",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def save_joint_gallery(joint, out_path):
    nrows, ncols = square_grid(joint["n_types"])
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows),
                             squeeze=False)
    draw_joint_gallery(joint, axes, nrows, ncols)
    fig.suptitle(f"Joint t-SNE on DCT-filtered waveforms (drop {DCT_N_LOW} lowest DCT "
                 f"coeffs) — every run pooled into ONE embedding; a type is the same across runs",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def save_compare(per_run, joint, out_path):
    n_left = max(1, len(per_run_panels(per_run)))
    n_right = max(1, joint["n_types"])
    cols_l = square_grid(n_left)[1]
    cols_r = square_grid(n_right)[1]
    rows = max(grid_for(n_left, cols_l)[0], grid_for(n_right, cols_r)[0])

    fig = plt.figure(figsize=(3.4 * (cols_l + cols_r) + 1.5, 3.2 * rows + 1.4))
    subfigs = fig.subfigures(1, 2, wspace=0.04, width_ratios=[cols_l, cols_r])

    gk = dict(hspace=0.95, wspace=0.4, top=0.93)
    axes_l = subfigs[0].subplots(rows, cols_l, squeeze=False, gridspec_kw=gk)
    draw_per_run_gallery(per_run, axes_l, rows, cols_l)
    subfigs[0].suptitle("A. per-run t-SNE (each run embedded/clustered alone)",
                        fontsize=11, y=0.955)

    axes_r = subfigs[1].subplots(rows, cols_r, squeeze=False, gridspec_kw=gk)
    draw_joint_gallery(joint, axes_r, rows, cols_r)
    subfigs[1].suptitle("B. joint t-SNE (all runs pooled into one embedding)",
                        fontsize=11, y=0.955)

    fig.suptitle(f"MUAP templates: per-run vs joint t-SNE, DCT-filtered waveforms "
                 f"(lowest {DCT_N_LOW} DCT coefficients removed)", fontsize=12, y=0.995)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    params = copy.deepcopy(BASE_PARAMS)
    params.update(PARAM_OVERRIDES)
    params["dct_filter_n"] = DCT_N_LOW

    print(f"Detecting spikes in {len(RUN_FILES)} run(s)...")
    runs = run_detection(RUN_FILES, params)

    for r in runs:                       # DCT high-pass every run's waveforms in place
        r["waveforms"] = dct_highpass_waveforms(r["waveforms"], DCT_N_LOW)
    print(f"DCT high-pass applied: zeroed the {DCT_N_LOW} lowest coefficients per waveform")

    print("A. per-run t-SNE:")
    per_run = sort_per_run(runs, params)
    print("B. joint t-SNE:")
    joint = sort_joint(runs, params)

    save_per_run_gallery(per_run, OUT_PER_RUN)
    save_joint_gallery(joint, OUT_JOINT)
    save_compare(per_run, joint, OUT_COMPARE)
    plt.show()


if __name__ == "__main__":
    main()
