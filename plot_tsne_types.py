#!/usr/bin/env python3
"""Show the 2D t-SNE feature space a sort run's clustering actually used,
with every detected spike FIXED at one position and only its COLOUR
(classification) changing across panels:

  1. Original t-SNE  -- coloured by the raw k-means cluster.
  2. After post-processing  -- min-ISI cleanup (removed spikes greyed,
     NOT moved) + latency-linkage merge (two clusters that fire together
     at a fixed short latency get the same colour; the resulting patch
     can legitimately be two disconnected blobs -- that is expected).
  3. After the split-refinement 2nd stage  -- a type re-split into two.

The embedding is computed exactly ONCE, from build_features with the
pipeline's own seed (42) and perplexity (30) on the full set of detected
+/-5 ms waveforms -- i.e. the standardized t-SNE columns k-means was
literally fit on (amp_dur_weight = polarity_weight = 0 zero out the other
feature columns). t-SNE is never re-run per panel or per stage, so no
point ever moves; a seeded t-SNE on the identical waveform input
reproduces the layout the original run clustered on.

This replicates the detection -> feature-extraction -> silhouette-swept
clustering -> min-ISI -> latency-linkage part of
call_spike_sort_emg.run_pipeline by hand (deterministic) only to recover
the RAW pre-post-processing labels, which run_pipeline does not return.
It never re-simulates the trace and never overwrites a sort_*.npz.

Usage:
    python plot_tsne_types.py --n 10 --method kmeans --postproc full
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import call_spike_sort_emg as cse
import muap_split_refinement as msr
import simulate_and_sort_curated as base  # noqa: F401  (import monkeypatches cse.silhouette_sweep)
import spikesort_simulated_traces as sst
from call_spike_sort_emg import BASE_PARAMS
from spike_sort.clustering import cluster_spikes, relabel_dense
from spike_sort.detection import bandpass, detect_spikes, extract_waveforms
from spike_sort.features import build_features
from spike_sort.io import load_signal
from spike_sort.isi_cleanup import enforce_min_isi_per_type
from spike_sort.latency_linkage import find_linked_type_pairs, group_linked_types, merge_linked_types
from spike_sort.superposition import find_superposition_candidates

SUPERPOSITION_MAX_SHIFT_MS = 3.0  # matches simulate_and_sort_curated / spikesort_simulated_traces

DETECTION_HIGHPASS_HZ = 100.0  # sort_simulated_trace's override for simulated traces


def cluster_from_scratch(trace, fs, baseline_s, method, overrides):
    """Detection -> +/-5 ms extraction -> build_features (t-SNE, seed 42)
    -> silhouette-swept k-means, exactly as run_pipeline does. Returns
    (peaks_t, peak_amps, waveforms, features_2d, raw_labels).

    Uses cse.silhouette_sweep -- the module-level name that importing
    simulate_and_sort_curated monkeypatches to a subsampled version --
    so the swept n_clusters matches the original run rather than the
    slow unpatched O(n^2) sweep."""
    params = dict(BASE_PARAMS)
    params.update(overrides)
    noise_baseline_samples = int(round(baseline_s * fs)) if baseline_s > 0 else None

    peaks, _, _, _ = detect_spikes(
        trace, fs, DETECTION_HIGHPASS_HZ, params["threshold_mad"], params["refractory_ms"],
        params["smoothing_ms"], noise_baseline_samples=noise_baseline_samples)
    measured = bandpass(trace, fs, params["measure_band"][0], params["measure_band"][1])
    peaks, waveforms, _ = extract_waveforms(measured, peaks, fs, params["window_ms"],
                                             params["baseline_window_ms"])
    t = np.arange(trace.size) / fs
    peaks_t, peak_amps = t[peaks], trace[peaks]
    analysis_duration = trace.size / fs

    features = build_features(waveforms, fs, 2, params["seed"], params["amp_dur_weight"],
                               method="tsne", wavelet=params["wavelet"],
                               wavelet_level=params["wavelet_level"],
                               tsne_perplexity=params["tsne_perplexity"],
                               wavelet_clip_percentile=params["wavelet_clip_percentile"],
                               polarity_weight=params["polarity_weight"])
    features_2d = features[:, :2]

    n0 = max(1, int(np.ceil(peaks.size / (params["max_rate_hz"] * analysis_duration))))
    scores = cse.silhouette_sweep(features, range(max(n0, 2), cse.SILHOUETTE_MAX_N + 1),
                                   params["seed"], method=method)
    valid = {k: v for k, v in scores.items() if v == v}
    n_clusters = max(valid, key=valid.get) if valid else max(n0, 1)

    raw_labels, _ = relabel_dense(cluster_spikes(features, n_clusters, params["seed"], method=method))
    return peaks_t, peak_amps, waveforms, features_2d, raw_labels


def post_process(peaks_t, peak_amps, waveforms, labels, fs):
    """run_pipeline's 'full' path + the superposition-drop step
    simulate_and_sort_curated applies after it (up to, not incl., the
    split-refinement L2 merge): min-ISI cleanup -> latency-linkage merge
    -> drop superposition-composite types. Returns (keep_mask,
    final_labels); keep_mask indexes the input order."""
    n = len(labels)
    keep = np.ones(n, dtype=bool)
    lab = labels.copy()

    def drop(mask_local):
        keep[np.where(keep)[0][~mask_local]] = False

    k1 = enforce_min_isi_per_type(peaks_t, lab, peak_amps, 10.0 * 1e-3)
    drop(k1)
    pt, pa, lab = peaks_t[k1], peak_amps[k1], lab[k1]

    linked = find_linked_type_pairs(pt, lab, max_latency_ms=8.0, min_co_occurrence=0.6,
                                     max_latency_std_ms=1.0)
    groups = group_linked_types(linked)
    if groups:
        merged, k2 = merge_linked_types(pt, lab, pa, groups, 8.0)
        drop(k2)
        pt, pa, lab = pt[k2], pa[k2], merged[k2]
        print(f"  latency-linkage merge groups (raw k-means ids): {groups}")
    else:
        print("  latency-linkage merge: no linked pairs")
    lab, n_types = relabel_dense(lab)

    if n_types >= 3:
        wf = waveforms[keep]
        templates = np.stack([wf[lab == c].mean(axis=0) for c in range(n_types)])
        max_shift = int(round(SUPERPOSITION_MAX_SHIFT_MS * 1e-3 * fs))
        dropped = sorted({c["type_c"] for c in find_superposition_candidates(templates, max_shift)})
        if dropped:
            k3 = ~np.isin(lab, dropped)
            drop(k3)
            lab, _ = relabel_dense(lab[k3])
            print(f"  superposition-drop: types {dropped} removed")

    return keep, lab


def scatter_panel(ax, embedding, labels_full, title):
    """labels_full: length == embedding rows; -1 = removed (drawn grey,
    at its original position). Every point is plotted; nothing moves."""
    cmap = plt.get_cmap("tab20")
    removed = labels_full < 0
    if removed.any():
        ax.scatter(embedding[removed, 0], embedding[removed, 1], s=4, alpha=0.25,
                   color="0.7", linewidths=0)
    n_types = int(labels_full.max()) + 1 if (labels_full >= 0).any() else 0
    for c in range(n_types):
        m = labels_full == c
        if not m.any():
            continue
        color = cmap(c % 20)
        ax.scatter(embedding[m, 0], embedding[m, 1], s=5, alpha=0.55, color=color, linewidths=0)
        cx, cy = embedding[m, 0].mean(), embedding[m, 1].mean()
        ax.annotate(str(c), (cx, cy), fontsize=9, fontweight="bold", color="black",
                    ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.15", fc=color, ec="black", lw=0.6, alpha=0.9))
    n_removed = int(removed.sum())
    kept = int((labels_full >= 0).sum())
    ax.set_title(f"{title}\n({n_types} types, {kept} spikes"
                 + (f", {n_removed} greyed" if n_removed else "") + ")", fontsize=11)
    ax.set_xlabel("t-SNE dim 1 (standardized)")
    ax.set_ylabel("t-SNE dim 2 (standardized)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--method", default="kmeans", choices=["kmeans", "gmm", "tmixture"])
    ap.add_argument("--postproc", default="full", choices=["full", "reduced"])
    args = ap.parse_args()

    _, lookup = base.load_curated_library()
    _, fs, true_units, baseline_s = sst.load_simulated_trace(
        os.path.join(base.TRACE_DIR, f"simulated_trace_n{args.n}.npz"), lookup)
    # Detect on the SAME signal the pipeline saw: the EMGRUN text file it wrote/read
    # (np.savetxt with %.6f), not the raw in-memory float64 -- that ~1e-6 rounding shifts a
    # few near-threshold detections and would otherwise change the t-SNE input and the
    # silhouette-swept k.
    emg_txt = os.path.join(sst.SPIKESORT_INPUT_DIR, f"EMGRUN__curated_n{args.n}.txt")
    trace = load_signal(emg_txt, fs, 0.0, None)
    cse.SILHOUETTE_MAX_N = args.n + 10

    overrides = dict(cluster_method=args.method, amp_dur_weight=0.0, polarity_weight=0.0,
                      threshold_mad=sst.DETECTION_THRESHOLD_MAD)
    peaks_t, peak_amps, waveforms, emb, raw_labels = cluster_from_scratch(
        trace, fs, baseline_s, args.method, overrides)
    n_spikes = raw_labels.size
    print(f"n={args.n}: t-SNE embedding computed once for {n_spikes} spikes; "
          f"raw k-means -> {int(raw_labels.max()) + 1} types")

    # --- panel 2: after post-processing ---
    if args.postproc == "full":
        keep, pp_labels = post_process(peaks_t, peak_amps, waveforms, raw_labels, fs)
    else:
        keep, pp_labels = np.ones(n_spikes, dtype=bool), raw_labels.copy()
    pp_full = np.full(n_spikes, -1)
    pp_full[keep] = pp_labels
    print(f"  {args.postproc} post-processing -> {int(pp_labels.max()) + 1} types, "
          f"{keep.sum()} spikes ({(~keep).sum()} removed/greyed)")

    # --- panel 3: after split-refinement (on the post-processed set) ---
    n_pp = int(pp_labels.max()) + 1
    split_raw, report = msr.apply_split(pp_labels, waveforms[keep], n_pp)
    n_split = sum(1 for r in report if r["split"])
    split_dense, n_after_split, _ = msr.relabel_dense_tracked(split_raw)
    merged_raw, l2_thr, _, groups, _ = msr.apply_l2_merge(split_dense, waveforms[keep],
                                                            n_after_split, fs)
    after_labels, n_after, _ = msr.relabel_dense_tracked(merged_raw)
    after_full = np.full(n_spikes, -1)
    after_full[keep] = after_labels
    print(f"  split-refinement -> {n_after_split} (split, {n_split} type(s)) -> "
          f"{n_after} (L2 merge, {len(groups)} group(s))")

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharex=True, sharey=True)
    scatter_panel(axes[0], emb, raw_labels, "Original t-SNE  -  raw k-means")
    scatter_panel(axes[1], emb, pp_full, f"After post-processing  ({args.postproc})")
    scatter_panel(axes[2], emb, after_full, "After split-refinement")
    fig.suptitle(f"t-SNE feature space (computed once, points fixed), n={args.n} MUAPs | "
                 f"tsne + {args.method} | {args.postproc}  -  only the classification changes",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = os.path.join(base.SIM_ROOT, "tsne",
                             f"tsne_embedding_n{args.n}_{args.method}_{args.postproc}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
