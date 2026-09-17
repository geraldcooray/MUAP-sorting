#!/usr/bin/env python3
"""Run the spike-sorting pipeline multiple times with different parameter
overrides and compare the results side by side, in one figure:

    - One row per run showing the raw EMG trace with detected/sorted
      spikes marked, over a short comparison window (COMPARE_WINDOW_S).
    - Below that, one row per run of per-type MUAP superimposition +
      mean-template panels, computed over the FULL analysed record (not
      just the comparison window) -- so unit shapes are based on all the
      data, even though only a few seconds are shown in the trace above.

This is a comparison harness, not a CLI tool: edit RUNS and BASE_PARAMS
directly to change what's compared. Each entry in RUNS is a dict of
overrides applied on top of BASE_PARAMS (same meanings as the tunables in
spike_sort_emg.py) for one labeled run, e.g. to compare PCA vs t-SNE shape
features:

    RUNS = [
        {"label": "PCA",   "feature_method": "pca"},
        {"label": "t-SNE", "feature_method": "tsne"},
    ]

Any BASE_PARAMS key can be overridden per run this way -- clustering
method, detection thresholds, window sizes, etc. -- not just feature
extraction.

Usage:
    python call_spike_sort_emg.py [filename]
"""
import copy
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spike_sort.clustering import (cluster_spikes, refine_labels_by_firing_regularity,
                                    relabel_dense, silhouette_sweep)
from spike_sort.detection import bandpass, detect_spikes, extract_waveforms
from spike_sort.features import build_features, dct_highpass_waveforms, spike_amplitude_duration
from spike_sort.io import load_signal, lookup_sample_rate
from spike_sort.isi_cleanup import enforce_min_isi_per_type
from spike_sort.latency_linkage import find_linked_type_pairs, group_linked_types, merge_linked_types
from spike_sort.l2_merge import (elbow_threshold, find_close_type_pairs, group_close_types,
                                  merge_close_types, outlier_fence_threshold,
                                  pairwise_l2_distances, percentile_threshold)
from spike_sort.metrics import isi_cv, muap_metrics
from spike_sort.plotting import CLUSTER_COLORS, plot_feature_space_2d, plot_muap_panel

# ---------------------------------------------------------------------------
# Comparison runs. Add/remove/edit entries to compare whatever you like --
# each dict overrides BASE_PARAMS below for that one labeled run.
# ---------------------------------------------------------------------------
#RUNS = [
#    {"label": "refractory=4ms",  "refractory_ms": 4.0},
#    {"label": "refractory=10ms", "refractory_ms": 10.0},
#]

RUNS = [
    {"label": "Threshold=1.0", "threshold_mad": 1.0, "feature_method": "tsne", "compare_window_s": 2.0,
     "isi_weight": 0.0, "polarity_weight": 0.0, "amp_dur_weight": 0.0, "n_components": 2, "max_rate_hz":10.0},
    # isi_weight/polarity_weight forced to 0 here so the feature-space plot's dim1-vs-dim2 view
    # shows exactly what drove clustering -- otherwise labels get shaped by polarity and firing-rate
    # regularity, which aren't visible on those 2 axes and make the plotted clusters look scrambled
    # even though the underlying (higher-dimensional) clustering is doing what it's told. Re-enable
    # them for real analysis once you trust the shape clustering on its own.
]
# ---------------------------------------------------------------------------
# Parameters shared by every run unless overridden in RUNS above. Same
# meanings as the tunables in spike_sort_emg.py.
# ---------------------------------------------------------------------------
BASE_PARAMS = dict(
    filename="EMGRUN__001.txt",
    fs=None,
    start=0.0,
    duration=None,              # None = full signal, used for clustering/MUAPs in every run
    compare_window_s=5.0,       # seconds of raw trace shown per run in the comparison row

    # Spike detection
    highpass=500.0,
    threshold_mad=3.0,
    smoothing_ms=3.0,
    refractory_ms=2.0,
    noise_baseline_samples=None,  # if set, the noise-sigma estimate (see detect_spikes) only
                                   # uses the signal's first this-many samples, rather than the
                                   # whole record -- for a known spike-free lead-in segment

    # Waveform extraction / measurement
    window_ms=5.0,
    baseline_window_ms=10.0,    # half-window used only to estimate each spike's baseline -- wider
                                 # than window_ms so the extracted waveform keeps its own real DC
                                 # offset (e.g. from needle position) instead of being forced to
                                 # zero mean by baselining against itself -- see extract_waveforms
    measure_band=(20.0, 10000.0),

    # Post-extraction waveform conditioning
    dct_filter_n=0,              # if > 0, DCT-transform every extracted waveform, zero its
                                 # `dct_filter_n` lowest coefficients (DC + slowest cosines =
                                 # residual baseline offset / slow drift), and invert -- a
                                 # ringing-free high-pass on the per-trial MUAP snippets, applied
                                 # before feature extraction, clustering AND mean-template
                                 # averaging (see dct_highpass_waveforms). 0 disables.

    # Shape features
    feature_method="tsne",       # Types of clustering methods tsne, "wavelet"
    n_components=2,
    wavelet="sym2",
    wavelet_level=None,
    wavelet_clip_percentile=1.0,  # winsorize wavelet coefficients against single-outlier
                                   # domination before scoring/selecting them (0 disables)
    tsne_perplexity=30.0,
    amp_dur_weight=0.0,
    polarity_weight=1.0,  # up-weighting of polarity (sign of each spike's largest deflection)
                           # vs shape when clustering -- amplitude/duration can't distinguish a
                           # spike from its polarity-inverted mirror image by construction, so
                           # this is what actually separates them (0 disables)

    # Clustering
    n_clusters=None,
    max_rate_hz=10.0,
    cluster_method="kmeans",   # clustering algorithm: "kmeans", "gmm", "tmixture"
    tmix_dof=8.0,
    tmix_iters=100,
    hdbscan_min_cluster_size=5,
    hdbscan_min_samples=None,
    spc_knn=10,
    spc_q_states=20,
    spc_t_min=0.05,
    spc_t_max=1.5,
    spc_n_temps=20,
    spc_mc_steps=100,
    spc_min_clus_frac=0.1,

    # Firing-regularity refinement
    isi_weight=0.2,
    isi_window_s=2.5,
    isi_iters=5,

    # Minimum-ISI cleanup, applied per type after final labels are set (0 disables)
    min_isi_ms=10.0,

    # Same-MUAP type-linkage cleanup, applied after min-ISI cleanup (0 disables): merge pairs
    # of types that almost always fire together at a tight, near-fixed latency (a secondary
    # lobe/turn extracted as its own "type" rather than being fully collapsed into its parent
    # MUAP's detection -- see spike_sort.latency_linkage). link_max_latency_ms doubles as both
    # the co-occurrence search window and the post-merge duplicate-detection cleanup window.
    link_max_latency_ms=8.0,
    link_min_co_occurrence=0.6,
    link_max_latency_std_ms=1.0,

    # Shape-similarity merge, applied after the linkage-cleanup above (None/0 disables): merge
    # any two types whose averaged (mean) waveforms are close in raw, UNNORMALIZED L2
    # (Euclidean) distance -- t-SNE's nonlinear embedding can still split one true MUAP shape
    # into two nearby types even after the timing-based linkage merge (see spike_sort.l2_merge).
    # No shipped default: what counts as "close" depends entirely on this recording's own raw
    # amplitude scale, so pick a threshold deliberately (e.g. by inspecting
    # spike_sort.l2_merge.pairwise_l2_distances on one run first). Unlike link_max_latency_ms,
    # this never drops any spikes -- it only relabels, since an L2 shape match says two types
    # ARE the same MUAP, not that any individual detection is a duplicate of another.
    l2_merge_threshold=None,
    # Alternative to a fixed l2_merge_threshold: merge the closest l2_merge_percentile% of type
    # pairs by L2 distance (e.g. 20 -> merge the smallest 20% of pairwise distances), recomputed
    # per run from that run's own distance distribution. Only used when l2_merge_threshold is
    # None; ignored otherwise. CAUTION -- validated against ground truth on the simulated traces
    # and found to over-merge badly as type count grows (union-find chains distinct types
    # together once enough pairs clear a fixed percentile); l2_merge_elbow below tends to do
    # much better.
    l2_merge_percentile=None,
    # Another alternative: pick the threshold automatically at the largest gap in the sorted
    # pairwise-distance curve (see spike_sort.l2_merge.elbow_threshold). CAUTION -- validated
    # against ground truth and found to fail badly (collapsed almost everything into one type on
    # 3 of 4 test traces; the largest gap usually sits between the two largest distances, not at
    # a real cluster boundary). Kept for reference; prefer l2_merge_outlier_fence_k below. Only
    # used when l2_merge_threshold and l2_merge_percentile are both None.
    l2_merge_elbow=False,
    # Best-validated automatic option: median(distances) - k * MAD(distances), a robust low-
    # outlier fence (see spike_sort.l2_merge.outlier_fence_threshold) -- matches this problem's
    # actual shape (a few genuine duplicates as low outliers, not a bimodal split) much better
    # than either alternative above. None disables; only used when l2_merge_threshold,
    # l2_merge_percentile, and l2_merge_elbow are all unset/False.
    l2_merge_outlier_fence_k=None,
    # Compare each type pair's mean waveform after scaling both to unit area (sum(|waveform|)
    # = 1) instead of raw units -- an explicit, opt-in shape-only comparison, blind to overall
    # amplitude (see spike_sort.l2_merge.normalize_area). Mixed/exploratory result on the
    # simulated traces: roughly equivalent to raw at low unit counts, but triggers much more
    # aggressive (though roughly purity-neutral, not clearly harmful) merging at higher unit
    # counts -- not validated as a clear improvement, use deliberately. Applies to whichever
    # threshold method above is actually selected (explicit/percentile/elbow/outlier-fence).
    l2_merge_normalize_area=False,

    seed=42,
)

MUAP_YLIM = (-500, 500)
NEGATIVE_UP = True
OUT = None                # comparison-plot PNG path; None = auto-named from input filename
OUT_FEATURE_SPACE = None  # feature-space-comparison PNG path; None = auto-named
FEATURE_SPACE_SHOW_ELLIPSES = True  # draw a per-type covariance ellipse on the 2D
                                     # feature-space scatters (see plot_feature_space_2d)
FEATURE_SPACE_ELLIPSE_STD = 2.0     # ellipse size, in standard deviations

OUT_SILHOUETTE = None       # silhouette-sweep-plot PNG path; None = auto-named
OUT_AMP_DUR_SHAPE = None    # amplitude/duration/shape-dim comparison PNG path; None = auto-named
SILHOUETTE_SWEEP = True     # for fixed-n_clusters methods (kmeans/gmm/tmixture), fit every
                             # n_clusters from n0 (the max_rate_hz-based firing-rate estimate --
                             # the fewest types consistent with no unit exceeding max_rate_hz, so
                             # it's a floor, not a target) up to SILHOUETTE_MAX_N, and score each
                             # with the silhouette coefficient; the highest-scoring n_clusters is
                             # then used for the actual clustering in this run (n0 itself is kept
                             # only for reporting/plot titles) -- see silhouette_sweep
SILHOUETTE_MAX_N = 20

EXPORT_FEATURES_CSV = True  # write each run's per-spike feature/embedding data (shape-feature
                             # coords, amplitude, duration, polarity, cluster label) to a CSV --
                             # see export_features_csv


def run_pipeline(params):
    """Detection -> feature extraction -> clustering -> firing-regularity
    refinement for one parameter set. Returns everything needed to plot
    it, so the raw signal/detection only ever get run once per RUNS
    entry."""
    filename = params["filename"]
    fs = params["fs"] if params["fs"] is not None else lookup_sample_rate(filename)
    signal = load_signal(filename, fs, params["start"], params["duration"])
    t = params["start"] + np.arange(signal.size) / fs
    analysis_duration = (t[-1] - t[0]) + 1 / fs

    peaks, threshold, det_filtered, det_envelope = detect_spikes(
        signal, fs, params["highpass"], params["threshold_mad"], params["refractory_ms"],
        params["smoothing_ms"], noise_baseline_samples=params.get("noise_baseline_samples"))

    picks_own_cluster_count = params["cluster_method"] in ("hdbscan", "spc")
    if picks_own_cluster_count:
        n_clusters = None
    elif params["n_clusters"] is not None:
        n_clusters = params["n_clusters"]
    else:
        n_clusters = max(1, int(np.ceil(peaks.size / (params["max_rate_hz"] * analysis_duration))))

    if not picks_own_cluster_count and peaks.size < n_clusters:
        sys.exit(f"[{params['label']}] Only {peaks.size} spikes detected -- lower "
                  f"threshold_mad or n_clusters (need >= {n_clusters}).")

    measured = bandpass(signal, fs, params["measure_band"][0], params["measure_band"][1])
    peaks, waveforms, half_win = extract_waveforms(measured, peaks, fs, params["window_ms"],
                                                    params["baseline_window_ms"])
    if params.get("dct_filter_n", 0) > 0:
        waveforms = dct_highpass_waveforms(waveforms, params["dct_filter_n"])
    peaks_t = t[peaks]

    features = build_features(waveforms, fs, params["n_components"], params["seed"],
                               params["amp_dur_weight"], method=params["feature_method"],
                               wavelet=params["wavelet"], wavelet_level=params["wavelet_level"],
                               tsne_perplexity=params["tsne_perplexity"],
                               wavelet_clip_percentile=params["wavelet_clip_percentile"],
                               polarity_weight=params["polarity_weight"])

    n0 = n_clusters  # firing-rate-based estimate, kept for reporting even if the sweep overrides it

    silhouette_scores = {}
    if SILHOUETTE_SWEEP and not picks_own_cluster_count:
        n_values = range(max(n0, 2), SILHOUETTE_MAX_N + 1)
        silhouette_scores = silhouette_sweep(
            features, n_values, params["seed"], method=params["cluster_method"],
            tmix_dof=params["tmix_dof"], tmix_iters=params["tmix_iters"])
        valid = {n: s for n, s in silhouette_scores.items() if s == s}  # drop nan
        if valid:
            n_clusters = max(valid, key=valid.get)  # use the silhouette-maximizing count for real

    labels = cluster_spikes(features, n_clusters, params["seed"], method=params["cluster_method"],
                             tmix_dof=params["tmix_dof"], tmix_iters=params["tmix_iters"],
                             hdbscan_min_cluster_size=params["hdbscan_min_cluster_size"],
                             hdbscan_min_samples=params["hdbscan_min_samples"],
                             spc_knn=params["spc_knn"], spc_q_states=params["spc_q_states"],
                             spc_t_min=params["spc_t_min"], spc_t_max=params["spc_t_max"],
                             spc_n_temps=params["spc_n_temps"], spc_mc_steps=params["spc_mc_steps"],
                             spc_min_clus_frac=params["spc_min_clus_frac"])

    n_noise = 0
    if (labels == -1).any():
        n_noise = int((labels == -1).sum())
        keep = labels != -1
        if not keep.any():
            sys.exit(f"[{params['label']}] {params['cluster_method']} left no spikes "
                      f"after removing noise.")
        peaks, waveforms, peaks_t, features, labels = (
            peaks[keep], waveforms[keep], peaks_t[keep], features[keep], labels[keep])

    labels = refine_labels_by_firing_regularity(
        labels, peaks_t, features, int(labels.max()) + 1,
        params["isi_weight"], params["isi_window_s"], params["isi_iters"])
    labels, n_types = relabel_dense(labels)

    peak_amps = signal[peaks]

    n_isi_dropped = 0
    if params["min_isi_ms"] > 0:
        keep = enforce_min_isi_per_type(peaks_t, labels, peak_amps, params["min_isi_ms"] * 1e-3)
        if not keep.all():
            n_isi_dropped = int((~keep).sum())
            peaks, waveforms, peaks_t, peak_amps, features, labels = (
                peaks[keep], waveforms[keep], peaks_t[keep], peak_amps[keep],
                features[keep], labels[keep])

    n_types_linked = 0
    n_link_dropped = 0
    linked_pairs = []
    if params["link_max_latency_ms"] > 0:
        linked_pairs = find_linked_type_pairs(
            peaks_t, labels, max_latency_ms=params["link_max_latency_ms"],
            min_co_occurrence=params["link_min_co_occurrence"],
            max_latency_std_ms=params["link_max_latency_std_ms"])
        groups = group_linked_types(linked_pairs)
        if groups:
            n_types_linked = sum(len(g) for g in groups)
            merged_labels, keep = merge_linked_types(
                peaks_t, labels, peak_amps, groups, params["link_max_latency_ms"])
            n_link_dropped = int((~keep).sum())
            peaks, waveforms, peaks_t, peak_amps, features, labels = (
                peaks[keep], waveforms[keep], peaks_t[keep], peak_amps[keep],
                features[keep], merged_labels[keep])
            labels, n_types = relabel_dense(labels)

    # Computed unconditionally (cheap -- at most a few dozen types) so callers always have the
    # pre-merge distance distribution available for diagnostics/plotting, whether or not a merge
    # actually runs below.
    l2_normalize = params.get("l2_merge_normalize_area", False)
    l2_distances = pairwise_l2_distances(waveforms, labels, normalize=l2_normalize)

    n_types_l2_merged = 0
    close_pairs = []
    l2_threshold_used = params["l2_merge_threshold"]
    if l2_threshold_used is None and params.get("l2_merge_percentile"):
        l2_threshold_used = percentile_threshold(l2_distances, params["l2_merge_percentile"])
    if l2_threshold_used is None and params.get("l2_merge_elbow"):
        l2_threshold_used = elbow_threshold(l2_distances)
        if l2_threshold_used != l2_threshold_used:  # NaN (fewer than 3 types -> no gap to find)
            l2_threshold_used = None
    if l2_threshold_used is None and params.get("l2_merge_outlier_fence_k") is not None:
        l2_threshold_used = outlier_fence_threshold(l2_distances, params["l2_merge_outlier_fence_k"])
        if l2_threshold_used != l2_threshold_used:  # NaN (fewer than 3 types)
            l2_threshold_used = None
    if l2_threshold_used:
        close_pairs = find_close_type_pairs(waveforms, labels, l2_threshold_used,
                                             normalize=l2_normalize)
        l2_groups = group_close_types(close_pairs)
        if l2_groups:
            n_types_l2_merged = sum(len(g) for g in l2_groups)
            labels = merge_close_types(labels, l2_groups)  # relabel only -- never drops spikes
            labels, n_types = relabel_dense(labels)

    metrics_by_cluster = {c: muap_metrics(waveforms[labels == c].mean(axis=0), fs)
                           for c in range(n_types)}
    firing_rates = {c: (labels == c).sum() / analysis_duration for c in range(n_types)}
    isi_cvs = {c: isi_cv(peaks_t[labels == c]) for c in range(n_types)}
    amplitudes, durations, polarities = spike_amplitude_duration(waveforms, fs)

    return dict(
        label=params["label"], t=t, signal=signal, peaks_t=peaks_t, peak_amps=peak_amps,
        labels=labels, waveforms=waveforms, features=features, half_win=half_win, fs=fs,
        n_types=n_types, metrics_by_cluster=metrics_by_cluster, firing_rates=firing_rates,
        isi_cvs=isi_cvs, n_spikes=int(peaks.size), n_noise=n_noise, n_isi_dropped=n_isi_dropped,
        n_types_linked=n_types_linked, n_link_dropped=n_link_dropped, linked_pairs=linked_pairs,
        n_types_l2_merged=n_types_l2_merged, close_pairs=close_pairs,
        l2_distances=l2_distances, l2_threshold_used=l2_threshold_used,
        analysis_duration=analysis_duration, compare_window_s=params["compare_window_s"],
        feature_method=params["feature_method"], n0=n0, n_clusters_used=n_clusters,
        silhouette_scores=silhouette_scores, amplitudes=amplitudes, durations=durations,
        polarities=polarities,
    )


def export_features_csv(r, filename, out_dir=None):
    """Write the per-spike feature/embedding data behind the feature-space
    scatterplots (plot_feature_space_comparison) to a CSV -- one row per
    spike: its shape-feature coordinates (e.g. t-SNE dim 1/2/3, PCA
    components, or wavelet coefficients -- whichever `feature_method` built
    `features`), plus physical amplitude/duration/polarity and the final
    cluster label, all unstandardized/unweighted (raw physical units) so
    the file is directly usable by other tools, unlike `features` itself
    (standardized and amp/dur/polarity-weighted for clustering).

    `r["features"]` always has exactly 3 trailing columns appended by
    build_features (amplitude, duration, polarity, in that order) after
    the shape-feature columns -- see build_features in spike_sort/features.py."""
    method = r["feature_method"]
    n_shape_dims = r["features"].shape[1] - 3
    shape_cols = {f"{method}_dim{d + 1}": r["features"][:, d] for d in range(n_shape_dims)}

    df = pd.DataFrame({
        "spike_index": np.arange(len(r["labels"])),
        "time_s": r["peaks_t"],
        "peak_amplitude_raw": r["peak_amps"],
        "amplitude": r["amplitudes"],
        "duration_ms": r["durations"],
        "polarity": r["polarities"],
        **shape_cols,
        "cluster_label": r["labels"],
    })

    out_dir = out_dir or os.path.dirname(filename) or "."
    label_slug = r["label"].replace(" ", "_").replace("=", "")
    out_path = os.path.join(
        out_dir, f"{os.path.splitext(os.path.basename(filename))[0]}_{label_slug}_features.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved feature/scatter data ({len(df)} spikes) to {out_path}")
    return out_path


def plot_comparison(results, filename, out_path):
    n_runs = len(results)
    max_types = max(r["n_types"] for r in results)

    fig = plt.figure(figsize=(4.5 * max_types, 5.6 * n_runs))
    gs = fig.add_gridspec(2 * n_runs, max_types, height_ratios=[2.2, 2.4] * n_runs,
                           hspace=0.8, wspace=0.35)

    for i, r in enumerate(results):
        colors = [CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in range(r["n_types"])]
        t, signal = r["t"], r["signal"]

        ax_emg = fig.add_subplot(gs[2 * i, :])
        ax_emg.plot(t, signal, linewidth=0.4, color="0.4", zorder=1)
        for c in range(r["n_types"]):
            mask = r["labels"] == c
            ax_emg.scatter(r["peaks_t"][mask], r["peak_amps"][mask], s=14, color=colors[c],
                            zorder=2, label=f"type {c} (n={mask.sum()})")
        window_end = t[0] + r["compare_window_s"]
        ax_emg.set_xlim(t[0], min(window_end, t[-1]))
        ax_emg.set_ylabel("Amplitude")
        noise_note = f", {r['n_noise']} noise" if r["n_noise"] else ""
        isi_note = f", {r['n_isi_dropped']} dropped (min-isi)" if r["n_isi_dropped"] else ""
        link_note = (f", {r['n_types_linked']} types merged/{r['n_link_dropped']} dropped "
                     f"(linkage)" if r["n_types_linked"] else "")
        l2_note = (f", {r['n_types_l2_merged']} types merged (L2)" if r["n_types_l2_merged"]
                   else "")
        n_clusters_note = (f" [n_clusters={r['n_clusters_used']} silhouette-optimal, n0={r['n0']}]"
                            if r["silhouette_scores"] else "")
        ax_emg.set_title(f"{r['label']} — {r['n_spikes']} spikes, {r['n_types']} types"
                          f"{noise_note}{isi_note}{link_note}{l2_note} over "
                          f"{r['analysis_duration']:.1f} s "
                          f"(showing first {r['compare_window_s']:.1f} s){n_clusters_note}",
                          fontsize=9)
        ax_emg.set_xlabel("Time (s)")
        ax_emg.legend(loc="upper right", fontsize=7, ncol=min(r["n_types"], 6))
        if NEGATIVE_UP:
            ax_emg.invert_yaxis()

        t_wave_ms = (np.arange(-r["half_win"], r["half_win"]) / r["fs"]) * 1000
        rng = np.random.default_rng(0)
        for c in range(max_types):
            ax = fig.add_subplot(gs[2 * i + 1, c])
            if c >= r["n_types"]:
                ax.axis("off")
                continue
            mask = r["labels"] == c
            unit_waveforms = r["waveforms"][mask]
            mean_wf = unit_waveforms.mean(axis=0)
            m = r["metrics_by_cluster"][c]
            title = (f"type {c} (n={mask.sum()}, {r['firing_rates'][c]:.1f} Hz, "
                     f"CV(ISI) {r['isi_cvs'][c]:.2f})\n"
                     f"dur {m['duration_ms']:.2f} ms, amp {m['amplitude']:.0f}, "
                     f"{m['phases']}ph/{m['turns']}t")
            plot_muap_panel(ax, unit_waveforms, mean_wf, m, t_wave_ms, colors[c], title,
                             muap_ylim=MUAP_YLIM, negative_up=NEGATIVE_UP, rng=rng,
                             ylabel="Amplitude\n(measurement band)" if c == 0 else None)

    fig.suptitle(f"{os.path.basename(filename)} — spike-sort comparison "
                 f"({', '.join(r['label'] for r in results)})", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    print(f"Saved comparison plot to {out_path}")
    plt.show()


def plot_feature_space_comparison(results, filename, out_path):
    """Separate figure, independent of plot_comparison: for each run, the
    3 pairwise 2D scatters of the first 3 shape-feature dimensions
    (dim1-vs-dim2, dim1-vs-dim3, dim2-vs-dim3) -- one row per run, 3
    columns -- so cluster separation in feature space can be compared
    directly across runs without competing for space with the
    raw-trace/MUAP panels, and without a 3D scatter's occlusion ambiguity."""
    n_runs = len(results)
    dim_pairs = [(0, 1), (0, 2), (1, 2)]

    fig, axes = plt.subplots(n_runs, 3, figsize=(5.0 * 3, 5.0 * n_runs), squeeze=False)

    for i, r in enumerate(results):
        method = r["feature_method"]
        for j, (dx, dy) in enumerate(dim_pairs):
            axis_labels = (f"{method} dim {dx + 1}", f"{method} dim {dy + 1}")
            title = f"{r['label']} ({r['n_types']} types) — dim {dx + 1} vs dim {dy + 1}"
            plot_feature_space_2d(axes[i][j], r["features"], r["labels"], dx, dy,
                                   axis_labels=axis_labels, title=title, legend=(j == 0),
                                   show_ellipses=FEATURE_SPACE_SHOW_ELLIPSES,
                                   ellipse_n_std=FEATURE_SPACE_ELLIPSE_STD)

    fig.suptitle(f"{os.path.basename(filename)} — shape-feature space comparison "
                 f"({', '.join(r['label'] for r in results)})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    print(f"Saved feature-space comparison plot to {out_path}")
    plt.show()


def plot_amp_dur_shape_comparison(results, filename, out_path):
    """Separate figure, independent of plot_comparison: for each run, the
    3 pairwise 2D scatters of peak-to-peak amplitude, duration, and the
    single most discriminative shape-feature dimension (dim 1 of whichever
    method built `features` -- PCA/wavelet/t-SNE) -- one row per run, 3
    columns. Amplitude and duration are standard, physically/clinically
    meaningful MUAP parameters (unlike the abstract shape-feature axes in
    plot_feature_space_comparison), so this is a more direct check of
    whether the types found actually correspond to differently-shaped/
    -sized motor units, rather than an artifact of the shape-feature
    space alone."""
    n_runs = len(results)
    pairs = [("amp", "dur"), ("amp", "shape"), ("dur", "shape")]

    fig, axes = plt.subplots(n_runs, 3, figsize=(5.0 * 3, 5.0 * n_runs), squeeze=False)

    for i, r in enumerate(results):
        method = r["feature_method"]
        combo = np.column_stack([r["amplitudes"], r["durations"], r["features"][:, 0]])
        names = {"amp": "amplitude", "dur": "duration (ms)", "shape": f"{method} dim 1"}
        col = {"amp": 0, "dur": 1, "shape": 2}
        for j, (a, b) in enumerate(pairs):
            axis_labels = (names[a], names[b])
            title = f"{r['label']} ({r['n_types']} types) — {names[a]} vs {names[b]}"
            plot_feature_space_2d(axes[i][j], combo, r["labels"], col[a], col[b],
                                   axis_labels=axis_labels, title=title, legend=(j == 0),
                                   show_ellipses=FEATURE_SPACE_SHOW_ELLIPSES,
                                   ellipse_n_std=FEATURE_SPACE_ELLIPSE_STD)

    fig.suptitle(f"{os.path.basename(filename)} — amplitude / duration / shape comparison "
                 f"({', '.join(r['label'] for r in results)})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    print(f"Saved amplitude/duration/shape comparison plot to {out_path}")
    plt.show()


def plot_silhouette_sweep(results, filename, out_path):
    """Separate figure: silhouette score vs. candidate n_clusters, one
    line per run, with the firing-rate-based estimate n0 (dotted) and the
    silhouette-maximizing count actually used for the run (solid) both
    marked -- lets you see at a glance whether n0 was anywhere near the
    best-scoring cluster count. Runs with no sweep data (cluster_method
    picks its own cluster count, e.g. hdbscan/spc) are skipped."""
    sweepable = [r for r in results if r["silhouette_scores"]]
    if not sweepable:
        print("No runs had a silhouette sweep to plot (fixed-n_clusters methods only).")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, r in enumerate(sweepable):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        ns = sorted(r["silhouette_scores"])
        scores = [r["silhouette_scores"][n] for n in ns]
        ax.plot(ns, scores, marker="o", color=color, label=r["label"])
        ax.axvline(r["n0"], color=color, linestyle=":", linewidth=1, alpha=0.7)
        ax.axvline(r["n_clusters_used"], color=color, linestyle="-", linewidth=1.5, alpha=0.7)

    ax.set_xlabel("n_clusters")
    ax.set_ylabel("silhouette score")
    ax.set_title(f"{os.path.basename(filename)} — silhouette score vs. n_clusters\n"
                 f"(dotted = firing-rate estimate n0, solid = silhouette-optimal count used)",
                 fontsize=10)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved silhouette-sweep plot to {out_path}")
    plt.show()


def main():
    if len(sys.argv) > 1:
        BASE_PARAMS["filename"] = sys.argv[1]

    if not os.path.isfile(BASE_PARAMS["filename"]):
        sys.exit(f"File not found: {BASE_PARAMS['filename']}")

    results = []
    for run in RUNS:
        params = copy.deepcopy(BASE_PARAMS)
        params.update(run)
        print(f"Running {params['label']}...")
        r = run_pipeline(params)
        results.append(r)
        print(f"  {r['n_spikes']} spikes, {r['n_types']} types"
              + (f", {r['n_noise']} noise" if r["n_noise"] else "")
              + (f", {r['n_isi_dropped']} dropped (min-isi)" if r["n_isi_dropped"] else "")
              + (f", {r['n_types_linked']} types merged/{r['n_link_dropped']} dropped (linkage)"
                 if r["n_types_linked"] else "")
              + (f", {r['n_types_l2_merged']} types merged (L2)" if r["n_types_l2_merged"] else ""))
        if r["silhouette_scores"]:
            print(f"  silhouette sweep (n0={r['n0']} -> using n={r['n_clusters_used']}): "
                  + ", ".join(f"n={n}: {s:.3f}" if s == s else f"n={n}: nan"
                              for n, s in sorted(r["silhouette_scores"].items())))
        if EXPORT_FEATURES_CSV:
            export_features_csv(r, BASE_PARAMS["filename"])

    out_path = OUT or f"{os.path.splitext(BASE_PARAMS['filename'])[0]}_diagnostic_compare.png"
    plot_comparison(results, BASE_PARAMS["filename"], out_path)

    feature_space_out_path = (OUT_FEATURE_SPACE
                               or f"{os.path.splitext(BASE_PARAMS['filename'])[0]}"
                                  f"_diagnostic_feature_space.png")
    plot_feature_space_comparison(results, BASE_PARAMS["filename"], feature_space_out_path)

    silhouette_out_path = (OUT_SILHOUETTE
                            or f"{os.path.splitext(BASE_PARAMS['filename'])[0]}"
                               f"_diagnostic_silhouette.png")
    plot_silhouette_sweep(results, BASE_PARAMS["filename"], silhouette_out_path)

    amp_dur_shape_out_path = (OUT_AMP_DUR_SHAPE
                               or f"{os.path.splitext(BASE_PARAMS['filename'])[0]}"
                                  f"_diagnostic_amp_dur_shape.png")
    plot_amp_dur_shape_comparison(results, BASE_PARAMS["filename"], amp_dur_shape_out_path)


if __name__ == "__main__":
    main()
