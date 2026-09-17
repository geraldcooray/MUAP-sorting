#!/usr/bin/env python3
"""Joint spike-sorting across every EMG run at once.

Where multi_run_spike_sort.py sorts each EMGRUN__*.txt file completely
independently (and deliberately never compares types across runs), this
script does the opposite for step 2:

    1. DETECTION -- per run, unchanged. Each EMGRUN__*.txt is highpass
       filtered, rectified/smoothed to an envelope, and peak-picked
       exactly as before (spike_sort.detection.detect_spikes), then a
       fixed-width waveform is cut around every detection
       (spike_sort.detection.extract_waveforms).

    2. CLUSTERING -- ONE pass over all runs together. Every detected
       spike's waveform from every run is stacked into a single matrix,
       one shape-feature embedding (t-SNE by default) is fitted on that
       whole pool, and one k-means labelling is produced from it. A
       cluster label therefore means the same thing in run 000 and run
       003: they were grouped in the same feature space at the same time.

       Caveat carried over from multi_run_spike_sort.py: different
       EMGRUN files are different needle placements, so a type shared
       across runs is a shared *waveform shape*, not proof of the same
       physical motor unit. Joint clustering here is an explicitly
       requested shape-space grouping; read cross-run type identity with
       that in mind.

    3. OUTPUTS built from that single labelling:
         * joint_run_raster.png       -- one raster panel per run, its own
                                         independent time axis, but colours
                                         now mean the same type across runs.
         * joint_run_muaps.png        -- one MUAP panel per global type,
                                         individual trials pooled across
                                         every run + the pooled mean template.
         * joint_run_muaps_by_run.png -- a type x run grid of mean templates,
                                         to eyeball whether a type's shape
                                         actually holds up across needle sites.
         * joint_run_feature_space.png-- the single joint embedding, once
                                         coloured by type, once by run.
         * joint_run_silhouette.png   -- silhouette score vs candidate
                                         cluster count for the pooled data.
         * joint_run_muap_metrics.png -- pooled peak-to-peak amplitude and
                                         duration, one value per global type.
         * processed_data/<muscle>_<side>_joint_spike_sort_data.npz --
                                         per-run arrays (schema-compatible
                                         with muap_post_selection.py, using
                                         the global type count) plus pooled
                                         per-type arrays and the raw global
                                         label/run-id vectors.

Edit RUN_FILES / PARAM_OVERRIDES / the config constants below directly --
this is a harness script, not a CLI tool. PARAM_OVERRIDES is layered on
top of call_spike_sort_emg.BASE_PARAMS, so every detection/window tunable
is inherited from there.

Usage (run from a directory holding EMGRUN__*.txt + Header.txt):
    cd .../test0_myopathy/EMG/Tibialis_anterior/Left
    python /path/to/python/joint_run_spike_sort.py
"""
import copy
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import silhouette_score

from call_spike_sort_emg import BASE_PARAMS
from spike_sort.clustering import cluster_spikes, relabel_dense
from spike_sort.detection import bandpass, detect_spikes, extract_waveforms
from spike_sort.features import build_features, dct_highpass_waveforms
from spike_sort.io import load_signal, lookup_sample_rate
from spike_sort.isi_cleanup import enforce_min_isi_per_type
from spike_sort.metrics import muap_metrics
from spike_sort.plotting import CLUSTER_COLORS, plot_feature_space_2d, plot_muap_panel

# ---------------------------------------------------------------------------
RUN_FILES = sorted(glob.glob("EMGRUN__*.txt"))

# Layered on top of call_spike_sort_emg.BASE_PARAMS for the whole pool --
# the t-SNE + k-means shape configuration confirmed to give clean,
# well-separated clusters (see multi_run_spike_sort.py / call_spike_sort_emg.py).
PARAM_OVERRIDES = dict(
    threshold_mad=1.0,
    feature_method="tsne",
    cluster_method="kmeans",
    isi_weight=0.0,
    polarity_weight=0.0,
    amp_dur_weight=0.0,
    n_components=2,
    max_rate_hz=10.0,
    dct_filter_n=2,        # DCT high-pass every extracted waveform (drop the 2 lowest
                           # coefficients) before pooling/embedding/averaging -- see
                           # spike_sort.features.dct_highpass_waveforms
)
# ---------------------------------------------------------------------------

OUT_RASTER = "joint_run_raster.png"
OUT_MUAPS = "joint_run_muaps.png"
OUT_MUAPS_BY_RUN = "joint_run_muaps_by_run.png"
OUT_FEATURE_SPACE = "joint_run_feature_space.png"
OUT_SILHOUETTE = "joint_run_silhouette.png"
OUT_MUAP_METRICS = "joint_run_muap_metrics.png"
OUT_DATA_DIR = "processed_data"

FORCE_N_CLUSTERS = None        # set an int to skip the silhouette sweep and use exactly that many
SILHOUETTE_SWEEP = True        # sweep candidate cluster counts on the pooled features
SILHOUETTE_MAX_N = 15          # highest cluster count tried in the sweep
SILHOUETTE_SAMPLE = 5000       # silhouette is O(n^2); score it on at most this many pooled spikes
MIN_SPIKES_PER_TYPE = 50       # global types with fewer pooled detections than this are dropped
                               #   from every plot (kept in full in the .npz)

MUAP_YLIM = None               # None = autoscale each MUAP panel to its own mean-template range
NEGATIVE_UP = True
FEATURE_SPACE_SHOW_ELLIPSES = True
FEATURE_SPACE_ELLIPSE_STD = 2.0
MUAP_METRICS_BINS = 25
MUAP_METRICS_KDE_POINTS = 200

RUN_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]
RUN_CMAP = plt.get_cmap("Set2").colors


# ---------------------------------------------------------------------------
# 1. Detection -- per run, independent
# ---------------------------------------------------------------------------
def detect_one_run(filename, params):
    """Run detection + waveform extraction for a single EMG run. No
    features, no clustering here -- those happen once, later, on the
    pooled set."""
    label = os.path.splitext(os.path.basename(filename))[0]
    fs = params["fs"] if params["fs"] is not None else lookup_sample_rate(filename)
    signal = load_signal(filename, fs, params["start"], params["duration"])
    t = params["start"] + np.arange(signal.size) / fs
    analysis_duration = (t[-1] - t[0]) + 1 / fs

    peaks, threshold, _, _ = detect_spikes(
        signal, fs, params["highpass"], params["threshold_mad"],
        params["refractory_ms"], params["smoothing_ms"])

    measured = bandpass(signal, fs, params["measure_band"][0], params["measure_band"][1])
    peaks, waveforms, half_win = extract_waveforms(
        measured, peaks, fs, params["window_ms"], params["baseline_window_ms"])

    return dict(
        label=label, fs=fs, half_win=half_win, analysis_duration=analysis_duration,
        peaks_t=t[peaks], peak_amps=signal[peaks], waveforms=waveforms,
    )


def run_detection(run_files, params):
    if not run_files:
        raise SystemExit("No EMGRUN__*.txt files found in the current directory.")

    runs = []
    for filename in run_files:
        r = detect_one_run(filename, params)
        runs.append(r)
        print(f"  {r['label']}: {r['waveforms'].shape[0]} spikes over "
              f"{r['analysis_duration']:.1f} s")

    widths = {r["waveforms"].shape[1] for r in runs if r["waveforms"].shape[0]}
    if len(widths) > 1:
        raise SystemExit(f"Runs have different waveform lengths {sorted(widths)} -- pooled "
                          f"clustering needs one common length (check sample rates in Header.txt).")
    rates = {r["fs"] for r in runs}
    if len(rates) > 1:
        raise SystemExit(f"Runs have different sample rates {sorted(rates)} -- cannot pool.")
    return runs


# ---------------------------------------------------------------------------
# 2. Clustering -- one pass over every run's spikes together
# ---------------------------------------------------------------------------
def choose_n_clusters(features, n0, seed, method):
    """Pick a cluster count for the pooled features. FORCE_N_CLUSTERS wins;
    otherwise sweep from max(n0, 2) to SILHOUETTE_MAX_N and take the
    silhouette-maximizing count, scoring on a random subsample (silhouette
    is O(n^2) in spikes). Returns (n_clusters, {n: score})."""
    if FORCE_N_CLUSTERS is not None:
        return FORCE_N_CLUSTERS, {}
    if not SILHOUETTE_SWEEP:
        return max(n0, 1), {}

    n = features.shape[0]
    rng = np.random.default_rng(seed)
    idx = (rng.choice(n, size=SILHOUETTE_SAMPLE, replace=False)
           if n > SILHOUETTE_SAMPLE else np.arange(n))

    sweep = {}
    for k in range(max(n0, 2), SILHOUETTE_MAX_N + 1):
        if k >= n:
            sweep[k] = float("nan")
            continue
        labels = cluster_spikes(features, k, seed, method=method)
        sub = labels[idx]
        if len(set(sub.tolist())) < 2:
            sweep[k] = float("nan")
            continue
        sweep[k] = float(silhouette_score(features[idx], sub))

    valid = {k: s for k, s in sweep.items() if s == s}
    best = max(valid, key=valid.get) if valid else max(n0, 1)
    return best, sweep


def joint_cluster(runs, params):
    """Stack every run's waveforms, build ONE feature embedding + ONE
    k-means labelling over the whole pool, then apply the per-type
    minimum-ISI cleanup within each run separately (spike times are only
    meaningful within their own run)."""
    g_waveforms = np.vstack([r["waveforms"] for r in runs])
    if params.get("dct_filter_n", 0) > 0:
        g_waveforms = dct_highpass_waveforms(g_waveforms, params["dct_filter_n"])
        print(f"  DCT high-pass: zeroed the {params['dct_filter_n']} lowest coefficients "
              f"of every waveform")
    g_run_id = np.concatenate([np.full(r["waveforms"].shape[0], i, dtype=int)
                                for i, r in enumerate(runs)])
    g_peaks_t = np.concatenate([r["peaks_t"] for r in runs])
    g_peak_amps = np.concatenate([r["peak_amps"] for r in runs])
    fs = runs[0]["fs"]
    half_win = runs[0]["half_win"]
    total_duration = float(sum(r["analysis_duration"] for r in runs))

    print(f"Pooled {g_waveforms.shape[0]} spikes from {len(runs)} runs -- building "
          f"{params['feature_method']} features (this can take a while for t-SNE)...")
    features = build_features(
        g_waveforms, fs, params["n_components"], params["seed"], params["amp_dur_weight"],
        method=params["feature_method"], wavelet=params["wavelet"],
        wavelet_level=params["wavelet_level"], tsne_perplexity=params["tsne_perplexity"],
        wavelet_clip_percentile=params["wavelet_clip_percentile"],
        polarity_weight=params["polarity_weight"])

    n0 = max(1, int(np.ceil(g_waveforms.shape[0] / (params["max_rate_hz"] * total_duration))))
    n_clusters, sweep = choose_n_clusters(features, n0, params["seed"], params["cluster_method"])

    labels = cluster_spikes(features, n_clusters, params["seed"],
                             method=params["cluster_method"], tmix_dof=params["tmix_dof"],
                             tmix_iters=params["tmix_iters"])

    if (labels == -1).any():
        keep = labels != -1
        g_waveforms, g_run_id, g_peaks_t, g_peak_amps, features, labels = (
            g_waveforms[keep], g_run_id[keep], g_peaks_t[keep], g_peak_amps[keep],
            features[keep], labels[keep])
    labels, n_types = relabel_dense(labels)

    n_isi_dropped = 0
    if params["min_isi_ms"] > 0:
        keep = np.ones(labels.size, dtype=bool)
        for i in range(len(runs)):
            m = np.where(g_run_id == i)[0]
            if m.size == 0:
                continue
            keep[m] = enforce_min_isi_per_type(
                g_peaks_t[m], labels[m], g_peak_amps[m], params["min_isi_ms"] * 1e-3)
        if not keep.all():
            n_isi_dropped = int((~keep).sum())
            g_waveforms, g_run_id, g_peaks_t, g_peak_amps, features, labels = (
                g_waveforms[keep], g_run_id[keep], g_peaks_t[keep], g_peak_amps[keep],
                features[keep], labels[keep])
        labels, n_types = relabel_dense(labels)

    return dict(
        fs=fs, half_win=half_win, total_duration=total_duration,
        features=features, run_id=g_run_id, peaks_t=g_peaks_t, peak_amps=g_peak_amps,
        waveforms=g_waveforms, labels=labels, n_types=n_types,
        n0=n0, n_clusters_used=n_clusters, sweep=sweep, n_isi_dropped=n_isi_dropped,
        feature_method=params["feature_method"],
    )


def make_view(sort, min_spikes):
    """A display copy of `sort` with global types below `min_spikes`
    pooled detections dropped (relabelled to -1, so plotting loops over
    range(n_types) simply skip them) and the survivors renumbered 0..k-1,
    plus per-type pooled metrics/firing rates."""
    counts = np.bincount(sort["labels"], minlength=sort["n_types"])
    keep_types = [c for c in range(sort["n_types"]) if counts[c] >= min_spikes]
    remap = {old: new for new, old in enumerate(keep_types)}
    labels = np.array([remap.get(label, -1) for label in sort["labels"]])
    n_types = len(keep_types)

    v = dict(sort)
    v["labels"] = labels
    v["n_types"] = n_types
    v["n_dropped_types"] = sort["n_types"] - n_types
    v["metrics_by_type"] = {
        c: muap_metrics(sort["waveforms"][labels == c].mean(axis=0), sort["fs"])
        for c in range(n_types)}
    v["firing_rates"] = {c: (labels == c).sum() / sort["total_duration"] for c in range(n_types)}
    return v


# ---------------------------------------------------------------------------
# 3. Plots + data
# ---------------------------------------------------------------------------
def square_grid_shape(n):
    n_side = max(1, int(np.ceil(np.sqrt(max(n, 1)))))
    return n_side, n_side


def plot_rasters(v, runs, out_path):
    n_runs = len(runs)
    fig, axes = plt.subplots(n_runs, 1, figsize=(12, 2.0 * n_runs), squeeze=False)

    for i, r in enumerate(runs):
        ax = axes[i][0]
        in_run = v["run_id"] == i
        for c in range(v["n_types"]):
            mask = in_run & (v["labels"] == c)
            ax.vlines(v["peaks_t"][mask], c - 0.4, c + 0.4,
                      color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)], linewidth=1)
        ax.set_ylim(-0.5, max(v["n_types"] - 0.5, 0.5))
        ax.set_yticks(range(v["n_types"]))
        ax.set_yticklabels([f"type {c}" for c in range(v["n_types"])], fontsize=7)
        ax.set_title(f"{r['label']} — {int(in_run.sum())} sorted spikes over "
                     f"{r['analysis_duration']:.1f} s", fontsize=9)
        if i == n_runs - 1:
            ax.set_xlabel("Time within this run (s) — independent axis per run")

    fig.suptitle("Per-run raster from ONE joint clustering — a colour/row is the same type "
                  "in every run\n(time axes are still independent: runs are separate needle "
                  "recordings)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    print(f"Saved raster plot to {out_path}")
    plt.show()


def plot_muap_gallery_pooled(v, out_path):
    """One panel per global type: every individual trial from every run
    that got that label, superimposed, plus the pooled mean template."""
    n_side, _ = square_grid_shape(v["n_types"])
    fig, axes = plt.subplots(n_side, n_side, figsize=(3.4 * n_side, 3.1 * n_side), squeeze=False)
    rng = np.random.default_rng(0)
    t_wave_ms = (np.arange(-v["half_win"], v["half_win"]) / v["fs"]) * 1000

    for idx in range(n_side * n_side):
        ax = axes[idx // n_side][idx % n_side]
        if idx >= v["n_types"]:
            ax.axis("off")
            continue
        mask = v["labels"] == idx
        unit_waveforms = v["waveforms"][mask]
        mean_wf = unit_waveforms.mean(axis=0)
        m = v["metrics_by_type"][idx]
        n_runs_present = len(np.unique(v["run_id"][mask]))
        title = (f"type {idx} (n={int(mask.sum())} pooled, {n_runs_present} run(s), "
                 f"{v['firing_rates'][idx]:.1f} Hz)\n"
                 f"dur {m['duration_ms']:.2f} ms, amp {m['amplitude']:.0f}, "
                 f"{m['phases']}ph/{m['turns']}t")
        plot_muap_panel(ax, unit_waveforms, mean_wf, m, t_wave_ms,
                         CLUSTER_COLORS[idx % len(CLUSTER_COLORS)], title,
                         muap_ylim=MUAP_YLIM, negative_up=NEGATIVE_UP, rng=rng,
                         ylabel="Amplitude" if idx % n_side == 0 else None)

    fig.suptitle("Pooled MUAP templates — one panel per global type, trials pooled across "
                  "all runs", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    print(f"Saved pooled MUAP gallery to {out_path}")
    plt.show()


def plot_muaps_by_run(v, runs, out_path):
    """type (rows) x run (columns) grid of mean templates -- an
    at-a-glance check of whether each joint type keeps the same shape
    across the different needle placements it was pooled from. Cells with
    no spikes of that type in that run are left blank."""
    n_types, n_runs = v["n_types"], len(runs)
    if n_types == 0:
        print("No surviving types -- skipping type x run grid.")
        return
    fig, axes = plt.subplots(n_types, n_runs, figsize=(2.9 * n_runs, 2.5 * n_types),
                              squeeze=False)
    rng = np.random.default_rng(0)
    t_wave_ms = (np.arange(-v["half_win"], v["half_win"]) / v["fs"]) * 1000

    for c in range(n_types):
        for i, r in enumerate(runs):
            ax = axes[c][i]
            mask = (v["labels"] == c) & (v["run_id"] == i)
            if not mask.any():
                ax.axis("off")
                if c == 0:
                    ax.set_title(f"{r['label']}\n(no type {c})", fontsize=7)
                continue
            unit_waveforms = v["waveforms"][mask]
            mean_wf = unit_waveforms.mean(axis=0)
            m = muap_metrics(mean_wf, v["fs"])
            title = f"{r['label'] if c == 0 else ''}\ntype {c}, n={int(mask.sum())}"
            plot_muap_panel(ax, unit_waveforms, mean_wf, m, t_wave_ms,
                             CLUSTER_COLORS[c % len(CLUSTER_COLORS)], title,
                             muap_ylim=MUAP_YLIM, negative_up=NEGATIVE_UP, rng=rng,
                             ylabel=f"type {c}\nAmplitude" if i == 0 else None)

    fig.suptitle("Joint type (rows) across runs (columns) — is each type's shape consistent "
                  "between needle sites?", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    print(f"Saved type x run MUAP grid to {out_path}")
    plt.show()


def plot_joint_feature_space(v, runs, out_path):
    """The single pooled embedding every spike was clustered in: left
    coloured by joint type (with covariance ellipses), right coloured +
    marker-styled by run so you can see how each run's spikes are spread
    through the shared shape space."""
    method = v["feature_method"]
    fig, (ax_type, ax_run) = plt.subplots(1, 2, figsize=(13, 6.2))

    plot_feature_space_2d(ax_type, v["features"], v["labels"], 0, 1,
                           axis_labels=(f"{method} dim 1", f"{method} dim 2"),
                           title="colour = joint type", legend=True,
                           show_ellipses=FEATURE_SPACE_SHOW_ELLIPSES,
                           ellipse_n_std=FEATURE_SPACE_ELLIPSE_STD)

    for i, r in enumerate(runs):
        mask = v["run_id"] == i
        ax_run.scatter(v["features"][mask, 0], v["features"][mask, 1], s=10, alpha=0.5,
                        color=RUN_CMAP[i % len(RUN_CMAP)],
                        marker=RUN_MARKERS[i % len(RUN_MARKERS)], label=r["label"])
    ax_run.set_xlabel(f"{method} dim 1", fontsize=8)
    ax_run.set_ylabel(f"{method} dim 2", fontsize=8)
    ax_run.set_title("colour/marker = run", fontsize=9)
    ax_run.legend(loc="best", fontsize=7)

    fig.suptitle("Single joint shape-feature embedding — every spike from every run in the "
                  "same space (one fit)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    print(f"Saved joint feature-space plot to {out_path}")
    plt.show()


def plot_silhouette(sort, out_path):
    if not sort["sweep"]:
        print("No silhouette sweep to plot (FORCE_N_CLUSTERS set or sweep disabled).")
        return
    ns = sorted(sort["sweep"])
    scores = [sort["sweep"][n] for n in ns]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ns, scores, marker="o", color=CLUSTER_COLORS[0])
    ax.axvline(sort["n0"], color="0.5", linestyle=":", linewidth=1.5,
               label=f"firing-rate floor n0={sort['n0']}")
    ax.axvline(sort["n_clusters_used"], color="crimson", linestyle="-", linewidth=1.5,
               label=f"used n={sort['n_clusters_used']}")
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("silhouette score (pooled spikes)")
    ax.set_title("Silhouette score vs candidate cluster count — joint pool", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved silhouette-sweep plot to {out_path}")
    plt.show()


def _hist_with_kde(ax, data, bins, hist_color, kde_points=MUAP_METRICS_KDE_POINTS):
    data = np.asarray(data, dtype=float)
    ax.hist(data, bins=bins, density=True, color=hist_color, edgecolor="white", alpha=0.6)
    if data.size >= 2 and np.ptp(data) > 0:
        kde = gaussian_kde(data)
        x = np.linspace(data.min(), data.max(), kde_points)
        ax.plot(x, kde(x), color="black", linewidth=1.5)


def plot_muap_metric_histograms(v, out_path, subject_label, bins=MUAP_METRICS_BINS):
    amps = [v["metrics_by_type"][c]["amplitude"] for c in range(v["n_types"])]
    durs = [v["metrics_by_type"][c]["duration_ms"] for c in range(v["n_types"])]
    if not amps:
        print("No surviving types -- skipping MUAP metric histograms.")
        return

    fig, (ax_amp, ax_dur) = plt.subplots(1, 2, figsize=(9, 4))
    _hist_with_kde(ax_amp, amps, bins, "steelblue")
    ax_amp.set_title("Amplitude (pooled mean-template MUAPs)", fontsize=10)
    ax_amp.set_xlabel("Amplitude (measurement band)")
    ax_amp.set_ylabel("density")
    _hist_with_kde(ax_dur, durs, bins, "indianred")
    ax_dur.set_title("Duration (pooled mean-template MUAPs)", fontsize=10)
    ax_dur.set_xlabel("Duration (ms)")

    fig.suptitle(f"{subject_label} — joint-type amplitude/duration "
                 f"(n={len(amps)} global types)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=150)
    print(f"Saved MUAP amplitude/duration histograms to {out_path}")
    plt.show()


def subject_muscle_side_label():
    parts = os.getcwd().rstrip(os.sep).split(os.sep)
    this_dir = parts[-1]
    muscle_dir = parts[-2] if len(parts) >= 2 else ""
    subject_dir = parts[-4] if len(parts) >= 4 else None
    return f"{subject_dir}/{muscle_dir}_{this_dir}" if subject_dir else f"{muscle_dir}_{this_dir}"


def default_data_path(data_dir):
    cwd = os.getcwd()
    this_dir = os.path.basename(cwd)
    parent_dir = os.path.basename(os.path.dirname(cwd))
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, f"{parent_dir}_{this_dir}_joint_spike_sort_data.npz")


def save_processed_data(sort, runs, out_path):
    """Raw (unfiltered) joint sort. Per-run keys follow the same schema
    muap_post_selection.py already reads ({label}_fs / _wave_time_ms /
    _n_types / _type{c}_spike_times_s / _type{c}_waveforms), except
    _n_types is the GLOBAL type count and a run simply has empty arrays
    for any global type it contributed no spikes to. Pooled per-type
    arrays and the raw global label/run-id vectors are stored alongside."""
    fs, half_win = sort["fs"], sort["half_win"]
    t_wave_ms = (np.arange(-half_win, half_win) / fs) * 1000
    run_labels = [r["label"] for r in runs]

    data = {
        "pooled_fs": fs,
        "pooled_wave_time_ms": t_wave_ms,
        "pooled_n_types": sort["n_types"],
        "run_labels": np.array(run_labels),
        "global_labels": sort["labels"],
        "global_run_id": sort["run_id"],
        "global_spike_times_s": sort["peaks_t"],
    }
    for c in range(sort["n_types"]):
        m = sort["labels"] == c
        data[f"pooled_type{c}_waveforms"] = sort["waveforms"][m]
        data[f"pooled_type{c}_spike_times_s"] = sort["peaks_t"][m]
        data[f"pooled_type{c}_run_id"] = sort["run_id"][m]

    for i, label in enumerate(run_labels):
        data[f"{label}_fs"] = fs
        data[f"{label}_wave_time_ms"] = t_wave_ms
        data[f"{label}_n_types"] = sort["n_types"]
        in_run = sort["run_id"] == i
        for c in range(sort["n_types"]):
            m = in_run & (sort["labels"] == c)
            data[f"{label}_type{c}_spike_times_s"] = sort["peaks_t"][m]
            data[f"{label}_type{c}_waveforms"] = sort["waveforms"][m]

    np.savez_compressed(out_path, **data)
    print(f"Saved joint processed data ({sort['n_types']} global types, "
          f"{sort['labels'].size} spikes) to {out_path}")


def main():
    params = copy.deepcopy(BASE_PARAMS)
    params.update(PARAM_OVERRIDES)

    print(f"Detecting spikes in {len(RUN_FILES)} run(s)...")
    runs = run_detection(RUN_FILES, params)

    sort = joint_cluster(runs, params)
    print(f"Joint clustering: {sort['labels'].size} pooled spikes -> {sort['n_types']} global "
          f"types (n0={sort['n0']}, used n_clusters={sort['n_clusters_used']}"
          + (f", {sort['n_isi_dropped']} dropped by min-ISI" if sort["n_isi_dropped"] else "")
          + ")")

    save_processed_data(sort, runs, default_data_path(OUT_DATA_DIR))

    v = make_view(sort, MIN_SPIKES_PER_TYPE)
    if v["n_dropped_types"]:
        print(f"  dropped {v['n_dropped_types']}/{sort['n_types']} global type(s) with "
              f"< {MIN_SPIKES_PER_TYPE} pooled spikes ({v['n_types']} kept for display)")

    plot_rasters(v, runs, OUT_RASTER)
    plot_muap_gallery_pooled(v, OUT_MUAPS)
    plot_muaps_by_run(v, runs, OUT_MUAPS_BY_RUN)
    plot_joint_feature_space(v, runs, OUT_FEATURE_SPACE)
    plot_silhouette(sort, OUT_SILHOUETTE)
    plot_muap_metric_histograms(v, OUT_MUAP_METRICS, subject_muscle_side_label())


if __name__ == "__main__":
    main()
