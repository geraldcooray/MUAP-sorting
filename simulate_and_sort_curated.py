#!/usr/bin/env python3
"""Simulate EMG traces from the CURATED MUAP library and run spike
estimation on each with three feature methods x two post-processing
variants.

Phase 1 (feature method 'tsne' only, run once) -- trace synthesis
(simulate_emg_traces.build_trace), curated library, N_MUAPS in
{3, 5, 10, 15, 20, 30}, drawn at random. Each trace is 120 s of active
signal + 1 s spike-free lead-in, pink background noise at 30 dB SNR.
Saved to simulated_traces/tsne/simulated_trace_n<K>.npz (+ .png). This
SAME trace set (and the SAME EMGRUN text curves under
simulated_traces/spikesort_input/) is reused as-is for every feature
method below -- nothing is ever re-simulated, so pca/wavelet/tsne runs
are sorting identical data.

Phase 2 -- for every trace, sort it with --feature-method in
{tsne, pca, wavelet} and each of k-means / GMM / Student's-t mixture,
under two post-processing settings:

    full     -- min-ISI cleanup + latency-linkage merge + superposition
                detection (flagged composite types removed)
    reduced  -- none of those three steps

Feature config matches the library build (2 components,
amp_dur_weight = polarity_weight = 0, no firing-regularity refinement).
Detection uses the 100 Hz high-pass / high MAD threshold / baseline-only
noise estimate used for simulated traces.

For each (trace x method x post-processing) run, one figure is written
with the TRUE-unit MUAP gallery on top and the ESTIMATED-type gallery
below (mean of the raw trace over the relevant time points, x = +/-10 ms,
y = -1000..1000 uV), plus an .npz with the estimated spike data. Both go
to simulated_traces/<feature-method>/. A run-level summary is written to
simulated_traces/<feature-method>/sort_summary.csv.

Usage:
    python simulate_and_sort_curated.py --phase traces        # once, tsne only
    python simulate_and_sort_curated.py --feature-method pca
    python simulate_and_sort_curated.py --feature-method wavelet
"""
import argparse
import copy
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import call_spike_sort_emg
import simulate_emg_traces as sim
import spikesort_simulated_traces as sst
from call_spike_sort_emg import run_pipeline
from spike_sort.clustering import cluster_spikes, relabel_dense
from spike_sort.detection import bandpass, extract_waveforms
from spike_sort.metrics import muap_metrics
from spike_sort.plotting import plot_muap_panel
from spike_sort.superposition import find_superposition_candidates

HERE = os.path.dirname(os.path.abspath(__file__))
CURATED_LIBRARY = os.path.join(HERE, "muap_library_curated", "muap_library.npz")
SIM_ROOT = os.path.join(HERE, "simulated_traces")
TRACE_DIR = os.path.join(SIM_ROOT, "tsne")  # traces are generated once (tsne run) and REUSED
                                             # as-is for every feature method -- never regenerated,
                                             # so all methods sort exactly the same curves.

# Set by main() from --feature-method; sort_* results/figures land in simulated_traces/<method>/
FEATURE_METHOD = "tsne"
OUT_DIR = os.path.join(SIM_ROOT, FEATURE_METHOD)

N_MUAPS_LIST = [3, 5, 10, 15, 20, 30]
DURATION_S = 120.0
BASELINE_S = 1.0
SNR_DB = 30.0
SEED = 42

CLUSTER_METHODS = ["kmeans", "gmm", "tmixture"]
POSTPROC = ["full", "reduced"]
N_COMPONENTS = 2  # shape features kept -- PCA components / t-SNE dims / top wavelet coeffs;
                   # same value for all three feature methods so runs are otherwise comparable

GALLERY_HALF_MS = 10.0            # +/- window (ms) for every gallery panel
GALLERY_YLIM = (-1000.0, 1000.0)  # fixed amplitude range (uV) for every gallery panel
GALLERY_NCOLS = 8
MATCH_TOL_MS = 3.0                # detected spike within this of a true spike counts as a match
SUPERPOSITION_MAX_SHIFT_MS = 3.0


# --------------------------------------------------------------------------- #
# Speed: score the silhouette sweep on a random subsample rather than every
# spike (call_spike_sort_emg's version is O(n_spikes^2) per candidate k).
# --------------------------------------------------------------------------- #
def _subsampled_silhouette_sweep(features, n_values, seed, method="kmeans", **cluster_kwargs):
    from sklearn.metrics import silhouette_score
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    idx = rng.choice(n, 4000, replace=False) if n > 4000 else np.arange(n)
    scores = {}
    for k in n_values:
        if k < 2 or k >= n:
            scores[k] = float("nan")
            continue
        labels = cluster_spikes(features, k, seed, method=method, **cluster_kwargs)
        sub_lab = labels[idx]
        keep = sub_lab != -1
        if len(set(sub_lab[keep].tolist())) < 2:
            scores[k] = float("nan")
            continue
        scores[k] = float(silhouette_score(features[idx][keep], sub_lab[keep]))
    return scores


call_spike_sort_emg.silhouette_sweep = _subsampled_silhouette_sweep


# --------------------------------------------------------------------------- #
def load_curated_library():
    sim.LIBRARY_PATH = CURATED_LIBRARY
    library = sim.load_library(CURATED_LIBRARY)          # respects MIN_TRIALS
    lookup = {(e["subject"], e["type_idx"]): e for e in library}
    return library, lookup


def make_traces(library):
    """Generate the shared trace set into TRACE_DIR. Only meant to be run
    once (feature method 'tsne'); every other feature method reads these
    same files back via sort_all -- never regenerates them."""
    os.makedirs(TRACE_DIR, exist_ok=True)
    fs = library[0]["fs"]
    manifest = []
    for n in N_MUAPS_LIST:
        rng = np.random.default_rng(SEED + n)
        trace, clean, noise, t, unit_records = sim.build_trace(
            n, library, fs, DURATION_S, SNR_DB, rng, BASELINE_S)
        stem = os.path.join(TRACE_DIR, f"simulated_trace_n{n}")
        sim.plot_trace(t, trace, unit_records, n, SNR_DB, BASELINE_S, stem + ".png")
        sim.save_trace_npz(t, trace, clean, noise, fs, unit_records, BASELINE_S, stem + ".npz")
        for u in unit_records:
            manifest.append(dict(n_muaps=n, subject=u["subject"], type_idx=u["type_idx"],
                                  category=u["category"],
                                  assigned_firing_rate_hz=round(u["assigned_firing_rate_hz"], 3),
                                  n_spikes=u["n_spikes"]))
        print(f"n={n}: {len(unit_records)} units, "
              f"{sum(u['n_spikes'] for u in unit_records)} true spikes")
    with open(os.path.join(TRACE_DIR, "simulated_traces_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)


# --------------------------------------------------------------------------- #
def mean_waveform(measured, times_s, fs, half_ms):
    """Mean +/-half_ms snippet of `measured` at the given spike times, plus
    the trial stack, the trial count, and the time axis (ms)."""
    half = int(round(half_ms * 1e-3 * fs))
    pk = np.rint(times_s * fs).astype(int)
    _, wf, half_win = extract_waveforms(measured, pk, fs, half_ms, half_ms)
    t_ms = (np.arange(-half_win, half_win) / fs) * 1000.0
    if wf.shape[0] == 0:
        return np.zeros(2 * half), wf, 0, t_ms
    return wf.mean(axis=0), wf, wf.shape[0], t_ms


def drop_superposition_types(r, fs):
    """Return (labels, n_types, dropped_ids). Flags estimated types that
    are the time-shifted sum of two others and removes them."""
    labels, n_types = r["labels"], r["n_types"]
    if n_types < 3:
        return labels, n_types, []
    templates = np.stack([r["waveforms"][labels == c].mean(axis=0) for c in range(n_types)])
    max_shift = int(round(SUPERPOSITION_MAX_SHIFT_MS * 1e-3 * fs))
    cands = find_superposition_candidates(templates, max_shift)
    dropped = sorted({c["type_c"] for c in cands})
    if not dropped:
        return labels, n_types, []
    keep = ~np.isin(labels, dropped)
    new_labels = np.full(len(labels), -1)
    new_labels[keep], n_new = relabel_dense(labels[keep])
    return new_labels, n_new, dropped


def gallery_figure(true_units, est, measured, fs, title, out_path):
    """One figure: TRUE-unit gallery (top) + ESTIMATED-type gallery
    (bottom). est = dict(peaks_t, labels, n_types, est_match, est_purity)."""
    n_true = len(true_units)
    n_est = est["n_types"]

    def grid(n):
        ncols = min(GALLERY_NCOLS, max(n, 1))
        return int(np.ceil(max(n, 1) / ncols)), ncols

    tr_rows, tr_cols = grid(n_true)
    es_rows, es_cols = grid(n_est)

    fig = plt.figure(figsize=(3.1 * max(tr_cols, es_cols),
                               3.3 * (tr_rows + es_rows) + 3.2))
    sub_true, sub_est = fig.subfigures(2, 1, height_ratios=[tr_rows + 0.6, es_rows + 0.6])

    ax_true = sub_true.subplots(tr_rows, tr_cols, squeeze=False)
    sub_true.suptitle(f"TRUE MUAPs (n={n_true} units)", fontsize=11, y=0.92)
    sub_true.subplots_adjust(top=0.80, bottom=0.11, hspace=0.85, wspace=0.32)
    for k in range(tr_rows * tr_cols):
        ax = ax_true[k // tr_cols][k % tr_cols]
        if k >= n_true:
            ax.axis("off")
            continue
        u = true_units[k]
        mean_wf, trials, n_tr, t_ms = mean_waveform(measured, u["spike_times"], fs, GALLERY_HALF_MS)
        m = muap_metrics(mean_wf, fs)
        plot_muap_panel(ax, trials if n_tr else mean_wf[None, :], mean_wf, m, t_ms,
                         f"C{k % 10}", f"true {k}: {u['subject']} t{u['type_idx']}  "
                         f"n={n_tr}, {m['duration_ms']:.1f} ms",
                         muap_ylim=GALLERY_YLIM, muap_xlim=(-GALLERY_HALF_MS, GALLERY_HALF_MS),
                         ylabel="uV" if k % tr_cols == 0 else None)

    ax_est = sub_est.subplots(es_rows, es_cols, squeeze=False)
    sub_est.suptitle(f"ESTIMATED types (n={n_est})", fontsize=11, y=0.97)
    sub_est.subplots_adjust(top=0.84, bottom=0.11, hspace=0.85, wspace=0.32)
    for k in range(es_rows * es_cols):
        ax = ax_est[k // es_cols][k % es_cols]
        if k >= n_est:
            ax.axis("off")
            continue
        times = est["peaks_t"][est["labels"] == k]
        mean_wf, trials, n_tr, t_ms = mean_waveform(measured, times, fs, GALLERY_HALF_MS)
        m = muap_metrics(mean_wf, fs)
        match, purity = est["est_match"][k], est["est_purity"][k]
        tag = f"-> true {match} ({purity:.0%})" if match >= 0 else "-> unmatched"
        plot_muap_panel(ax, trials if n_tr else mean_wf[None, :], mean_wf, m, t_ms,
                         f"C{(match if match >= 0 else k) % 10}",
                         f"est {k} {tag}  n={n_tr}, {m['duration_ms']:.1f} ms",
                         muap_ylim=GALLERY_YLIM, muap_xlim=(-GALLERY_HALF_MS, GALLERY_HALF_MS),
                         ylabel="uV" if k % es_cols == 0 else None)

    fig.suptitle(title, fontsize=12, y=1.005)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def sort_all(lookup):
    # Optional subsetting for quick checks: SORT_N=3,5  SORT_METHODS=kmeans  SORT_PP=reduced
    n_list = [int(x) for x in os.environ["SORT_N"].split(",")] if os.environ.get("SORT_N") \
        else N_MUAPS_LIST
    methods = os.environ["SORT_METHODS"].split(",") if os.environ.get("SORT_METHODS") \
        else CLUSTER_METHODS
    pps = os.environ["SORT_PP"].split(",") if os.environ.get("SORT_PP") else POSTPROC

    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []
    for n in n_list:
        # Always the SAME trace file (generated once under TRACE_DIR/tsne) -- every feature
        # method sorts identical curves, never a freshly-simulated one.
        npz_path = os.path.join(TRACE_DIR, f"simulated_trace_n{n}.npz")
        trace, fs, true_units, baseline_s = sst.load_simulated_trace(npz_path, lookup)
        emg_name = f"EMGRUN__curated_n{n}.txt"
        emg_txt = os.path.join(sst.SPIKESORT_INPUT_DIR, emg_name)
        if not os.path.isfile(emg_txt):  # reuse the exact same on-disk curve across methods
            emg_txt = sst.write_emgrun_file(trace, fs, sst.SPIKESORT_INPUT_DIR, emg_name)
        noise_baseline_samples = int(round(baseline_s * fs)) if baseline_s > 0 else None
        measured = bandpass(trace, fs, 20.0, 10000.0)
        call_spike_sort_emg.SILHOUETTE_MAX_N = n + 10

        for method in methods:
            for pp in pps:
                label = f"n{n}_{method}_{pp}"
                overrides = dict(cluster_method=method, amp_dur_weight=0.0,
                                  polarity_weight=0.0, isi_weight=0.0, isi_iters=0)
                if pp == "full":
                    overrides.update(min_isi_ms=10.0, link_max_latency_ms=8.0,
                                      link_min_co_occurrence=0.6, link_max_latency_std_ms=1.0)
                else:
                    overrides.update(min_isi_ms=0.0, link_max_latency_ms=0.0)

                r = sst.sort_simulated_trace(
                    trace, fs, emg_txt, label, feature_method=FEATURE_METHOD,
                    n_components=N_COMPONENTS, cluster_method=method,
                    noise_baseline_samples=noise_baseline_samples, param_overrides=overrides)

                dropped = []
                labels, n_types = r["labels"], r["n_types"]
                if pp == "full":
                    labels, n_types, dropped = drop_superposition_types(r, fs)
                    r = dict(r, labels=labels, n_types=n_types)

                matched_unit = sst.match_to_ground_truth(
                    r["peaks_t"], true_units, MATCH_TOL_MS / 1000.0)
                est_match, est_purity = sst.compute_match_purity(
                    true_units, r["labels"], matched_unit)
                est = dict(peaks_t=r["peaks_t"], labels=r["labels"], n_types=n_types,
                           est_match=est_match, est_purity=est_purity)

                title = (f"Simulated trace, n={n} MUAPs  |  {FEATURE_METHOD} + {method}  |  "
                         f"{pp} post-processing"
                         + (f"  (superposition-dropped: {dropped})" if dropped else ""))
                png = os.path.join(OUT_DIR, f"sort_{label}.png")
                gallery_figure(true_units, est, measured, fs, title, png)

                out_npz = os.path.join(OUT_DIR, f"sort_{label}.npz")
                np.savez_compressed(
                    out_npz,
                    peaks_t=r["peaks_t"], labels=r["labels"], waveforms=r["waveforms"],
                    fs=fs, n_types=n_types, n_muaps=n, cluster_method=method,
                    postprocessing=pp, feature_method=FEATURE_METHOD,
                    est_match=np.array(est_match), est_purity=np.array(est_purity),
                    matched_unit=matched_unit,
                    true_unit_names=np.array([u["name"] for u in true_units]),
                    superposition_dropped=np.array(dropped, dtype=int),
                    match_tol_ms=MATCH_TOL_MS)

                n_recovered = len({m for m in est_match if m >= 0})
                mean_purity = float(np.mean([p for p in est_purity])) if est_purity else 0.0
                summary.append(dict(
                    n_muaps=n, feature_method=FEATURE_METHOD, method=method, postprocessing=pp,
                    n_detected=int(r["peaks_t"].size), n_types_est=n_types, n_types_true=n,
                    n_true_recovered=n_recovered, mean_type_purity=round(mean_purity, 3),
                    n_superposition_dropped=len(dropped)))
                print(f"  [{label}] detected {r['peaks_t'].size}, {n_types} types, "
                      f"{n_recovered}/{n} true units recovered, "
                      f"mean purity {mean_purity:.2f}"
                      + (f", dropped {dropped}" if dropped else ""))

    with open(os.path.join(OUT_DIR, "sort_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nWrote {len(summary)} runs -> {os.path.join(OUT_DIR, 'sort_summary.csv')}")


# --------------------------------------------------------------------------- #
def main():
    global FEATURE_METHOD, OUT_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "traces", "sort"], default="sort",
                     help="'traces' (re)generates the shared trace set under "
                          "simulated_traces/tsne/ -- normally run once. 'sort' (default) "
                          "reads that existing trace set and only runs the sort sweep, "
                          "writing to simulated_traces/<feature-method>/.")
    ap.add_argument("--feature-method", choices=["tsne", "pca", "wavelet"], default="tsne")
    args = ap.parse_args()

    FEATURE_METHOD = args.feature_method
    OUT_DIR = os.path.join(SIM_ROOT, FEATURE_METHOD)

    library, lookup = load_curated_library()
    print(f"Curated library: {len(library)} templates (>= {sim.MIN_TRIALS} trials each)")
    print(f"Feature method: {FEATURE_METHOD}  |  trace source: {TRACE_DIR}  |  "
          f"output: {OUT_DIR}")

    if args.phase == "traces":
        make_traces(library)
        return
    if args.phase == "all":
        make_traces(library)
    sort_all(lookup)


if __name__ == "__main__":
    main()
