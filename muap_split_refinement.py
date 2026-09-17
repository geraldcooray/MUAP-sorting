#!/usr/bin/env python3
"""Optional SECOND post-processing step: per-type sub-clustering split,
then an L2 shape merge -- run on top of an already-completed first-pass
sort (full min-ISI + latency-linkage + superposition post-processing).

Step 2a (split): for each estimated type, look at its own individual
+/-5 ms trial waveforms -- the same window feature extraction/clustering
used -- and ask whether they actually form ONE cluster or TWO: re-embed
just that type's trials in 2D with a fresh t-SNE, split them with
k-means (k=2), and score the split with the silhouette coefficient. If
the split is well separated (>= SPLIT_SILHOUETTE_MIN), the type is
treated as two genuinely distinct MUAPs that the first pass merged and
is split into two new types.

Step 2b (L2 merge): on the resulting type set, compute the L2
(Euclidean) distance between every pair of types' mean waveforms
(spike_sort.l2_merge), each CROPPED to +/-L2_WINDOW_MS (default 3 ms,
around the detected peak -- the sharp core, not the noisier tails of the
wider +/-5 ms clustering window) and, if L2_NORMALIZE (default True),
RESCALED to unit L1 area first (spike_sort.l2_merge.normalize_area) so
the comparison is of shape alone, blind to overall amplitude. Pairs
closer than a threshold chosen by the robust OUTLIER-FENCE method --
median(distances) - L2_FENCE_K * MAD(distances)
(spike_sort.l2_merge.outlier_fence_threshold) -- are merged. This is the
module's own best-validated automatic threshold shape (a few genuine
duplicate-type pairs sit as low outliers below a broad spread of
genuinely-different-type distances) -- unlike the elbow method (largest
gap in the sorted distance curve), which was tried first here and found
to collapse almost every type into one on 4 of 6 simulated traces,
matching spike_sort.l2_merge's own recorded validation history.

This is a standalone, on-demand diagnostic (not wired into the main
sweep). Never re-simulates or re-sorts; only re-examines the
already-saved per-spike labels/waveforms of a completed run.

For each n in the batch, one figure is written with three rows: TRUE
units, the first-pass (BEFORE) estimated types, and the (AFTER) types
following the split + L2 merge -- BEFORE/AFTER panels are annotated with
each type's best-matching true unit and PURITY (%).

Usage:
    python muap_split_refinement.py                          # batch: n in
        # {3,5,10,15,20,30}, tsne + kmeans + full (as requested)
    python muap_split_refinement.py --n 20                    # single n
    python muap_split_refinement.py --n 20 --method gmm --postproc reduced
        --feature-method pca --silhouette-min 0.6 --l2-fence-k 2.0
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score

import simulate_and_sort_curated as base
import spikesort_simulated_traces as sst
from spike_sort.clustering import cluster_spikes
from spike_sort.features import shape_features_tsne
from spike_sort.l2_merge import (find_close_type_pairs, group_close_types, merge_close_types,
                                  outlier_fence_threshold, pairwise_l2_distances)

BATCH_N = [3, 5, 10, 15, 20, 30]
DEFAULT_METHOD = "kmeans"
DEFAULT_POSTPROC = "full"
DEFAULT_FEATURE_METHOD = "tsne"

SPLIT_SEED = 42
SPLIT_MIN_SPIKES = 20          # skip types with fewer trials than this -- too few for a
                                # meaningful re-embedding/split
SPLIT_TSNE_PERPLEXITY = 30.0
SPLIT_SILHOUETTE_MIN = 0.5     # keep the 2-way k-means split only if its silhouette score
                                # is at least this -- otherwise the type is left as one

L2_FENCE_K = 3.0                # outlier-fence L2-merge threshold = median(distances) -
                                 # L2_FENCE_K * MAD(distances). spike_sort.l2_merge's own
                                 # module default is k=2.0 (validated on the un-split library
                                 # build), but on these post-split type sets the pairwise-L2
                                 # distribution is a smooth, roughly continuous spread with NO
                                 # true near-duplicate cluster near zero -- k=2's fence still
                                 # sits inside that continuum (catches the lower tail, then
                                 # union-find chains 4-8 unrelated types into one merge group).
                                 # k=3 pushes the fence below every observed distance in all 6
                                 # traces tested, so it merges only genuine near-duplicates
                                 # (very small L2 difference) and correctly merges nothing when
                                 # none exist -- see the printed L2-distance stats each run.

L2_WINDOW_MS = 3.0               # the L2 comparison itself is restricted to +/-L2_WINDOW_MS
                                  # around the detected peak (cropped from the wider +/-5 ms
                                  # clustering waveform), so the distance reflects the MUAP's
                                  # sharp core rather than its noisier, more variable tails.
L2_NORMALIZE = True              # rescale each type's (cropped) mean waveform to unit L1 area
                                  # before computing L2 (spike_sort.l2_merge.normalize_area) --
                                  # compares SHAPE only, blind to overall amplitude, so two
                                  # types differing mainly in gain (e.g. needle distance) can
                                  # still register as close.


# --------------------------------------------------------------------------- #
# Step 2a: per-type split
# --------------------------------------------------------------------------- #
def try_split_type(waveforms_c, seed=SPLIT_SEED):
    """Re-embed one type's own +/-5 ms trial waveforms in 2D (t-SNE) and
    split with k-means (k=2). Returns (did_split, sub_labels_or_None,
    silhouette)."""
    n = waveforms_c.shape[0]
    if n < SPLIT_MIN_SPIKES:
        return False, None, float("nan")
    feats = shape_features_tsne(waveforms_c, 2, seed, perplexity=SPLIT_TSNE_PERPLEXITY)
    sub = cluster_spikes(feats, 2, seed, method="kmeans")
    if len(set(sub.tolist())) < 2:
        return False, None, float("nan")
    sil = float(silhouette_score(feats, sub))
    return sil >= SPLIT_SILHOUETTE_MIN, sub, sil


def apply_split(labels, waveforms, n_types):
    """Try try_split_type on every current type. Returns (new_labels
    (not yet dense), report rows)."""
    new_labels = labels.copy()
    next_id = n_types
    report = []
    for c in range(n_types):
        idx = np.where(labels == c)[0]
        did_split, sub, sil = try_split_type(waveforms[idx])
        if did_split:
            child_idx = idx[sub == 1]
            new_labels[child_idx] = next_id
            report.append(dict(type=c, n=idx.size, split=True, silhouette=round(sil, 3),
                                n_child0=int((sub == 0).sum()), n_child1=int((sub == 1).sum()),
                                new_type_id=next_id))
            next_id += 1
        else:
            report.append(dict(type=c, n=idx.size, split=False,
                                silhouette=(round(sil, 3) if sil == sil else "n/a")))
    return new_labels, report


def relabel_dense_tracked(labels, tag_ids=frozenset()):
    """Same convention as spike_sort.clustering.relabel_dense, but also
    returns which final ids came from a pre-relabel id in tag_ids."""
    used = sorted(set(labels.tolist()))
    remap = {old: new for new, old in enumerate(used)}
    final = np.array([remap[l] for l in labels])
    tagged_final = {remap[old] for old in used if old in tag_ids}
    return final, len(used), tagged_final


# --------------------------------------------------------------------------- #
# Step 2b: L2 shape merge (robust outlier-fence threshold)
# --------------------------------------------------------------------------- #
def crop_for_l2(waveforms, fs, window_ms=L2_WINDOW_MS):
    """Crop the wider +/-5 ms clustering waveforms down to +/-window_ms
    around the detected peak (assumed centered), for the L2 comparison
    only -- shape/labels elsewhere are untouched."""
    full_half = waveforms.shape[1] // 2
    tight_half = min(int(round(window_ms * 1e-3 * fs)), full_half)
    return waveforms[:, full_half - tight_half: full_half + tight_half]


def apply_l2_merge(labels, waveforms, n_types, fs):
    """L2 shape-merge on current types' mean waveforms, CROPPED to
    +/-L2_WINDOW_MS and (if L2_NORMALIZE) rescaled to unit L1 area before
    the distance is computed, with a threshold chosen by the outlier-
    fence method (median - L2_FENCE_K * MAD) -- i.e. only pairs whose
    distance is a low OUTLIER against the rest of the distribution
    (genuinely close duplicates), not just "below-median". Returns
    (new_labels (not yet dense), threshold, close_pairs, groups,
    distance_stats) where distance_stats = (n_pairs, min, median, MAD) of
    the pairwise L2 distances, for diagnostics."""
    if n_types < 3:
        return labels.copy(), float("nan"), [], [], None
    l2_waveforms = crop_for_l2(waveforms, fs, window_ms=L2_WINDOW_MS)
    l2_distances = pairwise_l2_distances(l2_waveforms, labels, normalize=L2_NORMALIZE)
    values = np.sort(np.fromiter(l2_distances.values(), dtype=float))
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    stats = (values.size, float(values.min()), median, mad)
    threshold = outlier_fence_threshold(l2_distances, k=L2_FENCE_K)
    if threshold != threshold:  # NaN -- fewer than 3 types worth of pairs
        return labels.copy(), threshold, [], [], stats
    close_pairs = find_close_type_pairs(l2_waveforms, labels, threshold, normalize=L2_NORMALIZE)
    groups = group_close_types(close_pairs)
    if not groups:
        return labels.copy(), threshold, close_pairs, groups, stats
    return merge_close_types(labels, groups), threshold, close_pairs, groups, stats


# --------------------------------------------------------------------------- #
def _panel(ax, times, measured, fs, color, title_text, col_idx, ncols):
    mean_wf, trials, n_tr, t_ms = base.mean_waveform(measured, times, fs, base.GALLERY_HALF_MS)
    m = base.muap_metrics(mean_wf, fs)
    base.plot_muap_panel(ax, trials if n_tr else mean_wf[None, :], mean_wf, m, t_ms, color,
                          title_text, muap_ylim=base.GALLERY_YLIM,
                          muap_xlim=(-base.GALLERY_HALF_MS, base.GALLERY_HALF_MS),
                          ylabel="uV" if col_idx == 0 else None)
    return n_tr, m


def _draw_row(subfig, n, get_panel, suptitle):
    ncols = min(base.GALLERY_NCOLS, max(n, 1))
    rows = int(np.ceil(max(n, 1) / ncols))
    ax = subfig.subplots(rows, ncols, squeeze=False)
    subfig.suptitle(suptitle, fontsize=11, y=0.92)
    subfig.subplots_adjust(top=0.80, bottom=0.11, hspace=0.85, wspace=0.32)
    for k in range(rows * ncols):
        a = ax[k // ncols][k % ncols]
        if k >= n:
            a.axis("off")
            continue
        times, color, title_text = get_panel(k)
        _panel(a, times, get_panel.measured, get_panel.fs, color, title_text, k % ncols, ncols)
    return rows


def three_row_gallery(true_units, measured, fs, peaks_t, before_labels, before_match,
                       before_purity, after_labels, after_match, after_purity, title, out_path):
    n_true, n_before, n_after = len(true_units), int(before_labels.max()) + 1, \
        int(after_labels.max()) + 1

    def grid(n):
        ncols = min(base.GALLERY_NCOLS, max(n, 1))
        return int(np.ceil(max(n, 1) / ncols))

    r_true, r_before, r_after = grid(n_true), grid(n_before), grid(n_after)
    fig = plt.figure(figsize=(3.1 * base.GALLERY_NCOLS, 3.3 * (r_true + r_before + r_after) + 4.6))
    sub_true, sub_before, sub_after = fig.subfigures(
        3, 1, height_ratios=[r_true + 0.6, r_before + 0.6, r_after + 0.6])

    def true_panel(k):
        u = true_units[k]
        return u["spike_times"], f"C{k % 10}", f"true {k}: {u['subject']} t{u['type_idx']}"
    true_panel.measured, true_panel.fs = measured, fs

    def est_panel(labels, match, purity):
        def fn(k):
            tag = f"-> true {match[k]} ({purity[k]:.0%})" if match[k] >= 0 else "-> unmatched"
            return (peaks_t[labels == k], f"C{(match[k] if match[k] >= 0 else k) % 10}",
                    f"est {k} {tag}")
        fn.measured, fn.fs = measured, fs
        return fn

    _draw_row(sub_true, n_true, true_panel, f"TRUE MUAPs (n={n_true} units)")
    _draw_row(sub_before, n_before, est_panel(before_labels, before_match, before_purity),
               f"BEFORE 2nd-stage (n={n_before} types, mean purity "
               f"{np.mean(before_purity):.0%})")
    _draw_row(sub_after, n_after, est_panel(after_labels, after_match, after_purity),
               f"AFTER split + L2 merge (n={n_after} types, mean purity "
               f"{np.mean(after_purity):.0%})")

    fig.suptitle(title, fontsize=12, y=1.003)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def run_one(n, method, postproc, feature_method):
    run_dir = os.path.join(base.SIM_ROOT, feature_method)
    run_label = f"n{n}_{method}_{postproc}"
    source_npz = os.path.join(run_dir, f"sort_{run_label}.npz")
    trace_npz = os.path.join(base.TRACE_DIR, f"simulated_trace_n{n}.npz")
    out_stem = os.path.join(run_dir, f"l2split_{run_label}")

    d = np.load(source_npz, allow_pickle=True)
    peaks_t, labels, waveforms = d["peaks_t"], d["labels"], d["waveforms"]
    fs, n_types = float(d["fs"]), int(d["n_types"])
    print(f"\n=== n={n} ({feature_method}+{method}, {postproc}) === "
          f"{n_types} types, {peaks_t.size} spikes")

    # -- 2a: per-type split --
    split_labels_raw, report = apply_split(labels, waveforms, n_types)
    n_split = sum(1 for row in report if row["split"])
    split_labels, n_after_split, _ = relabel_dense_tracked(split_labels_raw)
    for row in report:
        if row["split"]:
            print(f"  split: type {row['type']} (n={row['n']}, silhouette={row['silhouette']}) "
                  f"-> {row['n_child0']}/{row['n_child1']}")

    # -- 2b: L2 merge (outlier-fence threshold) on the post-split type set --
    merged_labels_raw, l2_threshold, close_pairs, groups, l2_stats = apply_l2_merge(
        split_labels, waveforms, n_after_split, fs)
    final_labels, n_final, _ = relabel_dense_tracked(merged_labels_raw)
    if l2_stats:
        n_pairs, dmin, dmed, dmad = l2_stats
        fence_txt = f"{l2_threshold:.4g}" if l2_threshold == l2_threshold else "n/a"
        print(f"  L2 distances (+/-{L2_WINDOW_MS:g} ms, "
              f"{'rescaled' if L2_NORMALIZE else 'raw'}): {n_pairs} pairs, min={dmin:.4g}, "
              f"median={dmed:.4g}, MAD={dmad:.4g}  ->  fence(k={L2_FENCE_K:g})={fence_txt}")
    if groups:
        for g in groups:
            print(f"  L2 merge (outlier-fence threshold={l2_threshold:.4g}): types {g} -> one")
    else:
        thr_txt = f"{l2_threshold:.4g}" if l2_threshold == l2_threshold else "n/a (<3 types)"
        print(f"  L2 merge: no pairs closer than the outlier-fence threshold ({thr_txt})")

    print(f"  {n_types} -> {n_after_split} (split) -> {n_final} (L2 merge) types")

    # -- re-score against ground truth --
    library, lookup = base.load_curated_library()
    trace, fs_t, true_units, baseline_s = sst.load_simulated_trace(trace_npz, lookup)
    matched_unit = sst.match_to_ground_truth(peaks_t, true_units, base.MATCH_TOL_MS / 1000.0)
    before_match, before_purity = sst.compute_match_purity(true_units, labels, matched_unit)
    after_match, after_purity = sst.compute_match_purity(true_units, final_labels, matched_unit)
    before_recovered = len({m for m in before_match if m >= 0})
    after_recovered = len({m for m in after_match if m >= 0})
    print(f"  purity {np.mean(before_purity):.0%} -> {np.mean(after_purity):.0%}  |  "
          f"recovered {before_recovered} -> {after_recovered}/{len(true_units)}")

    measured = base.bandpass(trace, fs, 20.0, 10000.0)
    thr_txt = f"{l2_threshold:.4g}" if l2_threshold == l2_threshold else "n/a"
    title = (f"n={n} MUAPs | {feature_method} + {method} | {postproc}  ->  2nd stage: "
             f"{n_split} split, {len(groups)} L2-merge group(s) (fence thr={thr_txt}, "
             f"k={L2_FENCE_K:g})\n"
             f"types {n_types} -> {n_final}   |   purity {np.mean(before_purity):.0%} -> "
             f"{np.mean(after_purity):.0%}   |   recovered {before_recovered} -> "
             f"{after_recovered}/{len(true_units)}")
    three_row_gallery(true_units, measured, fs, peaks_t, labels, before_match, before_purity,
                       final_labels, after_match, after_purity, title, out_stem + ".png")
    print(f"  wrote {out_stem}.png")

    with open(out_stem + "_report.csv", "w", newline="") as f:
        fields = ["type", "n", "split", "silhouette", "n_child0", "n_child1", "new_type_id"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in report:
            w.writerow({k: row.get(k, "") for k in fields})

    np.savez_compressed(
        out_stem + ".npz", peaks_t=peaks_t, labels_before=labels, labels_after=final_labels,
        fs=fs, n_types_before=n_types, n_types_after_split=n_after_split,
        n_types_after=n_final, l2_threshold=l2_threshold,
        est_match_before=np.array(before_match), est_purity_before=np.array(before_purity),
        est_match_after=np.array(after_match), est_purity_after=np.array(after_purity),
        split_silhouette_min=SPLIT_SILHOUETTE_MIN, split_min_spikes=SPLIT_MIN_SPIKES)

    return dict(n_muaps=n, feature_method=feature_method, method=method, postproc=postproc,
                n_types_before=n_types, n_types_after_split=n_after_split, n_types_after=n_final,
                n_split=n_split, n_l2_merge_groups=len(groups),
                l2_threshold=round(l2_threshold, 2) if l2_threshold == l2_threshold else "",
                mean_purity_before=round(float(np.mean(before_purity)), 3),
                mean_purity_after=round(float(np.mean(after_purity)), 3),
                n_true=len(true_units), n_recovered_before=before_recovered,
                n_recovered_after=after_recovered)


def main():
    global SPLIT_MIN_SPIKES, SPLIT_SILHOUETTE_MIN, L2_FENCE_K, L2_WINDOW_MS, L2_NORMALIZE

    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=None,
                     help=f"single n_muaps to run; omit to run the batch {BATCH_N}")
    ap.add_argument("--method", default=DEFAULT_METHOD, choices=["kmeans", "gmm", "tmixture"])
    ap.add_argument("--postproc", default=DEFAULT_POSTPROC, choices=["full", "reduced"])
    ap.add_argument("--feature-method", default=DEFAULT_FEATURE_METHOD,
                     choices=["tsne", "pca", "wavelet"])
    ap.add_argument("--silhouette-min", type=float, default=SPLIT_SILHOUETTE_MIN)
    ap.add_argument("--min-spikes", type=int, default=SPLIT_MIN_SPIKES)
    ap.add_argument("--l2-fence-k", type=float, default=L2_FENCE_K,
                     help="outlier-fence L2-merge threshold = median - k*MAD; lower = merges "
                          "more aggressively")
    ap.add_argument("--l2-window-ms", type=float, default=L2_WINDOW_MS,
                     help="half-window (ms) the L2 comparison is cropped to")
    ap.add_argument("--l2-no-normalize", action="store_true",
                     help="compare raw (unrescaled) cropped waveforms instead of unit-L1-area "
                          "rescaled ones")
    args = ap.parse_args()
    SPLIT_SILHOUETTE_MIN = args.silhouette_min
    SPLIT_MIN_SPIKES = args.min_spikes
    L2_FENCE_K = args.l2_fence_k
    L2_WINDOW_MS = args.l2_window_ms
    L2_NORMALIZE = not args.l2_no_normalize

    n_list = [args.n] if args.n is not None else BATCH_N
    summary = [run_one(n, args.method, args.postproc, args.feature_method) for n in n_list]

    out_csv = os.path.join(base.SIM_ROOT, args.feature_method, "l2split_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nWrote {out_csv}")
    print(f"\n{'n':>4}{'types before':>14}{'types after':>13}{'purity before':>16}"
          f"{'purity after':>15}{'recovered before':>19}{'recovered after':>18}")
    for row in summary:
        print(f"{row['n_muaps']:>4}{row['n_types_before']:>14}{row['n_types_after']:>13}"
              f"{row['mean_purity_before']:>16.0%}{row['mean_purity_after']:>15.0%}"
              f"{row['n_recovered_before']:>19}{row['n_recovered_after']:>18}")


if __name__ == "__main__":
    main()
