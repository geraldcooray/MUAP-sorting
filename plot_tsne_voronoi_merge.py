#!/usr/bin/env python3
"""Voronoi-adjacency-driven cluster merge on a raw k-means t-SNE
clustering.

k-means gives one Voronoi face per type. For every pair of ADJACENT
faces (centroids that share a Voronoi ridge), pool the two clusters'
+/-5 ms waveforms, re-embed that pool alone with a fresh t-SNE and split
it with k-means (k=2): if the split is NOT well separated (silhouette <
SPLIT_SILHOUETTE_MIN, via muap_split_refinement.try_split_type) the pool
is really ONE cluster and the two types are merged; otherwise they are
left alone. Every adjacent pair is tested once; merges are unioned, so a
chain (A-B one cluster, B-C one cluster) collapses {A, B, C}.

Outputs, in simulated_traces/tsne/:
  voronoi_merge_n<N>_<method>.png                 3 panels: t-SNE scatter,
                                                   raw Voronoi, merged Voronoi
  voronoi_merge_n<N>_<method>_gallery_true.png    mean trace-cut per TRUE unit
  voronoi_merge_n<N>_<method>_gallery_raw.png     mean MUAP per raw type
  voronoi_merge_n<N>_<method>_gallery_merged.png  mean MUAP per merged type

Usage:
    python plot_tsne_voronoi_merge.py --n 10 --method kmeans
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import Voronoi

import muap_split_refinement as msr
import simulate_and_sort_curated as base
import spikesort_simulated_traces as sst
from plot_tsne_types import cluster_from_scratch
from plot_tsne_voronoi import voronoi_finite_polygons_2d
from spike_sort.io import load_signal

CMAP = plt.get_cmap("tab20")


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def adjacent_centroid_pairs(centroids):
    """Unordered pairs of centroid indices that share a Voronoi ridge
    (== Delaunay edges == adjacent Voronoi cells)."""
    vor = Voronoi(centroids)
    return sorted({(min(i, j), max(i, j)) for i, j in vor.ridge_points})


def merge_by_voronoi_adjacency(emb, labels, waveforms):
    """Test every adjacent centroid pair; merge the pair when their
    pooled spikes form a single cluster. Returns (merged_labels, log)."""
    n_types = int(labels.max()) + 1
    centroids = np.stack([emb[labels == c].mean(axis=0) for c in range(n_types)])
    uf = UnionFind(n_types)
    log = []
    for i, j in adjacent_centroid_pairs(centroids):
        mask = (labels == i) | (labels == j)
        did_split, _, sil = msr.try_split_type(waveforms[mask])
        sil_txt = round(sil, 3) if sil == sil else None
        if not did_split:                      # pool is 1 cluster -> merge
            uf.union(i, j)
            log.append((i, j, sil_txt, "MERGE", int(mask.sum())))
        else:                                  # pool is 2+ clusters -> keep
            log.append((i, j, sil_txt, "keep", int(mask.sum())))

    roots = {c: uf.find(c) for c in range(n_types)}
    groups = {}
    for c, r in roots.items():
        groups.setdefault(r, []).append(c)
    remap = {}
    for new_id, members in enumerate(sorted(groups.values(), key=min)):
        for m in members:
            remap[m] = new_id
    merged_labels = np.array([remap[l] for l in labels])
    return merged_labels, log, [tuple(sorted(m)) for m in groups.values() if len(m) > 1]


def draw_voronoi(ax, centroids, emb, labels, title, radius):
    if len(centroids) >= 4:
        vor = Voronoi(centroids)
        regions, vertices = voronoi_finite_polygons_2d(vor, radius)
        for c, region in enumerate(regions):
            ax.add_patch(Polygon(vertices[region], closed=True, facecolor=CMAP(c % 20),
                                  edgecolor="0.25", lw=1.0, alpha=0.4, zorder=1))
    for c in range(int(labels.max()) + 1):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=3, color=CMAP(c % 20), alpha=0.35, linewidths=0,
                   zorder=2)
    ax.scatter(centroids[:, 0], centroids[:, 1], s=35, c="k", marker="x", lw=1.4, zorder=4)
    for c, (cx, cy) in enumerate(centroids):
        ax.annotate(str(c), (cx, cy), fontsize=9, fontweight="bold", ha="center", va="center",
                    zorder=5, bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="k", lw=0.6))
    ax.set_title(title, fontsize=11)


def _gallery(panels, measured, fs, title, out_path):
    """panels: list of (times_s, colour, panel_title)."""
    n = len(panels)
    ncols = min(base.GALLERY_NCOLS, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.0 * nrows), squeeze=False)
    for k in range(nrows * ncols):
        ax = axes[k // ncols][k % ncols]
        if k >= n:
            ax.axis("off")
            continue
        times, colour, ptitle = panels[k]
        mean_wf, trials, n_tr, t_ms = base.mean_waveform(measured, times, fs, base.GALLERY_HALF_MS)
        m = base.muap_metrics(mean_wf, fs)
        base.plot_muap_panel(ax, trials if n_tr else mean_wf[None, :], mean_wf, m, t_ms, colour,
                              f"{ptitle}   n={n_tr}, {m['duration_ms']:.1f} ms",
                              muap_ylim=base.GALLERY_YLIM,
                              muap_xlim=(-base.GALLERY_HALF_MS, base.GALLERY_HALF_MS),
                              ylabel="uV" if k % ncols == 0 else None)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def purity_vs_truth(labels, peaks_t, true_units):
    """Per-type (best-matching true unit, purity) + summary, using the
    same nearest-true-spike matching / majority vote as
    spikesort_simulated_traces. purity = fraction of a type's matched
    spikes that belong to its majority true unit."""
    matched_unit = sst.match_to_ground_truth(peaks_t, true_units, base.MATCH_TOL_MS / 1000.0)
    est_match, est_purity = sst.compute_match_purity(true_units, labels, matched_unit)
    n_types = int(labels.max()) + 1
    weights = np.array([(labels == c).sum() for c in range(n_types)], dtype=float)
    mean_p = float(np.mean(est_purity)) if est_purity else 0.0
    wmean_p = float(np.average(est_purity, weights=weights)) if est_purity else 0.0
    recovered = len({m for m in est_match if m >= 0})
    return est_match, est_purity, dict(mean=mean_p, weighted_mean=wmean_p,
                                        recovered=recovered, n_true=len(true_units))


def gallery(labels, peaks_t, measured, fs, title, out_path, est_match=None, est_purity=None):
    n_types = int(labels.max()) + 1
    panels = []
    for k in range(n_types):
        tag = f"type {k}"
        if est_match is not None:
            tag += (f" -> true {est_match[k]} ({est_purity[k]:.0%})" if est_match[k] >= 0
                    else " -> unmatched")
        panels.append((peaks_t[labels == k], CMAP(k % 20), tag))
    _gallery(panels, measured, fs, title, out_path)


def gallery_true(true_units, measured, fs, title, out_path):
    panels = [(u["spike_times"], CMAP(i % 20),
               f"true {i}: {u['subject']} t{u['type_idx']}") for i, u in enumerate(true_units)]
    _gallery(panels, measured, fs, title, out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--method", default="kmeans", choices=["kmeans", "gmm", "tmixture"])
    args = ap.parse_args()

    _, lookup = base.load_curated_library()
    _, fs, true_units, baseline_s = sst.load_simulated_trace(
        os.path.join(base.TRACE_DIR, f"simulated_trace_n{args.n}.npz"), lookup)
    trace = load_signal(os.path.join(sst.SPIKESORT_INPUT_DIR,
                                      f"EMGRUN__curated_n{args.n}.txt"), fs, 0.0, None)
    base.call_spike_sort_emg.SILHOUETTE_MAX_N = args.n + 10

    overrides = dict(cluster_method=args.method, amp_dur_weight=0.0, polarity_weight=0.0,
                      threshold_mad=sst.DETECTION_THRESHOLD_MAD)
    peaks_t, _, waveforms, emb, raw_labels = cluster_from_scratch(
        trace, fs, baseline_s, args.method, overrides)
    n_raw = int(raw_labels.max()) + 1
    print(f"n={args.n}: raw k-means -> {n_raw} types, {raw_labels.size} spikes")

    merged_labels, log, merge_groups = merge_by_voronoi_adjacency(emb, raw_labels, waveforms)
    n_merged = int(merged_labels.max()) + 1
    print(f"tested {len(log)} adjacent pairs; "
          f"{sum(1 for r in log if r[3] == 'MERGE')} flagged as one cluster")
    for i, j, sil, verdict, npool in log:
        if verdict == "MERGE":
            print(f"  {verdict}  types {i:>2} + {j:<2}  (pool n={npool}, silhouette={sil})")
    print(f"merge groups: {merge_groups}")
    print(f"{n_raw} -> {n_merged} types")

    raw_centroids = np.stack([emb[raw_labels == c].mean(axis=0) for c in range(n_raw)])
    merged_centroids = np.stack([emb[merged_labels == c].mean(axis=0) for c in range(n_merged)])
    radius = np.ptp(emb, axis=0).max() * 3
    lo, hi = emb.min(axis=0), emb.max(axis=0)
    pad = 0.08 * (hi - lo)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharex=True, sharey=True)
    for c in range(n_raw):
        m = raw_labels == c
        axes[0].scatter(emb[m, 0], emb[m, 1], s=5, color=CMAP(c % 20), alpha=0.6, linewidths=0)
        axes[0].annotate(str(c), raw_centroids[c], fontsize=9, fontweight="bold", ha="center",
                          va="center", zorder=5,
                          bbox=dict(boxstyle="circle,pad=0.15", fc=CMAP(c % 20), ec="k", lw=0.6))
    axes[0].set_title(f"t-SNE feature space -- raw k-means ({n_raw} types)", fontsize=11)
    draw_voronoi(axes[1], raw_centroids, emb, raw_labels,
                 f"Voronoi tessellation -- raw ({n_raw} faces)", radius)
    draw_voronoi(axes[2], merged_centroids, emb, merged_labels,
                 f"Voronoi tessellation -- after adjacency merge ({n_merged} faces)", radius)
    for ax in axes:
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_xlabel("t-SNE dim 1 (standardized)")
        ax.set_ylabel("t-SNE dim 2 (standardized)")
    fig.suptitle(f"Voronoi-adjacency cluster merge, n={args.n} MUAPs | {args.method} "
                 f"(raw clustering)   {n_raw} -> {n_merged} types", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_dir = os.path.join(base.SIM_ROOT, "tsne")
    fig.savefig(os.path.join(out_dir, f"voronoi_merge_n{args.n}_{args.method}.png"), dpi=150)
    print(f"Wrote {os.path.join(out_dir, f'voronoi_merge_n{args.n}_{args.method}.png')}")

    # --- purity of identified spikes vs ground-truth MUAPs ---
    raw_match, raw_purity, raw_stats = purity_vs_truth(raw_labels, peaks_t, true_units)
    mrg_match, mrg_purity, mrg_stats = purity_vs_truth(merged_labels, peaks_t, true_units)
    print(f"\nPurity vs {len(true_units)} ground-truth MUAPs "
          f"(match tol {base.MATCH_TOL_MS} ms):")
    print(f"  {'':10}{'types':>7}{'mean purity':>14}{'spike-wtd':>12}{'true units rec.':>18}")
    for name, st in (("raw k-means", raw_stats), ("merged", mrg_stats)):
        print(f"  {name:<10}{(n_raw if name.startswith('raw') else n_merged):>7}"
              f"{st['mean']:>13.0%}{st['weighted_mean']:>12.0%}"
              f"{st['recovered']:>10}/{st['n_true']}")
    print("  per raw type   :  " + "  ".join(
        f"{c}->{raw_match[c]}({raw_purity[c]:.0%})" if raw_match[c] >= 0 else f"{c}->--"
        for c in range(n_raw)))
    print("  per merged type:  " + "  ".join(
        f"{c}->{mrg_match[c]}({mrg_purity[c]:.0%})" if mrg_match[c] >= 0 else f"{c}->--"
        for c in range(n_merged)))

    measured = base.bandpass(trace, fs, 20.0, 10000.0)
    gallery_true(true_units, measured, fs,
                 f"Actual (ground-truth) MUAPs, n={args.n}  ({len(true_units)} units)",
                 os.path.join(out_dir, f"voronoi_merge_n{args.n}_{args.method}_gallery_true.png"))
    gallery(raw_labels, peaks_t, measured, fs,
            f"Raw k-means MUAPs, n={args.n} | {args.method}  ({n_raw} types, "
            f"mean purity {raw_stats['mean']:.0%}, {raw_stats['recovered']}/{raw_stats['n_true']} "
            f"true units)",
            os.path.join(out_dir, f"voronoi_merge_n{args.n}_{args.method}_gallery_raw.png"),
            est_match=raw_match, est_purity=raw_purity)
    gallery(merged_labels, peaks_t, measured, fs,
            f"After Voronoi-adjacency merge, n={args.n} | {args.method}  ({n_merged} types, "
            f"mean purity {mrg_stats['mean']:.0%}, {mrg_stats['recovered']}/{mrg_stats['n_true']} "
            f"true units)",
            os.path.join(out_dir, f"voronoi_merge_n{args.n}_{args.method}_gallery_merged.png"),
            est_match=mrg_match, est_purity=mrg_purity)


if __name__ == "__main__":
    main()
