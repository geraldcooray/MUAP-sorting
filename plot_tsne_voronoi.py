#!/usr/bin/env python3
"""t-SNE cluster scatter next to a VORONOI tessellation of the k-means
centroids -- one polygonal face per type.

k-means assigns each point to its nearest centroid, so the Voronoi cell
of a centroid IS that type's decision region: convex, exact, tiling the
whole plane with no gaps or overlaps. Both panels use the exact
embedding + raw k-means labels that
plot_tsne_types.cluster_from_scratch reproduces (seed 42, deterministic,
before any post-processing).

Usage:
    python plot_tsne_voronoi.py --n 10 --method kmeans
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import Voronoi

import simulate_and_sort_curated as base
import spikesort_simulated_traces as sst
from plot_tsne_types import cluster_from_scratch
from spike_sort.io import load_signal


def voronoi_finite_polygons_2d(vor, radius):
    """Reconstruct infinite Voronoi regions of a 2D diagram into finite
    polygons (standard recipe). Returns (regions, vertices)."""
    new_regions, new_vertices = [], vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    for p1, region in enumerate(vor.point_region):
        verts = vor.regions[region]
        if verts and all(v >= 0 for v in verts):
            new_regions.append(verts)
            continue
        ridges = all_ridges[p1]
        new_region = [v for v in verts if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            new_region.append(len(new_vertices))
            new_vertices.append((vor.vertices[v2] + direction * radius).tolist())
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        order = np.argsort(np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0]))
        new_regions.append(list(np.array(new_region)[order]))
    return new_regions, np.asarray(new_vertices)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--method", default="kmeans", choices=["kmeans", "gmm", "tmixture"])
    args = ap.parse_args()

    _, lookup = base.load_curated_library()
    _, fs, _, baseline_s = sst.load_simulated_trace(
        os.path.join(base.TRACE_DIR, f"simulated_trace_n{args.n}.npz"), lookup)
    trace = load_signal(os.path.join(sst.SPIKESORT_INPUT_DIR,
                                      f"EMGRUN__curated_n{args.n}.txt"), fs, 0.0, None)
    base.call_spike_sort_emg.SILHOUETTE_MAX_N = args.n + 10

    overrides = dict(cluster_method=args.method, amp_dur_weight=0.0, polarity_weight=0.0,
                      threshold_mad=sst.DETECTION_THRESHOLD_MAD)
    _, _, _, emb, labels = cluster_from_scratch(trace, fs, baseline_s, args.method, overrides)
    n_types = int(labels.max()) + 1
    centroids = np.stack([emb[labels == c].mean(axis=0) for c in range(n_types)])
    print(f"n={args.n}: {n_types} types, {labels.size} points")

    lo, hi = emb.min(axis=0), emb.max(axis=0)
    pad = 0.08 * (hi - lo)
    xlim, ylim = (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])
    cmap = plt.get_cmap("tab20")

    fig, axes = plt.subplots(1, 2, figsize=(17, 8), sharex=True, sharey=True)

    # --- Panel 1: the t-SNE cluster scatter ---
    ax = axes[0]
    for c in range(n_types):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=5, color=cmap(c % 20), alpha=0.6, linewidths=0)
        cx, cy = centroids[c]
        ax.annotate(str(c), (cx, cy), fontsize=9, fontweight="bold", ha="center", va="center",
                    zorder=5, bbox=dict(boxstyle="circle,pad=0.15", fc=cmap(c % 20), ec="k",
                                         lw=0.6, alpha=0.9))
    ax.set_title(f"t-SNE feature space -- raw k-means ({n_types} types)", fontsize=11)

    # --- Panel 2: Voronoi tessellation of the k-means centroids ---
    ax = axes[1]
    vor = Voronoi(centroids)
    radius = np.ptp(emb, axis=0).max() * 3
    regions, vertices = voronoi_finite_polygons_2d(vor, radius)
    for c, region in enumerate(regions):
        ax.add_patch(Polygon(vertices[region], closed=True, facecolor=cmap(c % 20),
                              edgecolor="0.25", lw=1.0, alpha=0.4, zorder=1))
    for c in range(n_types):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=3, color=cmap(c % 20), alpha=0.35, linewidths=0,
                   zorder=2)
    ax.scatter(centroids[:, 0], centroids[:, 1], s=35, c="k", marker="x", lw=1.4, zorder=4)
    for c, (cx, cy) in enumerate(centroids):
        ax.annotate(str(c), (cx, cy), fontsize=9, fontweight="bold", ha="center", va="center",
                    zorder=5, bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="k", lw=0.6))
    ax.set_title(f"Voronoi tessellation of the {n_types} k-means centroids\n"
                 f"(each face = one type's nearest-centroid region)", fontsize=11)

    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("t-SNE dim 1 (standardized)")
        ax.set_ylabel("t-SNE dim 2 (standardized)")

    fig.suptitle(f"t-SNE clustering vs Voronoi tessellation, n={args.n} MUAPs | "
                 f"{args.method} (raw clustering, before post-processing)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = os.path.join(base.SIM_ROOT, "tsne",
                             f"tsne_voronoi_n{args.n}_{args.method}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
