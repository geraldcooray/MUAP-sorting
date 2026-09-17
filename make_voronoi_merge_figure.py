"""Voronoi tessellation before and after the fixed-Voronoi sequential merge,
K=10 trace, t-SNE + k-means. Re-runs the clustering to recover the 2-D t-SNE
embedding, then post-processes and merges exactly as merge_effect_table.py /
plot_tsne_sequential_merge do. The tessellation of the 14 post-processed
centroids is built once; the two panels differ only in cell colour.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import Voronoi

import simulate_and_sort_curated as base
import spikesort_simulated_traces as sst
from plot_tsne_types import cluster_from_scratch, post_process
from plot_tsne_sequential_merge import sequential_anchor_merge
from plot_tsne_voronoi import voronoi_finite_polygons_2d

HERE = os.path.dirname(os.path.abspath(__file__))
TSNE = os.path.join(HERE, "simulated_traces", "tsne")
OUT = os.path.abspath(os.path.join(HERE, "..", "Manuscript", "figures",
                                   "fig_voronoi_merge_n10.png"))
CMAP = plt.get_cmap("tab20")
K = 10


def draw(ax, centroids, emb, type_labels, colour_of, radius, title):
    n = centroids.shape[0]
    vor = Voronoi(centroids)
    regions, vertices = voronoi_finite_polygons_2d(vor, radius)
    vertices = np.asarray(vertices)
    for c, region in enumerate(regions):
        ax.add_patch(Polygon(vertices[region], closed=True,
                             facecolor=CMAP(colour_of[c] % 20), edgecolor="0.25",
                             lw=1.0, alpha=0.45, zorder=1))
    for c in range(n):
        m = type_labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=3, color=CMAP(colour_of[c] % 20),
                   alpha=0.35, linewidths=0, zorder=2)
    ax.scatter(centroids[:, 0], centroids[:, 1], s=35, c="k", marker="x", lw=1.4, zorder=4)
    for c, (cx, cy) in enumerate(centroids):
        ax.annotate(f"t{c}", (cx, cy), fontsize=8, fontweight="bold", ha="center",
                    va="center", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="k", lw=0.6))
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    _, lookup = base.load_curated_library()
    trace, fs, true_units, baseline_s = sst.load_simulated_trace(
        os.path.join(TSNE, f"simulated_trace_n{K}.npz"), lookup)
    base.call_spike_sort_emg.SILHOUETTE_MAX_N = K + 10
    overrides = dict(cluster_method="kmeans", amp_dur_weight=0.0, polarity_weight=0.0,
                     threshold_mad=sst.DETECTION_THRESHOLD_MAD)
    pt, pa, wf, emb, raw = cluster_from_scratch(trace, fs, baseline_s, "kmeans", overrides)
    keep, pp = post_process(pt, pa, wf, raw, fs)
    k_emb = emb[keep]
    kwf = wf[keep]
    merged, merged_of, _log, groups = sequential_anchor_merge(pp, kwf, "gap")
    n_pp = int(pp.max()) + 1
    n_mg = int(merged.max()) + 1
    print(f"pp {n_pp} -> merged {n_mg}, groups {groups}")

    centroids = np.stack([k_emb[pp == c].mean(axis=0) for c in range(n_pp)])
    radius = float(np.ptp(k_emb, axis=0).max()) * 3.0
    lo, hi = k_emb.min(axis=0), k_emb.max(axis=0)
    pad = 0.08 * (hi - lo)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    draw(axes[0], centroids, k_emb, pp, np.arange(n_pp), radius,
         f"Before merge: {n_pp} post-processed types")
    draw(axes[1], centroids, k_emb, pp, merged_of, radius,
         f"After merge: {n_mg} types "
         f"({', '.join('t' + '+t'.join(map(str, g)) for g in groups)} fused)")
    for ax in axes:
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    fig.suptitle("Fixed t-SNE Voronoi tessellation, K=10, t-SNE + k-means "
                 "(same partition, cells recoloured by merged group)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    fig.savefig(OUT.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
