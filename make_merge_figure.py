"""Figure for the Results 'merging' section: the K=10 curated trace, its
post-processed t-SNE + k-means types, and the types after the fixed-Voronoi
sequential gap-statistic merge (seqmerge_n10_kmeans_gap).

Three stacked blocks: actual (ground-truth) MUAPs, estimated (post-processed)
types, merged types. Every waveform is re-cut on +/-10 ms from the
measurement-band trace and peak-normalised.
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spikesort_simulated_traces as sst
import simulate_and_sort_curated as base
from spike_sort.detection import bandpass
from number_library_gallery import library_index

HERE = os.path.dirname(os.path.abspath(__file__))
TSNE = os.path.join(HERE, "simulated_traces", "tsne")
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "Manuscript", "figures"))
MEASURE_BAND = (20.0, 10000.0)
WIN_MS = 10.0
NCOL = 7
LIB_INDEX = library_index()


def reaverage(measured, fs, times):
    half = int(round(WIN_MS * 1e-3 * fs))
    idx = np.round(np.asarray(times) * fs).astype(int)
    idx = idx[(idx - half >= 0) & (idx + half < len(measured))]
    snips = np.stack([measured[p - half:p + half] - measured[p - half:p + half].mean()
                      for p in idx])
    return snips.mean(axis=0)


def norm(x):
    m = np.max(np.abs(x))
    return x / m if m else x


def axis_ms(n, fs):
    return (np.arange(n) - n / 2) / fs * 1000.0


def panel(ax, w, fs, label, color):
    ax.plot(axis_ms(len(w), fs), norm(w), color=color, lw=1.1)
    ax.axhline(0, color="0.85", lw=0.5, zorder=0)
    ax.set_xlim(-WIN_MS, WIN_MS)
    ax.set_yticks([])
    ax.tick_params(labelsize=5.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.text(0.03, 0.02, label, transform=ax.transAxes, fontsize=5.8, va="bottom",
            ha="left", color="0.15",
            bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.7))


def blank(ax):
    ax.set_visible(False)


def main():
    library, lookup = base.load_curated_library()
    npz_trace = os.path.join(TSNE, "simulated_trace_n10.npz")
    trace, fs, true_units, baseline_s = sst.load_simulated_trace(npz_trace, lookup)
    measured = bandpass(trace, fs, *MEASURE_BAND)

    d = np.load(os.path.join(TSNE, "merge_effect_n10_kmeans.npz"), allow_pickle=True)
    peaks_t = d["peaks_t"]
    pp_labels = d["pp_labels"]
    merged_labels = d["merged_labels"]
    merged_of = d["merged_of"]            # pp type -> merged group
    mlog = (d["merge_log_anchor"], d["merge_log_other"], d["merge_log_verdict"])

    tol = base.MATCH_TOL_MS / 1000.0
    mu_pp = sst.match_to_ground_truth(peaks_t, true_units, tol)
    pp_match, pp_purity = sst.compute_match_purity(true_units, pp_labels, mu_pp)
    mu_mg = sst.match_to_ground_truth(peaks_t, true_units, tol)
    mg_match, mg_purity = sst.compute_match_purity(true_units, merged_labels, mu_mg)

    n_pp = int(pp_labels.max()) + 1
    n_mg = int(merged_labels.max()) + 1
    members = {g: [p for p in range(n_pp) if merged_of[p] == g] for g in range(n_mg)}

    # ---- report ----
    def summ(match, purity):
        rec = len({m for m in match if m >= 0})
        return rec, float(np.mean(purity))
    print(f"post-processed: {n_pp} types, recovered {summ(pp_match, pp_purity)[0]}/10, "
          f"mean purity {summ(pp_match, pp_purity)[1]:.2f}")
    print(f"merged:         {n_mg} types, recovered {summ(mg_match, mg_purity)[0]}/10, "
          f"mean purity {summ(mg_match, mg_purity)[1]:.2f}")
    for g, mem in members.items():
        if len(mem) > 1:
            print(f"  merged group {g} <- pp types {mem}")
    merges = [(int(a), int(o)) for a, o, v in zip(*mlog) if v == "MERGE"]
    print("  MERGE verdicts:", merges)

    def unit_num(u):
        return LIB_INDEX.get(f"{u['subject']}_type{u['type_idx']}", "?")

    # ---- figure: three stacked blocks via subfigures ----
    fig = plt.figure(figsize=(1.75 * NCOL, 12.5))
    sf = fig.subfigures(3, 1, height_ratios=[2, 2, 2], hspace=0.06)

    def block(subfig, title, n, waveform_fn, label_fn, color):
        subfig.suptitle(title, fontsize=10, fontweight="bold", x=0.02, ha="left")
        rows = int(np.ceil(max(n, 1) / NCOL))
        axs = subfig.subplots(rows, NCOL, sharex=True, squeeze=False)
        for i in range(rows * NCOL):
            a = axs[i // NCOL][i % NCOL]
            if i < n:
                panel(a, waveform_fn(i), fs, label_fn(i), color)
            else:
                blank(a)
        for j in range(NCOL):
            axs[-1][j].set_xlabel("ms", fontsize=6)

    block(sf[0], "Actual (ground-truth) MUAPs", len(true_units),
          lambda i: reaverage(measured, fs, true_units[i]["spike_times"]),
          lambda i: f"MUAP {unit_num(true_units[i])}", "black")

    def pp_label(k):
        tgt = (f"MUAP {unit_num(true_units[pp_match[k]])} {pp_purity[k]:.0%}"
               if pp_match[k] >= 0 else "unmatched")
        return f"t{k} -> {tgt}"
    block(sf[1], f"Estimated types (t-SNE + $k$-means, post-processed): {n_pp}", n_pp,
          lambda k: reaverage(measured, fs, peaks_t[pp_labels == k]), pp_label, "#1f77b4")

    def mg_label(g):
        tgt = (f"MUAP {unit_num(true_units[mg_match[g]])} {mg_purity[g]:.0%}"
               if mg_match[g] >= 0 else "unmatched")
        return f"{'+'.join(f't{m}' for m in members[g])} -> {tgt}"
    block(sf[2], f"After fixed-Voronoi sequential merge: {n_mg}", n_mg,
          lambda g: reaverage(measured, fs, peaks_t[merged_labels == g]), mg_label, "#2ca02c")

    out = os.path.join(OUT_DIR, "fig_merge_n10.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
