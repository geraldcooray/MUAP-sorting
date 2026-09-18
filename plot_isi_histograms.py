#!/usr/bin/env python3
"""ISI histograms, actual vs estimated, for the six simulated traces.

For each trace size K in {3,5,10,15,20,30}: one figure, two columns --
column 1 the inter-spike-interval histogram of every TRUE (ground-truth)
unit that was RECOVERED (matched by at least one estimated type; unmatched
true units are dropped, which keeps the high-density figures to roughly
the estimated-type count rather than n_true), in unit-index order; column
2 the ISI histogram of every ESTIMATED
type from the existing t-SNE + Student's-t-mixture + full-post-processing
sort (min-ISI cleanup, latency-linkage merge, superposition detection --
NO gap-statistic merge), grouped by which true unit each type best matches
(increasing true-unit index; unmatched types last) and, within a group,
by decreasing purity -- so column 2's row order reflects match quality,
not the arbitrary raw cluster-label order. Each true unit and every
estimated type matched to it share one colour (tab20, by true-unit index;
unmatched types grey), so a matching group is obvious at a glance across
the two columns. All panels within one figure share the same x-axis and
bins: the 99th-percentile ISI pooled over every panel in that figure
(capped at 2000 ms) is placed at 90% of the axis length, so panels are
directly comparable and one long-tailed panel doesn't get to set the scale
for everyone.

Reads only already-saved results (simulated_traces/tsne/sort_n<K>_
tmixture_full.npz and simulated_trace_n<K>.npz) -- no re-simulation, no
re-sorting. Written to simulated_traces/tsne/isi_check/; the K=3 and K=20
outputs are copied to the manuscript figures/ directory as Figures
"isi_n3" and "isi_n20" (Results, Section 3.1).

Usage:
    python plot_isi_histograms.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import simulate_and_sort_curated as base
import spikesort_simulated_traces as sst
from spike_sort.metrics import isi_cv

N_LIST = [3, 5, 10, 15, 20, 30]
METHOD = "tmixture"
POSTPROC = "full"
FEATURE_METHOD = "tsne"

BIN_WIDTH_MS = 20.0
OUT_DIR = os.path.join(base.SIM_ROOT, "tsne", "isi_check")


def isi_ms(times_s):
    t = np.sort(np.asarray(times_s, dtype=float))
    return np.diff(t) * 1000.0


def robust_max_isi(isi_arrays):
    """99th-percentile ISI (ms) pooled over every non-empty array passed,
    capped at 2000 ms so one outlier-heavy panel can't blow up the shared
    axis for the whole figure."""
    pooled = np.concatenate([a for a in isi_arrays if a.size])
    if pooled.size == 0:
        return BIN_WIDTH_MS * 3
    return max(min(np.percentile(pooled, 99), 2000.0), BIN_WIDTH_MS * 3)


def panel(ax, times_s, color, label, bins, xlim_max):
    isis = isi_ms(times_s)
    n_spk = len(times_s)
    ax.set_xlim(0, xlim_max)
    if isis.size == 0:
        ax.text(0.5, 0.5, "no ISIs", ha="center", va="center", fontsize=7,
                transform=ax.transAxes, color="0.5")
        ax.set_title(f"{label}  n={n_spk}", fontsize=7)
        ax.set_yticks([])
        return
    cv = isi_cv(np.asarray(times_s, dtype=float))
    ax.hist(isis, bins=bins, color=color, edgecolor="none")
    ax.set_title(f"{label}  n={n_spk} spk, {len(isis)} ISI, CV={cv:.2f}", fontsize=7)
    ax.set_yticks([])
    ax.tick_params(labelsize=6)


def make_figure(n):
    library, lookup = base.load_curated_library()
    trace_npz = os.path.join(base.TRACE_DIR, f"simulated_trace_n{n}.npz")
    _, fs_true, true_units, _ = sst.load_simulated_trace(trace_npz, lookup)

    run_dir = os.path.join(base.SIM_ROOT, FEATURE_METHOD)
    d = np.load(os.path.join(run_dir, f"sort_n{n}_{METHOD}_{POSTPROC}.npz"), allow_pickle=True)
    peaks_t, labels, n_types = d["peaks_t"], d["labels"], int(d["n_types"])
    est_match, est_purity = d["est_match"], d["est_purity"]

    n_true = len(true_units)
    est_times = [peaks_t[labels == r] for r in range(n_types)]

    # Column 1: only true units RECOVERED by at least one estimated type (dropping the
    # rest keeps the high-density figures to roughly the estimated-type count, not n_true).
    recovered_ids = sorted(set(int(m) for m in est_match if m >= 0))
    true_times = {t: true_units[t]["spike_times"] for t in recovered_ids}
    nrows = max(len(recovered_ids), n_types)

    # Column 2 order: grouped by matched true unit (increasing; unmatched types last),
    # and within a group by decreasing purity -- not by raw type index.
    est_order = sorted(range(n_types),
                        key=lambda r: (int(est_match[r]) if est_match[r] >= 0 else n_true,
                                       -float(est_purity[r])))

    xlim_max = robust_max_isi([isi_ms(t) for t in list(true_times.values()) + est_times])
    xlim_max /= 0.9  # the robust max ISI sits at 90% of the axis length
    bins = np.arange(0, xlim_max + BIN_WIDTH_MS, BIN_WIDTH_MS)

    # One colour per true unit, shared by that unit's panel (column 1) and every
    # estimated type matched to it (column 2), so a matching group is visually
    # obvious at a glance; unmatched estimated types are grey.
    cmap = plt.get_cmap("tab20")
    color_for_true = {t: cmap(t % 20) for t in range(n_true)}
    UNMATCHED_COLOR = "0.6"

    fig, axes = plt.subplots(nrows, 2, figsize=(9, 1.5 * nrows), squeeze=False)

    for row in range(nrows):
        ax_true, ax_est = axes[row][0], axes[row][1]
        if row < len(recovered_ids):
            t = recovered_ids[row]
            u = true_units[t]
            panel(ax_true, true_times[t], color_for_true[t],
                  f"true {t}: {u['subject']} t{u['type_idx']}", bins, xlim_max)
        else:
            ax_true.axis("off")
        if row < n_types:
            r = est_order[row]
            match, purity = int(est_match[r]), float(est_purity[r])
            if match >= 0:
                tag, color = f"-> true {match} ({purity:.0%})", color_for_true[match]
            else:
                tag, color = "-> unmatched", UNMATCHED_COLOR
            panel(ax_est, est_times[r], color, f"type {r} {tag}", bins, xlim_max)
        else:
            ax_est.axis("off")
        if row == nrows - 1:
            if row < len(recovered_ids):
                ax_true.set_xlabel("ISI (ms)", fontsize=7)
            if row < n_types:
                ax_est.set_xlabel("ISI (ms)", fontsize=7)

    axes[0][0].annotate("ACTUAL (recovered ground-truth units)", xy=(0.5, 1.25),
                         xycoords="axes fraction", ha="center", fontsize=10, fontweight="bold")
    axes[0][1].annotate("ESTIMATED (sorted)", xy=(0.5, 1.25), xycoords="axes fraction",
                         ha="center", fontsize=10, fontweight="bold")
    fig.suptitle(f"ISI histograms, n={n} MUAPs | {FEATURE_METHOD} + {METHOD} | {POSTPROC} "
                 f"post-processing, no merge   ({len(recovered_ids)}/{n_true} true units "
                 f"recovered, {n_types} estimated types)",
                 fontsize=12, y=1.0 + 0.35 / nrows)
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"fig_isi_n{n}_{FEATURE_METHOD}_{METHOD}_{POSTPROC}.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"n={n}: {len(recovered_ids)}/{n_true} true units recovered, {n_types} "
          f"estimated types -> {out_path}")


def main():
    for n in N_LIST:
        make_figure(n)


if __name__ == "__main__":
    main()
