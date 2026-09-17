"""Exploratory: aggregate firing rate vs aggregate "power" for the experimental
sorts, as a single scatter cloud coloured by clinical category
(Myopathy = red, Normal = green, Neuro = blue).

For each subject (merged MUAP types from experimental_pipeline.py, summary.npz):
  * per-type instantaneous rate  r_i(t)  (Gaussian kernel, sigma = 1 s), estimated
    within each original epoch, on a 0.1 s grid;
  * per-type mean peak-to-peak amplitude  A_i  (from summary.csv, merged rows);
  * total firing rate      F(t) = sum_i r_i(t)                     [Hz]
  * total power            P(t) = sum_i A_i * r_i(t)               [a.u. * Hz]

Each panel pools every within-epoch grid point from the subjects in that category
and plots F(t) (x) against P(t) (y), one marker style per subject.

Writes
  Manuscript/figures/fig_exp_power_vs_rate.{png,pdf}       single shaded cloud
  Manuscript/figures/fig_exp_power_vs_rate_kde.{png,pdf}   one 2-D KDE panel per category
NOT referenced by the manuscript.

Run:  python experimental_power_vs_rate.py
"""
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_GLOB = os.path.join(HERE, "..", "data", "EMG",
                         "*", "EMG", "Tibialis_anterior", "*", "postprocessed", "summary.npz")
FIG_DIR = os.path.join(HERE, "..", "Manuscript", "figures")

SIGMA_S = 1.0
GRID_DT_S = 0.1

CATEGORIES = ["Myopathy", "Normal", "Neuro"]
CAT_OF = {"myopathy": "Myopathy", "normal": "Normal", "neuro": "Neuro"}
CAT_COLOR = {"Myopathy": "tab:red", "Normal": "tab:green", "Neuro": "tab:blue"}
CAT_CMAP = {"Myopathy": "Reds", "Normal": "Greens", "Neuro": "Blues"}


def gaussian_rate(spike_t, grid_t, sigma):
    if spike_t.size == 0:
        return np.zeros_like(grid_t)
    d = grid_t[:, None] - spike_t[None, :]
    k = np.exp(-0.5 * (d / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return k.sum(axis=1)


def merged_amplitudes(run_dir, n_types):
    """A_i per merged type from summary.csv (falls back to template p2p)."""
    csv_path = os.path.join(run_dir, "summary.csv")
    amp = {}
    if os.path.exists(csv_path):
        with open(csv_path) as fh:
            for row in csv.DictReader(fh):
                if row["stage"] == "merged":
                    amp[int(row["type"])] = float(row["p2p_amplitude"])
    return np.array([amp.get(i, np.nan) for i in range(n_types)])


def subject_series(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    fs = float(d["fs"])
    peaks_t = d["peaks_t"].astype(float)
    labels = d["merged_labels"].astype(int)
    edges_s = d["run_bounds"].astype(float) / fs
    n_types = int(labels.max()) + 1

    run_dir = os.path.dirname(npz_path)
    A = merged_amplitudes(run_dir, n_types)
    if not np.isfinite(A).all():
        tmpl = d["merged_templates"]
        A = np.where(np.isfinite(A), A, np.ptp(tmpl, axis=1))

    F_parts, P_parts = [], []
    for a, b in zip(edges_s[:-1], edges_s[1:]):
        if b - a < GRID_DT_S:
            continue
        g = np.arange(a, b, GRID_DT_S)
        r = np.stack([gaussian_rate(peaks_t[(labels == c) & (peaks_t >= a) & (peaks_t < b)],
                                    g, SIGMA_S) for c in range(n_types)])
        F_parts.append(r.sum(axis=0))
        P_parts.append((A[:, None] * r).sum(axis=0))
    F = np.concatenate(F_parts)
    P = np.concatenate(P_parts)
    return F, P, n_types, A


def main():
    by_cat = {c: [] for c in CATEGORIES}
    for npz_path in sorted(glob.glob(DATA_GLOB)):
        subj = npz_path.split(os.sep + "data" + os.sep + "EMG" + os.sep)[1].split(os.sep)[0]
        cat = CAT_OF[subj.split("_", 1)[1]]
        F, P, n_types, A = subject_series(npz_path)
        by_cat[cat].append((subj, F, P))
        print(f"{subj:16s} {cat:9s} types={n_types}  A_i={np.round(A,0)}  "
              f"F: median {np.median(F):.1f}, max {F.max():.1f} Hz   "
              f"P: median {np.median(P):.0f}, max {P.max():.0f}")

    pooled = {cat: (np.concatenate([f for _, f, _ in by_cat[cat]]),
                    np.concatenate([p for _, _, p in by_cat[cat]]))
              for cat in CATEGORIES if by_cat[cat]}

    # per-point slope  s = P/R = effective mean amplitude (a.u.), for F above a floor
    F_FLOOR_HZ = 5.0
    med_slope = {}
    for cat, (F, P) in pooled.items():
        m = F >= F_FLOOR_HZ
        med_slope[cat] = float(np.median(P[m] / F[m]))
    # region boundaries: geometric mean of adjacent category median slopes
    b_lo = np.sqrt(med_slope["Myopathy"] * med_slope["Normal"])
    b_hi = np.sqrt(med_slope["Normal"] * med_slope["Neuro"])

    xmax = max(F.max() for F, _ in pooled.values()) * 1.02
    ymax = max(P.max() for _, P in pooled.values()) * 1.02

    fig, ax = plt.subplots(figsize=(7.8, 6.8))
    xs = np.array([0.0, xmax])
    ax.fill_between(xs, 0, b_lo * xs, color=CAT_COLOR["Myopathy"], alpha=0.10, zorder=0)
    ax.fill_between(xs, b_lo * xs, b_hi * xs, color=CAT_COLOR["Normal"], alpha=0.10, zorder=0)
    ax.fill_between(xs, b_hi * xs, ymax, color=CAT_COLOR["Neuro"], alpha=0.10, zorder=0)
    for b in (b_lo, b_hi):
        ax.plot(xs, b * xs, color="0.35", lw=0.9, ls="--", zorder=1)
    ax.text(0.97, 0.03, f"region boundaries: P/R = {b_lo:.0f} and {b_hi:.0f} a.u.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="0.35")

    for cat, (F, P) in pooled.items():
        ax.scatter(F, P, s=8, color=CAT_COLOR[cat], alpha=0.28, edgecolor="none", zorder=3,
                   label=f"{cat} (n={len(by_cat[cat])}, median P/R = {med_slope[cat]:.0f})")

    # how cleanly the slope wedges separate the categories
    edges = {"Myopathy": (0, b_lo), "Normal": (b_lo, b_hi), "Neuro": (b_hi, np.inf)}
    print("\nslope-wedge separation (points with F >= "
          f"{F_FLOOR_HZ:g} Hz):")
    for cat, (F, P) in pooled.items():
        m = F >= F_FLOOR_HZ
        s = P[m] / F[m]
        lo, hi = edges[cat]
        print(f"  {cat:9s}: {np.mean((s >= lo) & (s < hi)) * 100:5.1f}% of {m.sum()} points in its own region")

    ax.set_xlabel(r"total firing rate  $R=\sum_c r_c(t)$   (Hz)", fontsize=10)
    ax.set_ylabel(r"total power  $P=\sum_c A_c\, r_c(t)$   (a.u.$\cdot$Hz)", fontsize=10)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.tick_params(labelsize=9)
    leg = ax.legend(fontsize=8.5, framealpha=0.95, markerscale=2.5, loc="upper left")
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    ax.set_title("Experimental recordings: aggregate power vs aggregate firing rate,\n"
                 "shaded into disorder regions by the P/R slope "
                 f"(rate: Gaussian kernel sigma = {SIGMA_S:g} s, {GRID_DT_S:g} s grid)",
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_exp_power_vs_rate.{ext}"),
                    dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {os.path.join(FIG_DIR, 'fig_exp_power_vs_rate.png')} (+ .pdf)  [not in manuscript]")

    # -------------------------------------------------------------------
    # 3 panels: 2-D KDE of the individual (F(t), P(t)) points per category.
    # Near-silent inter-contraction samples (F < F_FLOOR) are dropped so the
    # KDE is not swamped by a spike at the origin.
    # -------------------------------------------------------------------
    xhi = max(F[F >= F_FLOOR_HZ].max() for F, _ in pooled.values()) * 1.03
    yhi = max(P[F >= F_FLOOR_HZ].max() for F, P in pooled.values()) * 1.10
    ylo = max(50.0, min(P[(F >= F_FLOOR_HZ) & (P > 0)].min() for F, P in pooled.values()))
    # KDE is fitted in (F, log10 P) space so the density is meaningful on a log y axis
    gx, glog = np.mgrid[0:xhi:220j, np.log10(ylo):np.log10(yhi):220j]
    grid = np.vstack([gx.ravel(), glog.ravel()])
    gy = 10.0 ** glog
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)
    for ax2, cat in zip(axes2, CATEGORIES):
        F0, P0 = pooled[cat]
        m = (F0 >= F_FLOOR_HZ) & (P0 > 0)
        F, P = F0[m], P0[m]
        kde = gaussian_kde(np.vstack([F, np.log10(P)]))
        dens = kde(grid).reshape(gx.shape)
        ax2.contourf(gx, gy, dens, levels=14, cmap=CAT_CMAP[cat])
        ax2.contour(gx, gy, dens, levels=7, colors="white", linewidths=0.4, alpha=0.6)
        ax2.scatter(F, P, s=3, color="0.15", alpha=0.10, edgecolor="none")
        xs = np.linspace(0, xhi, 200)
        for b in (b_lo, b_hi):
            ax2.plot(xs, b * xs, color="0.4", lw=0.8, ls="--")
        ax2.set_title(f"{cat}  (n={len(by_cat[cat])} subjects, "
                      f"{F.size} samples, R $\\geq$ {F_FLOOR_HZ:g} Hz)",
                      fontsize=10, color=CAT_COLOR[cat])
        ax2.set_xlabel(r"total firing rate  $R=\sum_c r_c(t)$   (Hz)", fontsize=9)
        ax2.set_yscale("log")
        ax2.set_xlim(0, xhi)
        ax2.set_ylim(ylo, yhi)
        ax2.tick_params(labelsize=8)
    axes2[0].set_ylabel(r"total power  $P=\sum_c A_c\, r_c(t)$   (a.u.$\cdot$Hz)", fontsize=9)
    fig2.suptitle("Experimental recordings: 2-D KDE of the individual $(R, P)$ samples per disorder "
                  f"(rate: Gaussian kernel sigma = {SIGMA_S:g} s, {GRID_DT_S:g} s grid; "
                  "dashed = P/R region boundaries)", fontsize=10)
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("png", "pdf"):
        fig2.savefig(os.path.join(FIG_DIR, f"fig_exp_power_vs_rate_kde.{ext}"),
                     dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {os.path.join(FIG_DIR, 'fig_exp_power_vs_rate_kde.png')} (+ .pdf)  [not in manuscript]")


if __name__ == "__main__":
    main()
