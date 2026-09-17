"""Apply the fixed t-SNE + Student-t-mixture + full-post-processing sort, then
the fixed-Voronoi sequential gap-merge, to each subject's concatenated
Tibialis-anterior intramuscular EMG.

Per subject, into <run_dir>/postprocessed/ :
  gallery_postproc.png   mean MUAP per post-processed type
  gallery_merged.png     mean MUAP per merged type
  raster_postproc.png    spike raster over the whole concatenated recording
  raster_merged.png      same, merged labels
  ampdur_kde_postproc.png   per-type mean p2p amplitude vs duration, KDE density
  ampdur_kde_merged.png
  summary.npz / summary.csv   labels, spike times, per-type metrics, merge log

Run:  python experimental_pipeline.py            # all subjects
      python experimental_pipeline.py test3_myopathy
"""
import csv
import glob
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import plot_tsne_types as ptt
import simulate_and_sort_curated as base  # noqa: F401  (monkeypatches cse.silhouette_sweep)
import call_spike_sort_emg as cse
from plot_tsne_types import cluster_from_scratch, post_process
from plot_tsne_sequential_merge import sequential_anchor_merge
from spike_sort.detection import bandpass
from spike_sort.io import load_signal, lookup_sample_rate
from spike_sort.metrics import muap_metrics

# clinical detection band, not the 100 Hz simulation override
ptt.DETECTION_HIGHPASS_HZ = 500.0
cse.SILHOUETTE_MAX_N = 25

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "..", "data", "EMG")
TEMPLATE_WIN_MS = 10.0
CMAP = plt.get_cmap("tab20")

SUBJECTS = [
    ("test0_myopathy", "Left"),
    ("test1_normal", "Left"),
    ("test2_normal", "Left"),
    ("test3_myopathy", "Right"),
    ("test4_neuro", "Left"),
    ("test5_neuro", "Left"),
]


def load_concatenated(run_dir):
    run_files = sorted(glob.glob(os.path.join(run_dir, "EMGRUN__*.txt")))
    if not run_files:
        raise FileNotFoundError(run_dir)
    fs = lookup_sample_rate(run_files[0])
    segs, bounds = [], [0]
    for f in run_files:
        s = load_signal(f, fs, 0.0, None)
        segs.append(s)
        bounds.append(bounds[-1] + s.size)
    return np.concatenate(segs), fs, np.array(bounds), [os.path.basename(f) for f in run_files]


def reaverage(measured, fs, times, win_ms=TEMPLATE_WIN_MS):
    half = int(round(win_ms * 1e-3 * fs))
    idx = np.round(np.asarray(times) * fs).astype(int)
    idx = idx[(idx - half >= 0) & (idx + half < len(measured))]
    if idx.size == 0:
        return np.zeros(2 * half)
    return np.stack([measured[p - half:p + half] - measured[p - half:p + half].mean()
                     for p in idx]).mean(axis=0)


def type_table(measured, fs, times_by_type):
    """per type: mean +/-10 ms template, n spikes, p2p amplitude, duration (ms)."""
    rows = []
    for c, tt in enumerate(times_by_type):
        wf = reaverage(measured, fs, tt)
        m = muap_metrics(wf, fs)
        rows.append(dict(type=c, n=len(tt), template=wf,
                         amplitude=float(m["amplitude"]), duration_ms=float(m["duration_ms"])))
    return rows


def fig_gallery(rows, fs, title, out_path):
    n = len(rows)
    ncol = min(6, max(n, 1))
    nrow = int(np.ceil(n / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 1.7 * nrow),
                           squeeze=False, sharex=True)
    order = sorted(range(n), key=lambda i: -rows[i]["n"])
    t_ms = (np.arange(len(rows[0]["template"])) - len(rows[0]["template"]) / 2) / fs * 1000 \
        if n else np.array([])
    for slot in range(nrow * ncol):
        a = ax[slot // ncol][slot % ncol]
        if slot >= n:
            a.set_visible(False)
            continue
        r = rows[order[slot]]
        a.plot(t_ms, r["template"], color=CMAP(r["type"] % 20), lw=1.2)
        a.axhline(0, color="0.85", lw=0.5)
        a.set_title(f"type {r['type']}  n={r['n']}\n"
                    f"p2p={r['amplitude']:.0f}, {r['duration_ms']:.1f} ms", fontsize=7)
        a.set_yticks([])
        a.tick_params(labelsize=6)
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
    for j in range(ncol):
        ax[-1][j].set_xlabel("ms", fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_raster(peaks_t, labels, bounds, fs, title, out_path):
    n_types = int(labels.max()) + 1 if labels.size else 0
    dur_s = bounds[-1] / fs
    fig, axr = plt.subplots(figsize=(min(22, 4 + dur_s / 12), 1.1 + 0.32 * max(n_types, 1)))
    for c in range(n_types):
        m = labels == c
        axr.scatter(peaks_t[m], np.full(m.sum(), c), s=4, marker="|",
                    color=CMAP(c % 20), linewidths=0.6)
    for b in bounds[1:-1]:
        axr.axvline(b / fs, color="0.8", lw=0.7, zorder=0)
    axr.set_yticks(range(n_types))
    axr.set_ylabel("type")
    axr.set_xlabel("time (s)")
    axr.set_xlim(0, dur_s)
    axr.set_ylim(-0.5, n_types - 0.5)
    axr.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_ampdur_kde(rows, title, out_path):
    amp = np.array([r["amplitude"] for r in rows], float)
    dur = np.array([r["duration_ms"] for r in rows], float)
    wt = np.array([r["n"] for r in rows], float)
    fig, axk = plt.subplots(figsize=(6.2, 5.2))
    ok = len(rows) >= 3 and np.ptp(amp) > 0 and np.ptp(dur) > 0
    if ok:
        try:
            kde = gaussian_kde(np.vstack([amp, dur]), weights=wt)
            ax_pad = 0.20 * (amp.max() - amp.min() + 1e-9)
            dr_pad = 0.20 * (dur.max() - dur.min() + 1e-9)
            xs = np.linspace(amp.min() - ax_pad, amp.max() + ax_pad, 160)
            ys = np.linspace(max(0, dur.min() - dr_pad), dur.max() + dr_pad, 160)
            gx, gy = np.meshgrid(xs, ys)
            dens = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
            cf = axk.contourf(gx, gy, dens, levels=12, cmap="magma")
            axk.contour(gx, gy, dens, levels=6, colors="white", linewidths=0.4, alpha=0.5)
            fig.colorbar(cf, ax=axk, label="probability density")
        except Exception as e:  # noqa: BLE001
            axk.text(0.5, 0.5, f"KDE failed: {e}", transform=axk.transAxes, ha="center")
    else:
        axk.text(0.5, 0.9, "too few / degenerate types for KDE", transform=axk.transAxes,
                 ha="center", fontsize=8, color="0.4")
    axk.scatter(amp, dur, s=20 + 120 * wt / wt.max(), facecolor="none",
                edgecolor="cyan", linewidths=1.3, zorder=5)
    for r in rows:
        axk.annotate(str(r["type"]), (r["amplitude"], r["duration_ms"]), fontsize=7,
                     color="white", ha="center", va="center", zorder=6)
    axk.set_xlabel("mean peak-to-peak amplitude (a.u.)")
    axk.set_ylabel("mean duration (ms)")
    axk.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def process_subject(subject, side):
    run_dir = os.path.join(DATA_ROOT, subject, "EMG", "Tibialis_anterior", side)
    out_dir = os.path.join(run_dir, "postprocessed")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    trace, fs, bounds, run_names = load_concatenated(run_dir)
    print(f"[{subject}] {len(run_names)} runs, {trace.size} samples "
          f"({trace.size / fs:.1f} s) @ {fs:g} Hz")

    overrides = dict(cluster_method="tmixture", amp_dur_weight=0.0, polarity_weight=0.0)
    peaks_t, peak_amps, wf, emb, raw = cluster_from_scratch(trace, fs, 0.0, "tmixture", overrides)
    n_raw = int(raw.max()) + 1
    print(f"[{subject}] detected {peaks_t.size} spikes, raw t-mixture -> {n_raw} types")

    keep, pp = post_process(peaks_t, peak_amps, wf, raw, fs)
    n_pp = int(pp.max()) + 1
    kpt = peaks_t[keep]
    kwf = wf[keep]
    print(f"[{subject}] post-processing -> {n_pp} types, {keep.sum()}/{keep.size} spikes")

    merged, merged_of, mlog, groups = sequential_anchor_merge(pp, kwf, "gap")
    n_mg = int(merged.max()) + 1
    print(f"[{subject}] sequential gap-merge -> {n_mg} types, groups {groups}")

    measured = bandpass(trace, fs, 20.0, 10000.0)
    pp_rows = type_table(measured, fs, [kpt[pp == c] for c in range(n_pp)])
    mg_rows = type_table(measured, fs, [kpt[merged == g] for g in range(n_mg)])

    cat = subject.rsplit("_", 1)[-1].capitalize()
    fig_gallery(pp_rows, fs, f"{subject} ({cat}) — post-processed types ({n_pp})",
                os.path.join(out_dir, "gallery_postproc.png"))
    fig_gallery(mg_rows, fs, f"{subject} ({cat}) — after merge ({n_mg})",
                os.path.join(out_dir, "gallery_merged.png"))
    fig_raster(kpt, pp, bounds, fs, f"{subject} — post-processed raster ({n_pp} types)",
               os.path.join(out_dir, "raster_postproc.png"))
    fig_raster(kpt, merged, bounds, fs, f"{subject} — merged raster ({n_mg} types)",
               os.path.join(out_dir, "raster_merged.png"))
    fig_ampdur_kde(pp_rows, f"{subject} — post-processed types: amplitude vs duration KDE",
                   os.path.join(out_dir, "ampdur_kde_postproc.png"))
    fig_ampdur_kde(mg_rows, f"{subject} — merged types: amplitude vs duration KDE",
                   os.path.join(out_dir, "ampdur_kde_merged.png"))

    np.savez(os.path.join(out_dir, "summary.npz"),
             peaks_t=kpt, pp_labels=pp, merged_labels=merged, merged_of=merged_of,
             run_bounds=bounds, run_names=np.array(run_names), fs=fs,
             pp_templates=np.stack([r["template"] for r in pp_rows]),
             merged_templates=np.stack([r["template"] for r in mg_rows]))
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "type", "n_spikes", "p2p_amplitude", "duration_ms"])
        for r in pp_rows:
            w.writerow(["postproc", r["type"], r["n"], f"{r['amplitude']:.1f}", f"{r['duration_ms']:.2f}"])
        for r in mg_rows:
            w.writerow(["merged", r["type"], r["n"], f"{r['amplitude']:.1f}", f"{r['duration_ms']:.2f}"])
    print(f"[{subject}] done in {time.time() - t0:.0f}s -> {out_dir}")
    return dict(subject=subject, n_raw=n_raw, n_pp=n_pp, n_merged=n_mg,
               n_spikes=int(kpt.size), groups=groups)


def main():
    want = set(sys.argv[1:])
    subs = [s for s in SUBJECTS if not want or s[0] in want]
    results = []
    for subject, side in subs:
        try:
            results.append(process_subject(subject, side))
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"[{subject}] FAILED: {e}")
            traceback.print_exc()
    print("\n=== summary ===")
    for r in results:
        print(f"  {r['subject']:16s} spikes={r['n_spikes']:6d}  raw={r['n_raw']:2d} "
              f"post={r['n_pp']:2d} merged={r['n_merged']:2d}  groups={r['groups']}")


if __name__ == "__main__":
    main()
