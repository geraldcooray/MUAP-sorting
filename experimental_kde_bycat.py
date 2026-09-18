"""Amplitude-vs-duration KDEs split by clinical category (Myopathy / Normal /
Neuro), two figures of three subplots each:

  fig_exp_kde_bycat_types.png    one point per post-processed type
                                 (mean-template p2p amplitude / duration)
  fig_exp_kde_bycat_spikes.png   one point per individual detected spike
                                 (its own +/-5 ms p2p amplitude / FWHM duration)

Saved to Manuscript/figures/ and data/EMG/postprocessed_summary/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from spike_sort.detection import bandpass, extract_waveforms
from spike_sort.features import spike_amplitude_duration
from spike_sort.io import load_signal, lookup_sample_rate
from spike_sort.metrics import muap_metrics
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "..", "data", "EMG")
MAN_FIG = os.path.abspath(os.path.join(HERE, "..", "Manuscript", "figures"))
SUM_DIR = os.path.join(DATA_ROOT, "postprocessed_summary")
MEASURE_BAND = (20.0, 10000.0)
WIN_MS = 5.0

SUBJECTS = [
    ("test0_myopathy", "Left"), ("test1_normal", "Left"), ("test2_normal", "Left"),
    ("test3_myopathy", "Right"), ("test4_neuro", "Left"), ("test5_neuro", "Left"),
]
CATS = ["Myopathy", "Normal", "Neuro"]
CAT_COLOR = {"Myopathy": "#d62728", "Normal": "#1f77b4", "Neuro": "#2ca02c"}


def cat_of(subject):
    return subject.rsplit("_", 1)[-1].capitalize()


def concat_trace(run_dir):
    fs = lookup_sample_rate(sorted(glob.glob(os.path.join(run_dir, "EMGRUN__*.txt")))[0])
    segs = [load_signal(f, fs, 0.0, None)
            for f in sorted(glob.glob(os.path.join(run_dir, "EMGRUN__*.txt")))]
    return np.concatenate(segs), fs


def gather():
    """Per category: list of (amp, dur) per post-proc type (+ weight),
    and arrays of per-spike (amp, dur)."""
    types = {c: {"amp": [], "dur": [], "w": []} for c in CATS}
    spikes = {c: {"amp": [], "dur": []} for c in CATS}
    for subject, side in SUBJECTS:
        cat = cat_of(subject)
        run_dir = os.path.join(DATA_ROOT, subject, "EMG", "Tibialis_anterior", side)
        d = np.load(os.path.join(run_dir, "postprocessed", "summary.npz"), allow_pickle=True)
        fs = float(d["fs"])
        trace, fs2 = concat_trace(run_dir)
        assert abs(fs - fs2) < 1e-6
        measured = bandpass(trace, fs, *MEASURE_BAND)

        pt = d["peaks_t"]
        pp = d["pp_labels"]
        pk = np.round(pt * fs).astype(int)
        _, wf, _ = extract_waveforms(measured, pk, fs, WIN_MS, WIN_MS)
        amp_s, dur_s, _ = spike_amplitude_duration(wf, fs)
        spikes[cat]["amp"].append(amp_s)
        spikes[cat]["dur"].append(dur_s)

        # keep pp<->wf row alignment (extract_waveforms can drop edge spikes)
        keep_edge = (pk - int(round(WIN_MS * 1e-3 * fs)) >= 0) & \
                    (pk + int(round(WIN_MS * 1e-3 * fs)) < measured.size)
        pp_k = pp[keep_edge]
        T = d["pp_templates"]
        for c in range(int(pp.max()) + 1):
            m = muap_metrics(T[c], fs)
            types[cat]["amp"].append(float(m["amplitude"]))
            types[cat]["dur"].append(float(m["duration_ms"]))
            types[cat]["w"].append(int((pp_k == c).sum()))
    for c in CATS:
        spikes[c]["amp"] = np.concatenate(spikes[c]["amp"])
        spikes[c]["dur"] = np.concatenate(spikes[c]["dur"])
    return types, spikes


def _panel(ax, amp, dur, wt, title, color, xlim, ylim, scatter_n=None):
    amp = np.asarray(amp, float)
    dur = np.asarray(dur, float)
    kde = gaussian_kde(np.vstack([amp, dur]), weights=(None if wt is None else np.asarray(wt, float)))
    gx, gy = np.meshgrid(np.linspace(*xlim, 200), np.linspace(*ylim, 200))
    dens = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    cf = ax.contourf(gx, gy, dens, levels=12, cmap="magma")
    ax.contour(gx, gy, dens, levels=6, colors="white", linewidths=0.3, alpha=0.5)
    if scatter_n is None:
        ax.scatter(amp, dur, s=18 + 120 * (np.asarray(wt) / max(np.max(wt), 1)),
                   facecolor="none", edgecolor="cyan", linewidths=1.1, zorder=5)
    else:
        rng = np.random.default_rng(0)
        idx = rng.choice(amp.size, size=min(scatter_n, amp.size), replace=False)
        ax.scatter(amp[idx], dur[idx], s=3, color="cyan", alpha=0.25, linewidths=0, zorder=5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(f"{title}  (n={amp.size})", fontsize=10, color=color)
    ax.set_xlabel("peak-to-peak amplitude (µV)")
    return cf


def fig_types(types):
    allamp = np.concatenate([types[c]["amp"] for c in CATS])
    alldur = np.concatenate([types[c]["dur"] for c in CATS])
    xlim = (0, allamp.max() * 1.1)
    ylim = (0, alldur.max() * 1.1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    cf = None
    for i, c in enumerate(CATS):
        cf = _panel(ax[i], types[c]["amp"], types[c]["dur"], types[c]["w"], c,
                    CAT_COLOR[c], xlim, ylim)
    ax[0].set_ylabel("mean duration (ms)")
    fig.colorbar(cf, ax=ax, label="probability density", shrink=0.85)
    fig.suptitle("Post-processed types — mean p2p amplitude vs duration KDE, by category",
                 fontsize=11)
    for out in (os.path.join(MAN_FIG, "fig_exp_kde_bycat_types.png"),
                os.path.join(SUM_DIR, "kde_bycat_types.png")):
        fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(os.path.join(MAN_FIG, "fig_exp_kde_bycat_types.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_spikes(spikes):
    allamp = np.concatenate([spikes[c]["amp"] for c in CATS])
    alldur = np.concatenate([spikes[c]["dur"] for c in CATS])
    xlim = (0, np.percentile(allamp, 99))
    ylim = (0, np.percentile(alldur, 99))
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    cf = None
    for i, c in enumerate(CATS):
        cf = _panel(ax[i], spikes[c]["amp"], spikes[c]["dur"], None, c,
                    CAT_COLOR[c], xlim, ylim, scatter_n=800)
    ax[0].set_ylabel("FWHM duration (ms)")
    fig.colorbar(cf, ax=ax, label="probability density", shrink=0.85)
    fig.suptitle("Individual spikes — own p2p amplitude vs FWHM duration KDE, by category",
                 fontsize=11)
    for out in (os.path.join(MAN_FIG, "fig_exp_kde_bycat_spikes.png"),
                os.path.join(SUM_DIR, "kde_bycat_spikes.png")):
        fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(os.path.join(MAN_FIG, "fig_exp_kde_bycat_spikes.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    types, spikes = gather()
    for c in CATS:
        print(f"{c:9s}  {len(types[c]['amp'])} types  {spikes[c]['amp'].size} spikes")
    fig_types(types)
    fig_spikes(spikes)
    print("wrote fig_exp_kde_bycat_types / fig_exp_kde_bycat_spikes to", MAN_FIG)


if __name__ == "__main__":
    main()
