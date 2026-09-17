#!/usr/bin/env python3
"""Build a library of real MUAP templates -- meant as building blocks for
a simulated EMG trace -- from the DCT-filtered JOINT t-SNE selection
already implemented in dct_muap_compare.py (its "B. joint t-SNE" path,
the one that produces dct_muap_joint_tsne.png), run across every subject
(test0_myopathy .. test5_neuro), Tibialis anterior.

For each subject:
    1. DETECTION -- per EMGRUN file, unchanged
       (joint_run_spike_sort.run_detection).
    2. DCT HIGH-PASS -- every extracted waveform has its DCT_N_LOW lowest
       coefficients zeroed (spike_sort.features.dct_highpass_waveforms):
       DC + slowest cosines = residual baseline offset / slow drift.
    3. JOINT t-SNE + k-means -- one embedding/clustering over every run's
       (DCT-filtered) waveforms pooled together (dct_muap_compare.
       sort_joint), so a type means the same shape across every EMGRUN in
       that subject, not just within one.
    4. SELECT -- sort_joint already drops types with fewer than
       MIN_SPIKES_PER_TYPE pooled spikes (dct_muap_compare.
       keep_big_types) -- these survivors are "the selected MUAPs".

Each selected type's mean-template waveform becomes one library entry,
tagged with its subject/diagnosis category, sampling rate, pooled trial
count, firing rate, and duration/amplitude/phases/turns
(spike_sort.metrics.muap_metrics). Saved under python/muap_library/:

    muap_library.npz          -- every template (+ its pooled individual
                                  trials, for later variability sampling)
                                  and per-type metadata, flat-keyed
                                  <subject>_type<c>_*
    muap_library.csv          -- the same metadata as one row per
                                  template, for quick browsing/filtering
    muap_library_gallery.png  -- every kept template plotted (grey trials
                                  + black mean), grouped/colored by
                                  subject, as a visual sanity check

Edit SUBJECTS below to add/remove subjects, or PARAM_OVERRIDES /
DCT_N_LOW / MIN_SPIKES_PER_TYPE to change the selection itself -- this
mirrors dct_muap_compare.py's own config, kept in sync deliberately so
"the selected MUAPs" means the same thing in both places.

Usage:
    python build_muap_library.py
"""
import copy
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from call_spike_sort_emg import BASE_PARAMS
from dct_muap_compare import MIN_SPIKES_PER_TYPE, PARAM_OVERRIDES, sort_joint
from joint_run_spike_sort import run_detection
from spike_sort.detection import bandpass, extract_waveforms
from spike_sort.features import dct_highpass_waveforms
from spike_sort.io import load_signal, lookup_sample_rate
from spike_sort.metrics import muap_metrics
from spike_sort.plotting import CLUSTER_COLORS, plot_muap_panel

# No DCT high-pass anywhere in this build -- 0 makes dct_highpass_waveforms a no-op, so
# both clustering and the mean templates see the raw measurement-band waveform.
DCT_N_LOW = 0

# Half-window (ms) used to re-cut each clustered spike's waveform for the mean-template /
# gallery display. Clustering still runs on the narrower params["window_ms"] snippets;
# only the per-type mean MUAP shown/saved is built from this wider window.
TEMPLATE_WINDOW_MS = 10.0

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "EMG")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "muap_library_curated")

# (subject folder name, side) -- the Tibialis-anterior side actually recorded for each,
# same layout used throughout this project's other multi-subject scripts.
SUBJECTS = [
    ("test0_myopathy", "Left"),
    ("test1_normal", "Left"),
    ("test2_normal", "Left"),
    ("test3_myopathy", "Right"),
    ("test4_neuro", "Left"),
    ("test5_neuro", "Left"),
]

# Curated subset (visual review, GKC): only these per-subject joint-t-SNE type indices
# are kept in the final library. A subject key mapping to None keeps all of its types.
KEEP_TYPES = {
    "test0_myopathy": {0, 1, 2, 3, 4, 5, 6, 7, 9, 11},
    "test1_normal": {3, 4},
    "test2_normal": {0, 1, 2, 3, 5, 6, 7, 8},
    "test3_myopathy": {4, 5},
    "test4_neuro": {0, 1, 2, 3, 4, 5},
    "test5_neuro": {0, 2, 4},
}

GALLERY_MAX_COLS = 10


def category_from_subject(subject):
    """"test0_myopathy" -> "Myopathy", "test5_neuro" -> "Neuro"."""
    return subject.rsplit("_", 1)[-1].capitalize()


def wide_template_waveforms(run_files, params, joint, window_ms):
    """Re-cut a +/-window_ms snippet for every spike that survived joint
    clustering, from the same measurement-band signal detection used, so
    per-type mean MUAPs can be built on a wider window than the one
    clustering ran on.

    `joint` carries the pooled spikes in run-file order: joint["peaks_t"]
    (spike time, s), joint["run_id"] (index into run_files) and
    joint["labels"], all the same length. Spike sample indices are
    reconstructed exactly from peaks_t (t = start + k/fs in detect_one_run).

    The clustering extraction already used a baseline half-window of
    params["baseline_window_ms"] (10 ms by default), so every surviving
    spike is already at least that far from both signal edges; requesting
    the same or a smaller half-window here therefore drops nothing and the
    returned rows stay 1:1 with `joint`'s spikes. Returns an
    (n_pooled_spikes, 2*half_win) array in joint order, plus half_win."""
    max_baseline_ms = max(window_ms, params["baseline_window_ms"])
    if max_baseline_ms > params["baseline_window_ms"]:
        raise ValueError(
            f"template window {window_ms} ms needs a baseline half-window > the "
            f"{params['baseline_window_ms']} ms clustering used, so some clustered "
            f"spikes would be dropped near the signal edges and rows would misalign")

    n = len(joint["peaks_t"])
    wide = None
    half_win = None
    for i, filename in enumerate(run_files):
        sel = np.where(joint["run_id"] == i)[0]
        if sel.size == 0:
            continue
        fs = params["fs"] if params["fs"] is not None else lookup_sample_rate(filename)
        signal = load_signal(filename, fs, params["start"], params["duration"])
        measured = bandpass(signal, fs, params["measure_band"][0], params["measure_band"][1])
        pk = np.rint((joint["peaks_t"][sel] - params["start"]) * fs).astype(int)
        kept, wf, half_win = extract_waveforms(measured, pk, fs, window_ms, window_ms)
        if wf.shape[0] != sel.size:
            raise RuntimeError(
                f"{filename}: wide re-extraction kept {wf.shape[0]} of {sel.size} clustered "
                f"spikes -- edge drop, rows would misalign with labels")
        if wide is None:
            wide = np.zeros((n, wf.shape[1]))
        wide[sel] = wf
    return wide, half_win


def build_subject_library(subject, side):
    """Run detection + DCT high-pass + joint t-SNE/k-means selection for
    one subject's Tibialis_anterior/<side> directory. Returns a list of
    entry dicts, one per surviving joint type. Clustering runs on the
    params["window_ms"] snippets; each type's mean MUAP is then built from
    a wider +/-TEMPLATE_WINDOW_MS re-cut of the same spikes."""
    run_dir = os.path.join(DATA_ROOT, subject, "EMG", "Tibialis_anterior", side)
    prev_cwd = os.getcwd()
    os.chdir(run_dir)  # Header.txt / EMGRUN__*.txt are resolved relative to cwd throughout
    try:
        run_files = sorted(glob.glob("EMGRUN__*.txt"))
        params = copy.deepcopy(BASE_PARAMS)
        params.update(PARAM_OVERRIDES)

        print(f"[{subject}/{side}] detecting spikes in {len(run_files)} run(s)...")
        runs = run_detection(run_files, params)
        for r in runs:
            r["waveforms"] = dct_highpass_waveforms(r["waveforms"], DCT_N_LOW)

        print(f"[{subject}/{side}] joint t-SNE + k-means over the pooled "
              f"waveforms (no DCT filtering)...")
        joint = sort_joint(runs, params)
        print(f"[{subject}/{side}] {joint['waveforms'].shape[0]} pooled spikes -> "
              f"{joint['n_types']} selected type(s) (>= {MIN_SPIKES_PER_TYPE} spikes each)")

        print(f"[{subject}/{side}] re-cutting +/-{TEMPLATE_WINDOW_MS:g} ms windows for the "
              f"mean-template display...")
        wide_waveforms, wide_half_win = wide_template_waveforms(
            run_files, params, joint, TEMPLATE_WINDOW_MS)
    finally:
        os.chdir(prev_cwd)

    category = category_from_subject(subject)
    t_wave_ms = (np.arange(-wide_half_win, wide_half_win) / joint["fs"]) * 1000

    keep = KEEP_TYPES.get(subject, None)  # None -> keep every type for this subject

    entries = []
    dropped = []
    for c in range(joint["n_types"]):
        if keep is not None and c not in keep:
            dropped.append(c)
            continue
        mask = joint["labels"] == c
        trial_waveforms = wide_waveforms[mask]
        mean_wf = trial_waveforms.mean(axis=0)
        m = muap_metrics(mean_wf, joint["fs"])
        entries.append(dict(
            subject=subject, category=category, muscle="Tibialis_anterior", side=side,
            type_idx=c, fs=joint["fs"], t_wave_ms=t_wave_ms, mean_wf=mean_wf,
            trial_waveforms=trial_waveforms, n_spikes=int(mask.sum()),
            n_runs_present=int(len(np.unique(joint["run_id"][mask]))),
            firing_rate_hz=float(mask.sum() / joint["total_dur"]),
            duration_ms=m["duration_ms"], amplitude=m["amplitude"],
            phases=m["phases"], turns=m["turns"],
        ))
    print(f"[{subject}/{side}] curated: kept {len(entries)} type(s)"
          + (f", dropped {dropped}" if dropped else ""))
    return entries


def save_library_npz(entries, out_path):
    data = {}
    for e in entries:
        key = f"{e['subject']}_type{e['type_idx']}"
        data[f"{key}_waveform"] = e["mean_wf"]
        data[f"{key}_trial_waveforms"] = e["trial_waveforms"]
        data[f"{key}_t_wave_ms"] = e["t_wave_ms"]
        data[f"{key}_fs"] = e["fs"]
        data[f"{key}_n_spikes"] = e["n_spikes"]
        data[f"{key}_firing_rate_hz"] = e["firing_rate_hz"]
        data[f"{key}_duration_ms"] = e["duration_ms"]
        data[f"{key}_amplitude"] = e["amplitude"]
        data[f"{key}_category"] = e["category"]
    np.savez_compressed(out_path, **data)
    print(f"Saved {len(entries)} template(s) to {out_path}")


def save_library_csv(entries, out_path):
    fields = ["subject", "category", "muscle", "side", "type_idx", "n_spikes",
              "n_runs_present", "fs", "duration_ms", "amplitude", "firing_rate_hz",
              "phases", "turns"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for e in entries:
            writer.writerow({k: e[k] for k in fields})
    print(f"Saved library index to {out_path}")


def save_library_gallery(entries, out_path):
    n = max(len(entries), 1)
    n_cols = min(GALLERY_MAX_COLS, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.0 * n_rows),
                              squeeze=False)
    subjects = sorted({e["subject"] for e in entries})
    subject_color = {s: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, s in enumerate(subjects)}
    rng = np.random.default_rng(0)

    for idx in range(n_rows * n_cols):
        ax = axes[idx // n_cols][idx % n_cols]
        if idx >= len(entries):
            ax.axis("off")
            continue
        e = entries[idx]
        color = subject_color[e["subject"]]
        m = muap_metrics(e["mean_wf"], e["fs"])
        title = (f"{e['subject']} type {e['type_idx']}\n"
                 f"n={e['n_spikes']}, {e['duration_ms']:.1f} ms, {e['amplitude']:.0f}")
        plot_muap_panel(ax, e["trial_waveforms"], e["mean_wf"], m, e["t_wave_ms"], color, title,
                         muap_ylim=None, negative_up=True, rng=rng, mean_color=color,
                         ylabel="Amplitude" if idx % n_cols == 0 else None)

    handles = [plt.Line2D([0], [0], color=subject_color[s], lw=2, label=s) for s in subjects]
    fig.legend(handles=handles, loc="upper center", ncol=len(subjects), fontsize=8,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"MUAP library (curated) — {len(entries)} templates across "
                 f"{len(subjects)} subject(s) (no DCT filtering, per-subject joint t-SNE "
                 f"selection, ±{TEMPLATE_WINDOW_MS:g} ms mean-template window)",
                 fontsize=12, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved library gallery to {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_entries = []
    for subject, side in SUBJECTS:
        all_entries.extend(build_subject_library(subject, side))

    print(f"\n{len(all_entries)} total selected MUAP template(s) across {len(SUBJECTS)} "
          f"subject(s)")

    save_library_npz(all_entries, os.path.join(OUT_DIR, "muap_library.npz"))
    save_library_csv(all_entries, os.path.join(OUT_DIR, "muap_library.csv"))
    save_library_gallery(all_entries, os.path.join(OUT_DIR, "muap_library_gallery.png"))


if __name__ == "__main__":
    main()
