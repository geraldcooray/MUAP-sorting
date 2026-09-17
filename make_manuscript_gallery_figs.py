"""Generate the two K=10 MUAP-gallery figures for the manuscript.

Fig 1 (fig_gallery_features):  true 10 MUAPs  vs  recovered types, one column
       per feature representation (t-SNE, PCA, wavelet), k-means clustering,
       full post-processing.
Fig 2 (fig_gallery_tsne_clustering):  true 10 MUAPs  vs  recovered types with
       t-SNE features and each clustering algorithm (k-means, GMM, Student-t).

The sorted-type mean waveforms are NOT taken from the stored +/-5 ms feature
snippets: the K=10 trace is regenerated exactly (same seed / library / params),
band-pass filtered to the measurement band, and every spike of each sorted type
is re-cut on a +/-10 ms window and averaged -- so the recovered templates span
the same window as the ground-truth library templates. Waveforms are
peak-normalised so shape, not band-dependent amplitude, is what is compared.
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simulate_emg_traces as S
from spike_sort.detection import bandpass
from number_library_gallery import library_index

LIB_INDEX = library_index()  # {"subject_typeC": 1..31}, gallery row-major order

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.join(HERE, "simulated_traces")
LIB = np.load(os.path.join(HERE, "muap_library_curated", "muap_library.npz"), allow_pickle=True)
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "Manuscript", "figures"))
os.makedirs(OUT_DIR, exist_ok=True)

K = 10
WIN_MS = 10.0                 # half-width of every panel AND of the re-averaged templates
MEASURE_BAND = (20.0, 10000.0)  # BASE_PARAMS["measure_band"] -- waveform-extraction filter
_UNIT_RE = re.compile(r"^unit\d+_(.*)_type(\d+)$")


def regenerate_measured_trace():
    """Rebuild the K=10 simulated trace exactly as simulate_emg_traces.py did
    (seed 42+K, curated library, default params) and return the
    measurement-band-filtered signal plus fs."""
    lib = S.load_library(os.path.join(HERE, "muap_library_curated", "muap_library.npz"))
    fs = lib[0]["fs"]
    rng = np.random.default_rng(S.SEED + K)
    trace, _clean, _noise, _t, _recs = S.build_trace(
        K, lib, fs, S.DURATION_S, S.SNR_DB, rng, S.BASELINE_S)
    return bandpass(trace, fs, *MEASURE_BAND), fs


def true_template(unit_name):
    """+/-WIN_MS slice of the library mean template for a `unitN_subject_typeC` name."""
    m = _UNIT_RE.match(unit_name)
    key = f"{m.group(1)}_type{m.group(2)}"
    wf = LIB[f"{key}_waveform"].astype(float)
    t = LIB[f"{key}_t_wave_ms"].astype(float)
    sel = np.abs(t) <= WIN_MS
    return t[sel], wf[sel]


def reaverage_type(measured, fs, peaks_t, mask):
    """Mean of +/-WIN_MS windows (each minus its own mean) cut from `measured`
    around every spike time in peaks_t[mask]."""
    half = int(round(WIN_MS * 1e-3 * fs))
    idx = np.round(np.asarray(peaks_t)[mask] * fs).astype(int)
    idx = idx[(idx - half >= 0) & (idx + half < len(measured))]
    snips = np.stack([measured[p - half:p + half] - measured[p - half:p + half].mean()
                      for p in idx])
    return snips.mean(axis=0), len(idx)


def recovered_mean(npz, true_idx, measured, fs):
    """Re-averaged +/-WIN_MS mean waveform of the sorted type assigned to true
    unit `true_idx` (largest such type if several), plus its purity. None if
    not recovered."""
    labels = npz["labels"]
    est_match = npz["est_match"]
    est_purity = npz["est_purity"]
    peaks_t = npz["peaks_t"]
    cand = [k for k in range(len(est_match)) if est_match[k] == true_idx]
    if not cand:
        return None
    k = max(cand, key=lambda k: int(np.sum(labels == k)))
    w, _n = reaverage_type(measured, fs, peaks_t, labels == k)
    return w, float(est_purity[k])


def norm(x):
    m = np.max(np.abs(x))
    return x / m if m > 0 else x


def centred_axis(n, fs):
    return (np.arange(n) - n / 2) / fs * 1000.0


def build_figure(columns, true_names, measured, fs, out_path, col_titles):
    """columns: list of loaded npz (one per non-true column)."""
    ncol = 1 + len(columns)
    fig, axes = plt.subplots(K, ncol, figsize=(2.0 * ncol, 1.35 * K),
                             sharex=True, squeeze=False)
    for i, uname in enumerate(true_names):
        t_true, wf_true = true_template(uname)
        ax = axes[i][0]
        ax.plot(t_true, norm(wf_true), color="black", lw=1.2)
        short = _UNIT_RE.match(uname)
        key = f"{short.group(1)}_type{short.group(2)}"
        num = LIB_INDEX.get(key, "?")
        ax.set_ylabel(f"MUAP {num}", fontsize=8, rotation=0, ha="right", va="center")
        for j, npz in enumerate(columns):
            axc = axes[i][j + 1]
            rec = recovered_mean(npz, i, measured, fs)
            if rec is None:
                axc.text(0.5, 0.5, "not recovered", ha="center", va="center",
                         fontsize=7, color="0.6", transform=axc.transAxes)
            else:
                w, purity = rec
                tt = centred_axis(len(w), fs)
                axc.plot(tt, norm(w), color="#1f77b4", lw=1.2)
                axc.text(0.03, 0.04, f"purity {purity:.2f}", fontsize=6.5,
                         color="0.2", ha="left", va="bottom", transform=axc.transAxes,
                         bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none",
                                   alpha=0.7))
        for ax_ in axes[i]:
            ax_.set_xlim(-WIN_MS, WIN_MS)
            ax_.axhline(0.0, color="0.85", lw=0.6, zorder=0)
            ax_.set_yticks([])
            for s in ("top", "right", "left"):
                ax_.spines[s].set_visible(False)
            ax_.tick_params(labelsize=6)
    for j, title in enumerate(["True MUAP"] + col_titles):
        axes[0][j].set_title(title, fontsize=9)
    for j in range(ncol):
        axes[-1][j].set_xlabel("ms", fontsize=7)
    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.replace(".png", ".pdf"))
    plt.close(fig)
    print("wrote", out_path)


def load(feature, method):
    p = os.path.join(SIM_DIR, feature, f"sort_n{K}_{method}_full.npz")
    return np.load(p, allow_pickle=True)


def build_true_vs_all_estimated(npz, true_names, measured, fs, out_path, run_title):
    """One row per true unit: its template (left) then every sorted type
    assigned to it (right), re-averaged on +/-WIN_MS, largest type first --
    so over-segmentation (several types for one unit) is visible. A final
    row holds any sorted type matched to no true unit."""
    labels = npz["labels"]
    est_match = npz["est_match"]
    est_purity = npz["est_purity"]
    peaks_t = npz["peaks_t"]

    assigned = {i: [] for i in range(len(true_names))}
    unmatched = []
    for k in range(len(est_match)):
        nsp = int(np.sum(labels == k))
        (assigned[int(est_match[k])] if est_match[k] >= 0 else unmatched).append((k, nsp))
    for i in assigned:
        assigned[i].sort(key=lambda t: -t[1])
    unmatched.sort(key=lambda t: -t[1])

    max_est = max([len(v) for v in assigned.values()] + [len(unmatched), 1])
    rows = [(uname, assigned[i]) for i, uname in enumerate(true_names)]
    if unmatched:
        rows.append((None, unmatched))
    nrow, ncol = len(rows), 1 + max_est

    fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol, 1.3 * nrow),
                             sharex=True, squeeze=False)
    for r, (uname, ests) in enumerate(rows):
        ax0 = axes[r][0]
        if uname is not None:
            t_true, wf_true = true_template(uname)
            m = _UNIT_RE.match(uname)
            num = LIB_INDEX.get(f"{m.group(1)}_type{m.group(2)}", "?")
            ax0.plot(t_true, norm(wf_true), color="black", lw=1.2)
            ax0.set_ylabel(f"MUAP {num}", fontsize=8, rotation=0, ha="right", va="center")
        else:
            ax0.set_ylabel("no true\nmatch", fontsize=7, rotation=0, ha="right", va="center")
        for c in range(max_est):
            axc = axes[r][c + 1]
            if c < len(ests):
                k, nsp = ests[c]
                w, _ = reaverage_type(measured, fs, peaks_t, labels == k)
                axc.plot(centred_axis(len(w), fs), norm(w), color="#1f77b4", lw=1.2)
                axc.text(0.03, 0.04, f"type {k}\npurity {est_purity[k]:.2f}, n={nsp}",
                         fontsize=6, color="0.2", ha="left", va="bottom",
                         transform=axc.transAxes,
                         bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none",
                                   alpha=0.7))
            else:
                axc.set_visible(len(ests) == 0 and c == 0)
                if axc.get_visible():
                    axc.text(0.5, 0.5, "not recovered", ha="center", va="center",
                             fontsize=7, color="0.6", transform=axc.transAxes)
        for ax_ in axes[r]:
            if not ax_.get_visible():
                continue
            ax_.set_xlim(-WIN_MS, WIN_MS)
            ax_.axhline(0.0, color="0.85", lw=0.6, zorder=0)
            ax_.set_yticks([])
            for s in ("top", "right", "left"):
                ax_.spines[s].set_visible(False)
            ax_.tick_params(labelsize=6)
    axes[0][0].set_title("True MUAP", fontsize=9)
    axes[0][1].set_title("Sorted types assigned to that unit "
                         "(largest first)", fontsize=9, loc="left")
    for j in range(ncol):
        axes[-1][j].set_xlabel("ms", fontsize=7)
    fig.suptitle(run_title, fontsize=10, y=0.995)
    fig.tight_layout(h_pad=0.3, w_pad=0.3, rect=[0, 0, 1, 0.975])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def main():
    measured, fs = regenerate_measured_trace()
    ref = load("tsne", "kmeans")
    true_names = [str(x) for x in ref["true_unit_names"]]

    build_figure(
        [load("tsne", "kmeans"), load("pca", "kmeans"), load("wavelet", "kmeans")],
        true_names, measured, fs,
        os.path.join(OUT_DIR, "fig_gallery_features.png"),
        ["t-SNE", "PCA", "Wavelet"],
    )
    build_figure(
        [load("tsne", "kmeans"), load("tsne", "gmm"), load("tsne", "tmixture")],
        true_names, measured, fs,
        os.path.join(OUT_DIR, "fig_gallery_tsne_clustering.png"),
        ["t-SNE + $k$-means", "t-SNE + GMM", "t-SNE + $t$-mixture"],
    )
    build_true_vs_all_estimated(
        load("tsne", "tmixture"), true_names, measured, fs,
        os.path.join(OUT_DIR, "fig_gallery_tsne_alltypes.png"),
        "Simulated K=10 trace, t-SNE + Student's t mixture, full post-processing: "
        "13 sorted types vs 10 true units",
    )


if __name__ == "__main__":
    main()
