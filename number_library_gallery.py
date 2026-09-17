"""Re-plot the 31 curated MUAP templates as a numbered gallery (1..31 in the
same row-major order as muap_library_curated/muap_library_gallery.png), so the
numbers used as row labels in the manuscript's recovery figures have a visible
key. Rebuilt straight from muap_library.npz -- no re-detection.
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = np.load(os.path.join(HERE, "muap_library_curated", "muap_library.npz"), allow_pickle=True)
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "Manuscript", "figures"))
os.makedirs(OUT_DIR, exist_ok=True)

N_COLS = 6


def entry_names():
    return [k[:-9] for k in LIB.files if k.endswith("_waveform")]


def library_index():
    """{template_key: number 1..31} in gallery row-major order."""
    return {name: i + 1 for i, name in enumerate(entry_names())}


def main():
    names = entry_names()
    n = len(names)
    n_rows = int(np.ceil(n / N_COLS))
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(2.3 * N_COLS, 1.9 * n_rows),
                             squeeze=False)
    rng = np.random.default_rng(0)
    for idx in range(n_rows * N_COLS):
        ax = axes[idx // N_COLS][idx % N_COLS]
        if idx >= n:
            ax.axis("off")
            continue
        name = names[idx]
        t = LIB[f"{name}_t_wave_ms"].astype(float)
        mean_wf = LIB[f"{name}_waveform"].astype(float)
        trials = LIB[f"{name}_trial_waveforms"].astype(float)
        take = rng.choice(len(trials), size=min(40, len(trials)), replace=False)
        ax.plot(t, trials[take].T, color="0.8", lw=0.4, alpha=0.7)
        ax.plot(t, mean_wf, color="black", lw=1.3)
        ax.set_title(f"MUAP {idx + 1}", fontsize=7)
        ax.set_xlim(t[0], t[-1])
        # scale to the mean template, not the noisy trials, so the shape isn't flat
        pk = np.max(np.abs(mean_wf))
        if pk > 0:
            ax.set_ylim(-1.8 * pk, 1.8 * pk)
        ax.set_yticks([])
        ax.tick_params(labelsize=6)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        if idx // N_COLS == n_rows - 1 or idx + N_COLS >= n:
            ax.set_xlabel("ms", fontsize=7)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_muap_library_numbered.png")
    fig.savefig(out, dpi=200)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print("wrote", out)
    for name, num in library_index().items():
        print(f"  {num:2d}  {name}")


if __name__ == "__main__":
    main()
