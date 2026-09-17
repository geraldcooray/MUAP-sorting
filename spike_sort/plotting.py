"""Detection-diagnostic and spike-sorted-MUAP plots."""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

CLUSTER_COLORS = plt.get_cmap("tab10").colors


def _covariance_ellipse_params(x, y, n_std=2.0):
    """Return (center, width, height, angle_degrees) describing an ellipse
    spanning `n_std` standard deviations along each principal axis of the
    2D point cloud (x, y) -- i.e. the axis-aligned-in-its-own-frame
    "n_std-sigma" contour of a Gaussian fit to the points, found via
    eigendecomposition of their 2x2 covariance matrix (eigenvectors give
    the ellipse's orientation, eigenvalues its principal-axis variances).
    Returns None if there are too few points (<2) to estimate a
    covariance, or if the covariance is degenerate."""
    if len(x) < 2:
        return None
    cov = np.cov(x, y)
    eigvals, eigvecs = np.linalg.eigh(cov)
    if np.any(eigvals < 0):
        return None
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    return (float(np.mean(x)), float(np.mean(y))), width, height, angle


def plot_feature_space_3d(ax, features, labels, dims=(0, 1, 2), axis_labels=None,
                           elev=20, azim=-60, title=None, legend=True):
    """Scatter the first 3 shape-feature dimensions (whichever ones -- PCA
    components, wavelet coefficients, or a t-SNE embedding, depending on
    how `features` was built) in a 3D axes, one point per spike, colored
    by its assigned type. This is a genuine 3D plot (mplot3d), not a 2D
    projection -- `elev`/`azim` set the initial camera angle, and it's
    still rotatable interactively if the figure is shown rather than only
    saved to a static image.

    `dims` selects which 3 columns of `features` to plot (default: the
    first 3, which are the raw shape-feature columns before the
    amplitude/duration columns build_features appends). Silently no-ops
    (with a text note on the axes) if `features` has fewer than 3 columns
    to plot."""
    if features.shape[1] < 3:
        ax.text2D(0.5, 0.5, "fewer than 3 shape dimensions\n(nothing to plot)",
                   ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_axis_off()
        return

    n_types = labels.max() + 1
    d0, d1, d2 = dims
    for c in range(n_types):
        mask = labels == c
        ax.scatter(features[mask, d0], features[mask, d1], features[mask, d2],
                   s=12, alpha=0.7, color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                   label=f"type {c}", depthshade=True)

    axis_labels = axis_labels or [f"dim {d + 1}" for d in dims]
    ax.set_xlabel(axis_labels[0], fontsize=8)
    ax.set_ylabel(axis_labels[1], fontsize=8)
    ax.set_zlabel(axis_labels[2], fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title, fontsize=9)
    if legend:
        ax.legend(loc="upper left", fontsize=6, ncol=2, markerscale=1.5)


def plot_feature_space_2d(ax, features, labels, dim_x, dim_y, axis_labels=None,
                           title=None, legend=True, show_ellipses=True, ellipse_n_std=2.0):
    """Scatter two shape-feature dimensions (whichever ones -- PCA
    components, wavelet coefficients, or a t-SNE embedding, depending on
    how `features` was built) in a standard 2D axes, one point per spike,
    colored by its assigned type. A 2D pairwise projection like this
    avoids the occlusion/perspective ambiguity of a 3D scatter -- nothing
    is ever hidden behind something else along the missing third axis --
    at the cost of needing one plot per dimension pair to see the whole
    picture.

    When `show_ellipses` is set, each type also gets a dashed
    `ellipse_n_std`-sigma covariance ellipse (see
    `_covariance_ellipse_params`) showing that type's spread/orientation
    in this particular 2D projection -- a quick visual check of how
    tight and how separated the clusters are here, independent of
    whichever algorithm actually assigned the labels (k-means or
    otherwise: the ellipse is fit directly to each type's points in this
    projection, not to any model's internal cluster shape).

    Silently no-ops (with a text note on the axes) if `features` doesn't
    have both `dim_x` and `dim_y` as columns."""
    if max(dim_x, dim_y) >= features.shape[1]:
        ax.text(0.5, 0.5, f"fewer than {max(dim_x, dim_y) + 1} shape "
                           f"dimensions\n(nothing to plot)",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_axis_off()
        return

    n_types = labels.max() + 1
    for c in range(n_types):
        mask = labels == c
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        x, y = features[mask, dim_x], features[mask, dim_y]
        ax.scatter(x, y, s=10, alpha=0.6, color=color, label=f"type {c}", zorder=2)

        if show_ellipses:
            params = _covariance_ellipse_params(x, y, n_std=ellipse_n_std)
            if params is not None:
                center, width, height, angle = params
                ax.add_patch(Ellipse(center, width, height, angle=angle,
                                      edgecolor=color, facecolor="none",
                                      linewidth=1.5, linestyle="--", zorder=3))

    axis_labels = axis_labels or (f"dim {dim_x + 1}", f"dim {dim_y + 1}")
    ax.set_xlabel(axis_labels[0], fontsize=8)
    ax.set_ylabel(axis_labels[1], fontsize=8)
    ax.tick_params(labelsize=7)
    if title:
        ax.set_title(title, fontsize=9)
    if legend:
        ax.legend(loc="best", fontsize=6, ncol=2, markerscale=1.5)


def plot_detection_diagnostic(t, signal, filtered, envelope, threshold, peaks,
                               plot_window, filename, out_path, negative_up=True):
    """Raw signal, detection (highpass) filter output, and rectified/smoothed
    envelope stacked with the threshold and detected spikes overlaid, so
    missed spikes can be diagnosed as sub-threshold (visible bump in the
    envelope that never crosses the dashed line) vs. merged by the
    refractory/min-spacing window (two nearby raw deflections sharing one
    detection marker)."""
    plot_end = t[0] + plot_window if plot_window else t[-1]
    xlim = (t[0], min(plot_end, t[-1]))

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(t, signal, linewidth=0.5, color="0.3")
    axes[0].scatter(t[peaks], signal[peaks], s=16, color="crimson", zorder=3)
    axes[0].set_ylabel("Raw signal")
    axes[0].set_title(f"{os.path.basename(filename)} — detection diagnostic "
                       f"({peaks.size} spikes detected)")

    axes[1].plot(t, filtered, linewidth=0.5, color="0.3")
    axes[1].scatter(t[peaks], filtered[peaks], s=16, color="crimson", zorder=3)
    axes[1].set_ylabel("Detection filter\n(highpass)")

    axes[2].plot(t, envelope, linewidth=0.6, color="steelblue")
    axes[2].axhline(threshold, color="crimson", linestyle="--", linewidth=1,
                     label=f"threshold = {threshold:.3g}")
    axes[2].scatter(t[peaks], envelope[peaks], s=16, color="crimson", zorder=3)
    axes[2].set_ylabel("Rectified +\nsmoothed envelope")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right", fontsize=8)

    if negative_up:
        axes[0].invert_yaxis()
        axes[1].invert_yaxis()

    axes[0].set_xlim(*xlim)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved detection diagnostic to {out_path}")
    plt.show()


def plot_muap_panel(ax, unit_waveforms, mean_wf, m, t_wave_ms, color, title,
                     muap_ylim=(-500, 500), muap_xlim=None, negative_up=True, rng=None,
                     ylabel=None, mean_color="black"):
    """Draw one MUAP superimposition + mean-template panel (all individual
    waveforms of one unit, semi-transparent, plus the mean in black, with
    onset/offset dashed lines and a peak-to-peak amplitude bracket) onto
    `ax`. Factored out of plot_results so other scripts (e.g. a
    multi-run comparison plot) can build the same panel style without
    going through the full plot_results figure layout.

    muap_ylim/muap_xlim: fixed (low, high) axis ranges, or None to
    autoscale. muap_xlim=None falls back to matplotlib's default (the
    waveform's own extraction window). muap_ylim=None is NOT matplotlib's
    default autoscale -- it's explicitly scaled to mean_wf's own range
    only, ignoring the individual trial traces, since those are noisier
    and would otherwise dominate the autoscale and squash the mean
    waveform's shape down to a sliver.

    mean_color: color for the mean-template line and the onset/offset/
    amplitude-bracket annotations (default "black", independent of
    `color`, which is only the individual trial traces) -- lets a caller
    recolor the whole "summary" half of the panel, e.g. to mark a MUAP
    that failed a post-hoc selection in gray instead of black."""
    rng = rng if rng is not None else np.random.default_rng(0)
    show_idx = rng.choice(len(unit_waveforms), size=min(150, len(unit_waveforms)),
                           replace=False)
    for i in show_idx:
        ax.plot(t_wave_ms, unit_waveforms[i], color=color, alpha=0.08,
                linewidth=0.6, zorder=1)
    ax.plot(t_wave_ms, mean_wf, color=mean_color, linewidth=1.8, zorder=3)

    onset_ms, offset_ms = t_wave_ms[m["onset"]], t_wave_ms[m["offset"]]
    ax.axvline(onset_ms, color=color, linestyle="--", linewidth=1, zorder=2)
    ax.axvline(offset_ms, color=color, linestyle="--", linewidth=1, zorder=2)

    bracket_x = t_wave_ms[-1] * 0.95
    ax.annotate("", xy=(bracket_x, m["amp_max"]), xytext=(bracket_x, m["amp_min"]),
                arrowprops=dict(arrowstyle="<->", color=mean_color, linewidth=0.8))
    ax.text(bracket_x, (m["amp_max"] + m["amp_min"]) / 2, f' {m["amplitude"]:.0f}',
            fontsize=7, va="center", color=mean_color)

    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Time (ms)")
    if muap_ylim is not None:
        ax.set_ylim(*muap_ylim)
    else:
        lo, hi = mean_wf.min(), mean_wf.max()
        pad = 0.15 * (hi - lo) if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)
    if muap_xlim is not None:
        ax.set_xlim(*muap_xlim)
    if negative_up:
        ax.invert_yaxis()
    if ylabel:
        ax.set_ylabel(ylabel)


def plot_results(t, signal, peaks_t, peak_amps, labels, waveforms, half_win, fs,
                  filename, out_path, metrics_by_cluster, firing_rates, isi_cvs,
                  plot_window, muap_ylim=(-500, 500), muap_xlim=None, negative_up=True,
                  title_suffix="spike-sorted MUAPs"):
    n_clusters = labels.max() + 1
    colors = [CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in range(n_clusters)]
    t_wave_ms = (np.arange(-half_win, half_win) / fs) * 1000

    fig = plt.figure(figsize=(4.5 * n_clusters, 9))
    gs = fig.add_gridspec(3, n_clusters, height_ratios=[3, 1, 2.4],
                           hspace=0.55, wspace=0.35)
    ax_emg = fig.add_subplot(gs[0, :])
    ax_raster = fig.add_subplot(gs[1, :], sharex=ax_emg)

    ax_emg.plot(t, signal, linewidth=0.4, color="0.4", zorder=1)
    for c in range(n_clusters):
        mask = labels == c
        ax_emg.scatter(peaks_t[mask], peak_amps[mask], s=14, color=colors[c],
                        zorder=2, label=f"type {c} (n={mask.sum()})")
    ax_emg.set_ylabel("Amplitude")
    ax_emg.set_title(f"{os.path.basename(filename)} — {title_suffix}")
    ax_emg.legend(loc="upper right", fontsize=8, ncol=n_clusters)
    plt.setp(ax_emg.get_xticklabels(), visible=False)
    if negative_up:
        ax_emg.invert_yaxis()

    for c in range(n_clusters):
        mask = labels == c
        ax_raster.vlines(peaks_t[mask], c - 0.4, c + 0.4, color=colors[c], linewidth=1)
    ax_raster.set_ylim(-0.5, n_clusters - 0.5)
    ax_raster.set_yticks(range(n_clusters))
    ax_raster.set_yticklabels([f"type {c} ({firing_rates[c]:.1f} Hz, "
                                f"CV {isi_cvs[c]:.2f})" for c in range(n_clusters)])
    ax_raster.set_xlabel("Time (s)")
    plot_end = t[0] + plot_window if plot_window is not None else t[-1]
    ax_raster.set_xlim(t[0], min(plot_end, t[-1]))

    rng = np.random.default_rng(0)
    for c in range(n_clusters):
        ax = fig.add_subplot(gs[2, c])
        mask = labels == c
        unit_waveforms = waveforms[mask]
        mean_wf = unit_waveforms.mean(axis=0)
        m = metrics_by_cluster[c]
        title = (f"type {c} (n={mask.sum()}, {firing_rates[c]:.1f} Hz, "
                 f"CV(ISI) {isi_cvs[c]:.2f})\n"
                 f"dur {m['duration_ms']:.2f} ms, amp {m['amplitude']:.0f}, "
                 f"{m['phases']}ph/{m['turns']}t")
        plot_muap_panel(ax, unit_waveforms, mean_wf, m, t_wave_ms, colors[c], title,
                         muap_ylim=muap_ylim, muap_xlim=muap_xlim, negative_up=negative_up,
                         rng=rng, ylabel="Amplitude (measurement band)" if c == 0 else None)

    fig.suptitle("MUAP superimposition + mean template (dashed = onset/offset, "
                  "bracket = peak-to-peak amplitude)", fontsize=9, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.show()
