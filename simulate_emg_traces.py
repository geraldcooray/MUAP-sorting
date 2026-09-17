#!/usr/bin/env python3
"""Simulate EMG traces by superimposing real MUAP templates from
muap_library/ (see build_muap_library.py) at synthetic firing times, plus
added colored background noise.

For each trace, N_MUAPS_LIST gives how many distinct motor units (MUAP
templates) are used, chosen at random from the whole library (any
subject/type mix). Each unit fires independently as a gamma renewal
process:

    - its own mean firing rate is drawn once, uniformly at random from
      FIRING_RATE_RANGE_HZ (1-10 Hz by default) -- so units in one trace
      don't all fire at the same rate;
    - its inter-spike intervals are then drawn from a Gamma distribution
      with that mean and shape GAMMA_SHAPE (mean ISI = 1/rate; CV of ISI
      = 1/sqrt(GAMMA_SHAPE) -- higher shape = more regular firing, closer
      to a real motor unit than a memoryless Poisson process would be).

The clean signal is every unit's template added in in linear
superposition at its own spike times (overlapping MUAPs simply sum, as
in a real interference pattern). Colored (1/f^NOISE_BETA) background
noise is generated separately and scaled to hit SNR_DB (an amplitude/RMS
based signal-to-noise ratio in dB -- rerun with a different SNR_DB, the
one knob meant to be varied, to make the trace noisier/cleaner).

Outputs, one per N_MUAPS_LIST entry, saved to simulated_traces/:
    simulated_trace_n<K>.png  -- the 10 s trace with each unit's true
                                 spike times marked, color-coded
    simulated_trace_n<K>.npz  -- trace, clean signal, noise, time axis,
                                 fs, and per-unit ground truth (subject/
                                 type origin, firing rate, spike times)
    simulated_traces_manifest.csv -- every unit used in every trace, one
                                 row each, for a quick overview

Usage:
    python simulate_emg_traces.py [--snr-db 6.0]
"""
import argparse
import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np

LIBRARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "muap_library", "muap_library.npz")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulated_traces")

DURATION_S = 120.0
BASELINE_S = 1.0                      # spike-free lead-in prepended to every trace, purely noise
                                       # -- gives the detector a clean segment to estimate the
                                       # noise floor from, instead of the whole (possibly
                                       # spike-dense) record (see detect_spikes'
                                       # noise_baseline_samples)
N_MUAPS_LIST = [3, 5, 10, 20]
MIN_TRIALS = 100                      # only library templates pooled from at least this many
                                       # spikes/trials (n_spikes) are eligible to be drawn for a
                                       # simulated trace -- a data-support/robustness floor on the
                                       # mean waveform used as ground truth (same spirit as
                                       # muap_selection.py's MIN_TRIALS for the real-data pipeline)
FIRING_RATE_RANGE_HZ = (1.0, 10.0)   # each unit's own mean rate is drawn uniformly from this
GAMMA_SHAPE = 10.0                    # ISI gamma shape (regularity); CV = 1/sqrt(shape) ~= 0.32
NOISE_BETA = 1.0                      # colored-noise power-law exponent (1/f^beta); 1.0 = pink
SNR_DB = 30.0                         # signal/noise RMS ratio in dB -- THE knob meant to be varied
                                       # (real EMG spikes are usually clearly visible by eye above
                                       # background noise, so default to a "clean" trace)
SEED = 42

SIM_COLORS = plt.get_cmap("tab20").colors  # up to 20 distinct unit colors (the largest N_MUAPS_LIST)


def load_library(npz_path, min_trials=MIN_TRIALS):
    """Parse muap_library.npz's flat <subject>_type<c>_<field> keys back
    into one dict per template, dropping any pooled from fewer than
    min_trials spikes (n_spikes)."""
    data = np.load(npz_path, allow_pickle=True)
    pattern = re.compile(r"^(.*)_type(\d+)_(\w+)$")
    grouped = {}
    for key in data.files:
        m = pattern.match(key)
        if not m:
            continue
        subject, type_idx, field = m.group(1), int(m.group(2)), m.group(3)
        grouped.setdefault((subject, type_idx), {})[field] = data[key]

    entries = []
    n_dropped = 0
    for (subject, type_idx), fields in grouped.items():
        n_spikes = int(fields["n_spikes"])
        if n_spikes < min_trials:
            n_dropped += 1
            continue
        entries.append(dict(
            subject=subject, type_idx=type_idx, category=str(fields["category"]),
            fs=float(fields["fs"]), mean_wf=fields["waveform"], t_wave_ms=fields["t_wave_ms"],
            trial_waveforms=fields["trial_waveforms"],
            n_spikes=n_spikes, firing_rate_hz=float(fields["firing_rate_hz"]),
            duration_ms=float(fields["duration_ms"]), amplitude=float(fields["amplitude"]),
        ))
    if n_dropped:
        print(f"Dropped {n_dropped} template(s) with fewer than {min_trials} trials")
    return entries


def gamma_spike_train(mean_rate_hz, duration_s, shape, rng):
    """Spike times (s) from a Gamma-renewal process: ISIs ~ Gamma(shape,
    scale) with mean = 1/mean_rate_hz, so CV(ISI) = 1/sqrt(shape)."""
    mean_isi = 1.0 / mean_rate_hz
    scale = mean_isi / shape
    times = []
    t = 0.0
    while True:
        t += rng.gamma(shape, scale)
        if t >= duration_s:
            break
        times.append(t)
    return np.array(times)


def colored_noise(n_samples, fs, beta, rng):
    """Unit-RMS colored noise with power spectrum ~ 1/f^beta (beta=0:
    white, 1: pink, 2: brown/red), via spectral shaping of white noise."""
    white = rng.standard_normal(n_samples)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    scale = np.ones_like(freqs)
    nonzero = freqs > 0
    scale[nonzero] = 1.0 / (freqs[nonzero] ** (beta / 2.0))
    shaped = np.fft.rfft(white) * scale
    noise = np.fft.irfft(shaped, n=n_samples)
    return noise / noise.std()


def build_trace(n_muaps, library, fs, duration_s, snr_db, rng, baseline_s=BASELINE_S):
    """Pick n_muaps random templates from `library`, give each its own
    gamma-process spike train, superimpose them into a clean signal, add
    colored noise scaled to snr_db. The first baseline_s seconds are kept
    spike-free (pure noise) so downstream detection has a known-quiet
    segment to estimate the noise floor from; every unit's spike times
    are confined to (baseline_s, baseline_s + duration_s]. Returns
    (trace, clean, noise, t, unit_records) -- all baseline_s + duration_s
    seconds long."""
    chosen_idx = rng.choice(len(library), size=n_muaps, replace=False)
    n_baseline = int(round(baseline_s * fs))
    n_samples = n_baseline + int(round(duration_s * fs))
    clean = np.zeros(n_samples)
    unit_records = []

    for i in chosen_idx:
        e = library[i]
        rate_hz = rng.uniform(*FIRING_RATE_RANGE_HZ)
        spike_times = gamma_spike_train(rate_hz, duration_s, GAMMA_SHAPE, rng) + baseline_s

        template = e["mean_wf"]
        half_win = len(template) // 2
        for st in spike_times:
            center = int(round(st * fs))
            lo, hi = center - half_win, center - half_win + len(template)
            t_lo, t_hi = max(lo, 0), min(hi, n_samples)
            if t_hi <= t_lo:
                continue
            clean[t_lo:t_hi] += template[t_lo - lo: t_hi - lo]

        unit_records.append(dict(
            subject=e["subject"], type_idx=e["type_idx"], category=e["category"],
            assigned_firing_rate_hz=rate_hz, n_spikes=len(spike_times),
            template_amplitude=e["amplitude"], template_duration_ms=e["duration_ms"],
            spike_times=spike_times,
        ))

    signal_rms = float(np.sqrt(np.mean(clean ** 2)))
    noise = colored_noise(n_samples, fs, NOISE_BETA, rng)
    snr_linear = 10 ** (snr_db / 20.0)
    noise = noise * (signal_rms / snr_linear if signal_rms > 0 else 1.0)

    t = np.arange(n_samples) / fs
    return clean + noise, clean, noise, t, unit_records


def plot_trace(t, trace, unit_records, n_muaps, snr_db, baseline_s, out_path):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axvspan(0, baseline_s, color="0.9", zorder=0, label="baseline (spike-free)")
    ax.plot(t, trace, linewidth=0.4, color="0.4", zorder=1)
    for i, u in enumerate(unit_records):
        color = SIM_COLORS[i % len(SIM_COLORS)]
        amps = np.interp(u["spike_times"], t, trace)
        ax.scatter(u["spike_times"], amps, s=14, color=color, zorder=2,
                   label=f"{u['subject']} type {u['type_idx']} ({u['assigned_firing_rate_hz']:.1f} Hz)")
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Simulated EMG — {n_muaps} MUAPs, SNR={snr_db:.1f} dB, "
                 f"{baseline_s:.0f}+{DURATION_S:.0f} s (baseline+active)", fontsize=11)
    ax.legend(loc="upper right", fontsize=6, ncol=min(len(unit_records) + 1, 5))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved trace plot to {out_path}")


def save_trace_npz(t, trace, clean, noise, fs, unit_records, baseline_s, out_path):
    data = {"t": t, "trace": trace, "clean": clean, "noise": noise, "fs": fs,
            "baseline_s": baseline_s}
    for i, u in enumerate(unit_records):
        key = f"unit{i}_{u['subject']}_type{u['type_idx']}"
        data[f"{key}_spike_times_s"] = u["spike_times"]
        data[f"{key}_firing_rate_hz"] = u["assigned_firing_rate_hz"]
    np.savez_compressed(out_path, **data)
    print(f"Saved trace data to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snr-db", type=float, default=SNR_DB,
                         help="Signal/noise RMS ratio in dB -- lower = noisier")
    parser.add_argument("--duration", type=float, default=DURATION_S,
                         help="Active (spiking) trace length (s), excluding baseline")
    parser.add_argument("--baseline", type=float, default=BASELINE_S,
                         help="Spike-free lead-in prepended to the trace (s)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    library = load_library(LIBRARY_PATH)
    print(f"Loaded {len(library)} MUAP template(s) from {LIBRARY_PATH}")

    manifest_rows = []
    for n_muaps in N_MUAPS_LIST:
        rng = np.random.default_rng(args.seed + n_muaps)  # distinct but reproducible per trace
        trace, clean, noise, t, unit_records = build_trace(
            n_muaps, library, library[0]["fs"], args.duration, args.snr_db, rng, args.baseline)

        print(f"n={n_muaps}: units = " +
              ", ".join(f"{u['subject']}/type{u['type_idx']}@{u['assigned_firing_rate_hz']:.1f}Hz"
                        for u in unit_records))

        out_png = os.path.join(OUT_DIR, f"simulated_trace_n{n_muaps}.png")
        out_npz = os.path.join(OUT_DIR, f"simulated_trace_n{n_muaps}.npz")
        plot_trace(t, trace, unit_records, n_muaps, args.snr_db, args.baseline, out_png)
        save_trace_npz(t, trace, clean, noise, library[0]["fs"], unit_records, args.baseline,
                        out_npz)

        for u in unit_records:
            manifest_rows.append(dict(
                n_muaps_in_trace=n_muaps, subject=u["subject"], category=u["category"],
                type_idx=u["type_idx"], assigned_firing_rate_hz=u["assigned_firing_rate_hz"],
                n_spikes=u["n_spikes"], template_amplitude=u["template_amplitude"],
                template_duration_ms=u["template_duration_ms"], snr_db=args.snr_db,
            ))
        plt.show()

    manifest_path = os.path.join(OUT_DIR, "simulated_traces_manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
