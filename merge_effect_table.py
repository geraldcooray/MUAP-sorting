"""Effect of the fixed-Voronoi sequential gap-merge across the tsne runs.

For every (K, clustering method) it records:
  actual   = K true units
  raw      = clusters from the main pipeline (silhouette-swept)
  post     = types after post-processing (min-ISI / latency-linkage / superposition)
  merged   = types after the sequential anchor gap-merge (no split refinement)
plus true units recovered and mean per-type purity before/after the merge.

Writes simulated_traces/tsne/merge_effect_table.csv incrementally.
"""
import os
import csv
import numpy as np

import simulate_and_sort_curated as base
import spikesort_simulated_traces as sst
from plot_tsne_types import cluster_from_scratch, post_process
from plot_tsne_sequential_merge import sequential_anchor_merge

HERE = os.path.dirname(os.path.abspath(__file__))
TSNE = os.path.join(HERE, "simulated_traces", "tsne")
OUT_CSV = os.path.join(TSNE, "merge_effect_table.csv")

KS = [3, 5, 10, 15, 20, 30]
METHODS = ["kmeans", "gmm", "tmixture"]
TOL = base.MATCH_TOL_MS / 1000.0


def recov_purity(labels, peaks_t, true_units):
    mu = sst.match_to_ground_truth(peaks_t, true_units, TOL)
    match, purity = sst.compute_match_purity(true_units, labels, mu)
    rec = len({m for m in match if m >= 0})
    mean_p = float(np.mean(purity)) if purity else 0.0
    return rec, mean_p


def main():
    _, lookup = base.load_curated_library()
    rows = []
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["K", "method", "actual", "raw", "post", "merged",
                    "rec_post", "purity_post", "rec_merged", "purity_merged"])
        for K in KS:
            trace, fs, true_units, baseline_s = sst.load_simulated_trace(
                os.path.join(TSNE, f"simulated_trace_n{K}.npz"), lookup)
            base.call_spike_sort_emg.SILHOUETTE_MAX_N = K + 10
            for m in METHODS:
                overrides = dict(cluster_method=m, amp_dur_weight=0.0, polarity_weight=0.0,
                                 threshold_mad=sst.DETECTION_THRESHOLD_MAD)
                pt, pa, wf, emb, raw = cluster_from_scratch(trace, fs, baseline_s, m, overrides)
                n_raw = int(raw.max()) + 1
                keep, pp = post_process(pt, pa, wf, raw, fs)
                n_pp = int(pp.max()) + 1
                kpt, kwf = pt[keep], wf[keep]
                merged, merged_of, mlog, groups = sequential_anchor_merge(pp, kwf, "gap")
                n_mg = int(merged.max()) + 1
                rp, pp_pur = recov_purity(pp, kpt, true_units)
                rm, mg_pur = recov_purity(merged, kpt, true_units)
                if K == 10 and m == "kmeans":
                    a = np.array([e["anchor"] for e in mlog])
                    o = np.array([e["other"] for e in mlog])
                    v = np.array([e["verdict"] for e in mlog])
                    np.savez(os.path.join(TSNE, "merge_effect_n10_kmeans.npz"),
                             peaks_t=kpt, pp_labels=pp, merged_labels=merged,
                             merged_of=merged_of, fs=fs,
                             merge_log_anchor=a, merge_log_other=o, merge_log_verdict=v)
                row = [K, m, K, n_raw, n_pp, n_mg, rp, round(pp_pur, 3), rm, round(mg_pur, 3)]
                w.writerow(row)
                fh.flush()
                rows.append(row)
                print(f"K={K:2d} {m:8s}  raw={n_raw:2d} post={n_pp:2d} merged={n_mg:2d} "
                      f"| rec {rp}->{rm} of {K}, purity {pp_pur:.2f}->{mg_pur:.2f}  "
                      f"groups={groups}")
    print("\nwrote", OUT_CSV)


if __name__ == "__main__":
    main()
