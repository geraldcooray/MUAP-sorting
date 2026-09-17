#!/usr/bin/env python3
"""Sequential anchor-based t-SNE merge of post-processed MUAP types.

Pipeline: full t-SNE + k-means sort WITH post-processing (min-ISI cleanup
-> latency-linkage merge -> superposition-composite drop) of the curated
n<N> simulated trace (exactly plot_tsne_types.cluster_from_scratch +
post_process).

INITIAL SPLIT-REFINEMENT (--split, default gap), run before any merging:
for every post-processed type, t-SNE-embed its own +/-5 ms trial
waveforms to 2D, k-means it (k=2), and keep the split only if the gap
statistic on that 2D embedding prefers k=2 over k=1 (Gap(2) > Gap(1)).
Repeat over all current types until a whole pass splits nothing -- so a
type carrying several sub-populations is broken into several smaller
types. (--split tsne uses muap_split_refinement's single-pass silhouette
rule instead; --split none skips it.)

Then a SEQUENTIAL pairwise merge on the split-refined types:

  anchor = type 0. Test anchor vs type 1: pool the two types' +/-5 ms
  trial waveforms and decide whether the pool is one cluster or two
  (--test):
    gap  (default) -- gap statistic (Tibshirani, Walther & Hastie 2001)
                      on the leading principal components of the pool:
                      Gap(k) = mean_b log W*_kb - log W_k for k in {1, 2},
                      with W*_kb from uniform reference sets over the
                      PCA-aligned bounding box. The pool is ONE cluster
                      when Gap(1) >= Gap(2) ("take the larger").
    tsne          -- re-embed the pool alone with a fresh t-SNE, k-means
                      k=2; one cluster when the split silhouette <
                      muap_split_refinement.SPLIT_SILHOUETTE_MIN.
  If one cluster -> merge the pair and the anchor round ends. Otherwise
  test anchor vs type 2, ... vs type n-1, stopping at the FIRST merge (or
  when the last type has been tested).

  The anchor then advances to the smallest type index not yet merged into
  an earlier anchor's group: no merge -> next index; anchor merged with k
  -> k is removed and the next smallest survivor becomes the anchor. This
  repeats until every surviving type has had its turn as anchor.

Each anchor round merges at most one partner, so a merged group has at
most two members. Merges are transitive only through shared membership
(they cannot chain here because an absorbed type never becomes an anchor
or a candidate again).

FINAL SUPERPOSITION RE-CHECK (--post-superposition, default drop): run
spike_sort.superposition.find_superposition_candidates once more on the
merged type templates and drop any merged type that is the time-shifted
sum of two other merged types (post-processing did this earlier, before
the split/merge changed the type set). Dropped types' spikes are removed;
their Voronoi cell goes grey ("X").

The Voronoi tessellation of the post-processed type centroids in the
fixed t-SNE space is built ONCE; merged cells simply share a colour --
the tessellation is never recomputed after a merge.

Outputs in simulated_traces/tsne/  (<tag> = <method>_<test>[_s<split>[-<split-space>]]):
  seqmerge_n<N>_<tag>_voronoi.png         one Voronoi panel (split-type tessellation,
                                          built once), cells coloured by merged group
  seqmerge_n<N>_<tag>_gallery_true.png    mean trace-cut per TRUE (ground-truth) unit
  seqmerge_n<N>_<tag>_gallery_est.png     mean MUAP per post-processed type (pre-split)
  seqmerge_n<N>_<tag>_gallery_split.png   mean MUAP per split-refined type (if --split)
  seqmerge_n<N>_<tag>_gallery_merged.png  mean MUAP per merged type
  seqmerge_n<N>_<tag>_gallery_final.png   mean MUAP per surviving type after the final
                                          superposition re-check (only if it drops any)
  seqmerge_n<N>_<tag>_data.npz            labels, group map, split + merge + superposition logs

Usage:
    python plot_tsne_sequential_merge.py --n 10 --method kmeans --test gap \\
        --split gap --split-space pca
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import Voronoi

import muap_split_refinement as msr
import simulate_and_sort_curated as base
import spikesort_simulated_traces as sst
from sklearn.cluster import KMeans
from plot_tsne_types import cluster_from_scratch, post_process
from plot_tsne_voronoi import voronoi_finite_polygons_2d
from plot_tsne_voronoi_merge import gallery, gallery_true, purity_vs_truth
from spike_sort.clustering import cluster_spikes, relabel_dense
from spike_sort.features import shape_features_tsne
from spike_sort.io import load_signal
from spike_sort.superposition import find_superposition_candidates, format_superposition_report

CMAP = plt.get_cmap("tab20")

# --- gap-statistic test (Tibshirani, Walther & Hastie 2001) --------------------
GAP_N_REFS = 50        # Monte-Carlo reference datasets per k
GAP_VAR_FRAC = 0.90    # keep PCs of the pooled waveforms up to this cumulative variance
GAP_MAX_PC = 10        # ... capped here
GAP_MIN_SPIKES = 10    # pools smaller than this are declared one cluster without testing
GAP_SEED = 42


def _log_within(data, k, rs):
    """log of the pooled within-cluster sum of squares for a k-means fit
    (k=1 -> distance to the global centroid)."""
    if k == 1:
        return float(np.log(((data - data.mean(axis=0)) ** 2).sum() + 1e-12))
    km = KMeans(n_clusters=k, n_init=10, random_state=rs).fit(data)
    return float(np.log(km.inertia_ + 1e-12))


def pca_reduce(waveforms, var_frac=GAP_VAR_FRAC, max_pc=GAP_MAX_PC):
    """Centre `waveforms` and project onto the leading principal
    components up to `var_frac` cumulative variance (>=2, capped at
    `max_pc`). Returns (scores, n_pc)."""
    x0 = waveforms - waveforms.mean(axis=0)
    _, s, vt = np.linalg.svd(x0, full_matrices=False)
    cum = np.cumsum(s ** 2) / max((s ** 2).sum(), 1e-12)
    n_pc = int(np.clip(np.searchsorted(cum, var_frac) + 1, 2, min(max_pc, len(s))))
    return x0 @ vt[:n_pc].T, n_pc


def _w_from_labels(data, labels):
    return float(sum(((data[labels == c] - data[labels == c].mean(axis=0)) ** 2).sum()
                     for c in np.unique(labels)) + 1e-12)


def gap_one_vs_two(pool_waveforms, n_refs=GAP_N_REFS, seed=GAP_SEED, fixed_two=None):
    """Gap statistic for k=1 and k=2 on the pooled waveforms, reduced to
    their leading principal components. For each k,
        Gap(k) = mean_b log W*_kb  -  log W_k
    where W_k is the observed within-cluster dispersion and W*_kb the same
    on `n_refs` uniform reference sets drawn over the (PCA-aligned)
    bounding box of the data -- Tibshirani et al.'s method (b). The
    observed k=2 dispersion uses a fresh k-means split unless `fixed_two`
    (a length-n 0/1 labelling) is given, in which case that partition is
    scored instead (a suboptimal given partition can only lower Gap(2), so
    the test stays conservative). Returns ({1: gap1, 2: gap2}, {1: s1,
    2: s2}, n_pc)."""
    x, n_pc = pca_reduce(pool_waveforms)
    lo, hi = x.min(axis=0), x.max(axis=0)
    rng = np.random.default_rng(seed)

    obs = {1: _log_within(x, 1, seed),
           2: (np.log(_w_from_labels(x, np.asarray(fixed_two))) if fixed_two is not None
               else _log_within(x, 2, seed))}
    ref = {1: [], 2: []}
    for b in range(n_refs):
        z = rng.uniform(lo, hi, size=x.shape)
        for k in (1, 2):
            ref[k].append(_log_within(z, k, seed + b + 1))
    gaps = {k: float(np.mean(ref[k]) - obs[k]) for k in (1, 2)}
    sk = {k: float(np.std(ref[k]) * np.sqrt(1.0 + 1.0 / n_refs)) for k in (1, 2)}
    return gaps, sk, n_pc


# --- initial split-refinement: t-SNE per type -> k-means -> gap statistic ------
SPLIT_MIN_SPIKES = 50       # types with fewer trials than this are never tested for a split
SPLIT_MIN_CHILD = 20        # reject a split whose smaller child has fewer trials than this
SPLIT_MAX_ROUNDS = 3        # cap on repeated split passes (each re-tests every current type)
SPLIT_GAP_MARGIN = 1.0      # keep a split only if Gap(2) - Gap(1) > margin * s_2 (Tibshirani
                            #   1-SE rule).
SPLIT_GAP_SPACE = "pca"     # feature space the split's gap statistic is scored in:
                            #   "pca"  -- leading PCs of the type's raw +/-5 ms waveforms
                            #             (t-SNE + k-means only PROPOSE the 2-way partition);
                            #             behaves like the merge test, conservative.
                            #   "tsne" -- the 2-D t-SNE embedding itself; over-splits badly
                            #             because t-SNE fabricates tight local clumps.


def try_split_type_gap(waveforms_c, seed=msr.SPLIT_SEED, space=SPLIT_GAP_SPACE):
    """t-SNE-embed ONE type's +/-5 ms trial waveforms to 2D and k-means it
    (k=2) to PROPOSE a 2-way partition, then keep that split only if the
    gap statistic prefers k=2 over k=1 by the Tibshirani 1-SE rule
    (Gap(2) - Gap(1) > SPLIT_GAP_MARGIN * s_2) AND both children have at
    least SPLIT_MIN_CHILD trials. `space` sets where the gap is scored
    (see SPLIT_GAP_SPACE). Returns (did_split, sub_labels_or_None,
    (gap1, gap2, s2) or None)."""
    n = len(waveforms_c)
    if n < SPLIT_MIN_SPIKES:
        return False, None, None
    emb = shape_features_tsne(waveforms_c, 2, seed, perplexity=msr.SPLIT_TSNE_PERPLEXITY)
    sub = cluster_spikes(emb, 2, seed, method="kmeans")
    if len(set(sub.tolist())) < 2:
        return False, None, None
    if space == "tsne":
        gaps, sk, _ = gap_one_vs_two(emb, seed=seed)
    else:
        gaps, sk, _ = gap_one_vs_two(waveforms_c, seed=seed, fixed_two=sub)
    child_min = int(min((sub == 0).sum(), (sub == 1).sum()))
    did = (gaps[2] - gaps[1] > SPLIT_GAP_MARGIN * sk[2]) and child_min >= SPLIT_MIN_CHILD
    return did, (sub if did else None), (round(gaps[1], 3), round(gaps[2], 3), round(sk[2], 3))


def split_refine_gap(labels, waveforms, space=SPLIT_GAP_SPACE, max_rounds=SPLIT_MAX_ROUNDS):
    """Repeatedly apply try_split_type_gap to every current type until a
    whole pass splits nothing (or max_rounds is hit). A split moves the
    k-means child-1 spikes to a fresh type id. Returns
    (dense_labels, n_types, report_rows)."""
    lab = labels.copy()
    report = []
    for rnd in range(max_rounds):
        n_types = int(lab.max()) + 1
        out = lab.copy()
        next_id = n_types
        any_split = False
        for c in range(n_types):
            idx = np.where(lab == c)[0]
            did, sub, gaps = try_split_type_gap(waveforms[idx], space=space)
            report.append(dict(round=rnd, type=int(c), n=int(idx.size), split=bool(did),
                                gap1=(gaps[0] if gaps else None),
                                gap2=(gaps[1] if gaps else None),
                                s2=(gaps[2] if gaps else None),
                                child_id=(int(next_id) if did else None)))
            if did:
                out[idx[sub == 1]] = next_id
                next_id += 1
                any_split = True
        lab, _ = relabel_dense(out)
        if not any_split:
            break
    return lab, int(lab.max()) + 1, report


def split_refine(labels, waveforms, how, space=SPLIT_GAP_SPACE):
    """Dispatch: 'gap' -> iterative t-SNE + k-means + gap statistic
    (split_refine_gap); 'tsne' -> muap_split_refinement.apply_split
    (single pass, t-SNE + k-means + silhouette >= 0.5)."""
    if how == "tsne":
        raw, rep = msr.apply_split(labels, waveforms, int(labels.max()) + 1)
        dense, n, _ = msr.relabel_dense_tracked(raw)
        rows = [dict(round=0, type=r["type"], n=r["n"], split=bool(r["split"]),
                     gap1=None, gap2=None, s2=None, child_id=r.get("new_type_id")) for r in rep]
        return dense, n, rows
    return split_refine_gap(labels, waveforms, space=space)


def pool_is_one_cluster(pool_waveforms, test):
    """Decide whether a pooled pair of types is really ONE cluster.

    test="gap"  -- gap statistic for k=1 vs k=2 (see gap_one_vs_two); the
                   pool is one cluster when Gap(1) >= Gap(2) ("take the
                   larger"). Returns (is_one, {"gap1":.., "gap2":.., ..}).
    test="tsne" -- fresh t-SNE + k-means(k=2), one cluster when the split
                   silhouette < msr.SPLIT_SILHOUETTE_MIN. Returns
                   (is_one, {"silhouette": ..}).
    """
    if test == "gap":
        if len(pool_waveforms) < GAP_MIN_SPIKES:
            return True, dict(gap1=None, gap2=None, s1=None, s2=None, n_pc=None)
        gaps, sk, n_pc = gap_one_vs_two(pool_waveforms)
        return (gaps[1] >= gaps[2]), dict(gap1=round(gaps[1], 3), gap2=round(gaps[2], 3),
                                           s1=round(sk[1], 3), s2=round(sk[2], 3), n_pc=n_pc)
    did_split, _, sil = msr.try_split_type(pool_waveforms)
    return (not did_split), dict(silhouette=round(float(sil), 3) if sil == sil else None)


def sequential_anchor_merge(labels, waveforms, test):
    """The sequential anchor procedure described in the module docstring.
    Returns (merged_labels, merged_of, log, merge_groups):
      merged_of[c]   final merged-type id for pre-merge type c (0..M-1)
      merged_labels  merged_of applied per spike
      log            one dict per pair test, in test order
      merge_groups   list of tuples of pre-merge type ids that were fused
    """
    n_types = int(labels.max()) + 1
    group_of = list(range(n_types))          # union-by-anchor-id; groups have <=2 members
    remaining = list(range(n_types))         # types not yet absorbed into an earlier group
    log = []

    ai = 0
    while ai < len(remaining):
        anchor = remaining[ai]
        for other in list(remaining[ai + 1:]):
            mask = (labels == anchor) | (labels == other)
            is_one, info = pool_is_one_cluster(waveforms[mask], test)
            log.append(dict(anchor=int(anchor), other=int(other), n_pool=int(mask.sum()),
                             verdict="MERGE" if is_one else "keep", **info))
            if is_one:
                absorbing = group_of[anchor]
                gone = group_of[other]
                for t in range(n_types):
                    if group_of[t] == gone:
                        group_of[t] = absorbing
                remaining.remove(other)
                break                        # "continue until merge happens"
        ai += 1                              # advance to the next surviving anchor

    remap = {g: i for i, g in enumerate(sorted(set(group_of)))}
    merged_of = np.array([remap[group_of[c]] for c in range(n_types)])
    merged_labels = merged_of[labels]
    groups = {}
    for c in range(n_types):
        groups.setdefault(int(merged_of[c]), []).append(c)
    merge_groups = [tuple(v) for v in sorted(groups.values(), key=min) if len(v) > 1]
    return merged_labels, merged_of, log, merge_groups


def _cell_color(g):
    """Colour for a Voronoi face given its final group id; grey for -1
    (a merged type dropped by the final superposition re-check)."""
    return "0.75" if int(g) < 0 else CMAP(int(g) % 20)


def draw_merged_voronoi(ax, centroids, emb, type_labels, merged_of, radius, title):
    """One Voronoi panel: one face per pre-merge type (tessellation of the
    type centroids), each face + its spikes coloured by the type's FINAL
    group id, so fused cells share a colour and a face dropped as a
    superposition composite (-1) goes grey. Computed once, never redrawn."""
    n_types = centroids.shape[0]
    if n_types >= 4:
        vor = Voronoi(centroids)
        regions, vertices = voronoi_finite_polygons_2d(vor, radius)
        for c, region in enumerate(regions):
            ax.add_patch(Polygon(vertices[region], closed=True,
                                  facecolor=_cell_color(merged_of[c]),
                                  edgecolor="0.25", lw=1.0, alpha=0.45, zorder=1))
    for c in range(n_types):
        m = type_labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=3, color=_cell_color(merged_of[c]),
                   alpha=0.35, linewidths=0, zorder=2)
    ax.scatter(centroids[:, 0], centroids[:, 1], s=35, c="k", marker="x", lw=1.4, zorder=4)
    for c, (cx, cy) in enumerate(centroids):
        dst = "X" if int(merged_of[c]) < 0 else str(int(merged_of[c]))
        ax.annotate(f"{c}→{dst}", (cx, cy), fontsize=8, fontweight="bold",
                    ha="center", va="center", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="k", lw=0.6))
    ax.set_title(title, fontsize=10)


def _fmt_purity(name, n, st):
    return (f"  {name:<10}{n:>7}{st['mean']:>13.0%}{st['weighted_mean']:>12.0%}"
            f"{st['recovered']:>10}/{st['n_true']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10, help="curated trace with this many true MUAPs")
    ap.add_argument("--method", default="kmeans", choices=["kmeans", "gmm", "tmixture"])
    ap.add_argument("--test", default="gap", choices=["gap", "tsne"],
                     help="per-pair one-vs-two-cluster test: 'gap' = gap statistic "
                          "(Tibshirani 2001), merge when Gap(1) >= Gap(2); 'tsne' = fresh "
                          "t-SNE + k-means silhouette")
    ap.add_argument("--split", default="gap", choices=["gap", "tsne", "none"],
                     help="initial split-refinement BEFORE merging: 'gap' = iterative "
                          "t-SNE-per-type + k-means (propose) + gap statistic 1-SE rule "
                          "(accept); 'tsne' = muap_split_refinement single-pass silhouette; "
                          "'none' = skip")
    ap.add_argument("--split-space", default=SPLIT_GAP_SPACE, choices=["pca", "tsne"],
                     help="feature space the --split gap statistic is scored in (see "
                          "SPLIT_GAP_SPACE): 'pca' (default, conservative) or 'tsne' "
                          "(over-splits)")
    ap.add_argument("--post-superposition", default="drop", choices=["drop", "off"],
                     help="re-run spike_sort.superposition.find_superposition_candidates on the "
                          "FINAL merged type templates and drop any merged type that is the "
                          "time-shifted sum of two others ('drop', default) or skip ('off')")
    args = ap.parse_args()
    tag = f"{args.method}_{args.test}"
    if args.split != "none":
        tag += f"_s{args.split}" + (f"-{args.split_space}" if args.split == "gap" else "")

    _, lookup = base.load_curated_library()
    _, fs, true_units, baseline_s = sst.load_simulated_trace(
        os.path.join(base.TRACE_DIR, f"simulated_trace_n{args.n}.npz"), lookup)
    emg_txt = os.path.join(sst.SPIKESORT_INPUT_DIR, f"EMGRUN__curated_n{args.n}.txt")
    trace = load_signal(emg_txt, fs, 0.0, None)
    base.call_spike_sort_emg.SILHOUETTE_MAX_N = args.n + 10

    overrides = dict(cluster_method=args.method, amp_dur_weight=0.0, polarity_weight=0.0,
                      threshold_mad=sst.DETECTION_THRESHOLD_MAD)
    peaks_t, peak_amps, waveforms, emb, raw_labels = cluster_from_scratch(
        trace, fs, baseline_s, args.method, overrides)
    n_raw = int(raw_labels.max()) + 1
    print(f"n={args.n}: raw {args.method} -> {n_raw} types, {raw_labels.size} spikes")

    keep, pp_labels = post_process(peaks_t, peak_amps, waveforms, raw_labels, fs)
    n_pp = int(pp_labels.max()) + 1
    k_t, k_wf, k_emb = peaks_t[keep], waveforms[keep], emb[keep]
    print(f"post-processing -> {n_pp} types, {keep.sum()}/{keep.size} spikes kept")

    # --- initial split-refinement (t-SNE per type -> k-means -> gap) ---
    if args.split != "none":
        sr_labels, n_sr, split_report = split_refine(pp_labels, k_wf, args.split,
                                                      space=args.split_space)
        n_split_events = sum(1 for r in split_report if r["split"])
        sp_note = f", {args.split_space}-space" if args.split == "gap" else ""
        print(f"\ninitial split-refinement ({args.split}{sp_note}) -- {len(split_report)} "
              f"type test(s), {n_split_events} split(s):")
        for r in split_report:
            g = (f"Gap(2)-Gap(1)={r['gap2'] - r['gap1']:+.3f}  (s2={r['s2']}, "
                 f"1-SE rule: split if > {SPLIT_GAP_MARGIN:g}*s2)"
                 if r["gap1"] is not None else "silhouette-rule")
            mark = f"  <== SPLIT -> new type {r['child_id']}" if r["split"] else ""
            print(f"  round {r['round']}  type {r['type']:>2} (n={r['n']:>4})   {g}{mark}")
        print(f"{n_pp} -> {n_sr} types after split-refinement")
    else:
        sr_labels, n_sr, split_report = pp_labels.copy(), n_pp, []

    merged_labels, merged_of, log, merge_groups = sequential_anchor_merge(
        sr_labels, k_wf, args.test)
    n_merged = int(merged_labels.max()) + 1

    print(f"\nsequential anchor merge ({args.test} test) -- {len(log)} pair test(s):")
    cur_anchor = None
    for e in log:
        if e["anchor"] != cur_anchor:
            cur_anchor = e["anchor"]
            print(f"  anchor {cur_anchor}:")
        if args.test == "gap":
            metric = (f"Gap(1)={e['gap1']} Gap(2)={e['gap2']}  "
                      f"[larger: k={1 if (e['gap1'] or -9) >= (e['gap2'] or -9) else 2}, "
                      f"n_pc={e['n_pc']}]")
        else:
            metric = f"silhouette={e['silhouette']}"
        mark = "  <== MERGE (one cluster)" if e["verdict"] == "MERGE" else ""
        print(f"    vs type {e['other']:>2}   {metric}   pool n={e['n_pool']:>4}   "
              f"{e['verdict']}{mark}")
    print(f"\nmerge groups (pre-merge type ids): {merge_groups if merge_groups else 'none'}")

    # --- final superposition re-check on the MERGED type templates ---
    #  merged_labels : per-spike merged id (len k_wf); merged_of : split-type -> merged id.
    #  final_of maps split-type -> FINAL id, with -1 for a merged type dropped here.
    sup_cands, sup_dropped = [], []
    final_labels = merged_labels
    final_of = merged_of.copy()
    n_final = n_merged
    if args.post_superposition == "drop" and n_merged >= 3:
        m_templates = np.stack([k_wf[merged_labels == c].mean(axis=0) for c in range(n_merged)])
        max_shift = int(round(base.SUPERPOSITION_MAX_SHIFT_MS * 1e-3 * fs))
        sup_cands = find_superposition_candidates(m_templates, max_shift)
        sup_dropped = sorted({c["type_c"] for c in sup_cands})
        print(f"\nfinal superposition re-check on {n_merged} merged types "
              f"(max shift {base.SUPERPOSITION_MAX_SHIFT_MS} ms):")
        print("  " + format_superposition_report(sup_cands, fs=fs).replace("\n", "\n  "))
        if sup_dropped:
            surv = [c for c in range(n_merged) if c not in sup_dropped]
            merged_to_final = -np.ones(n_merged, dtype=int)
            for new_id, old in enumerate(surv):
                merged_to_final[old] = new_id
            final_labels = merged_to_final[merged_labels]        # -1 for dropped, len k_wf
            final_of = merged_to_final[merged_of]                # split-type -> final id / -1
            n_final = len(surv)
            print(f"  dropped merged type(s) {sup_dropped}  ->  {n_merged} -> {n_final} types")
        else:
            print(f"  no merged type is a superposition composite  ->  {n_final} types kept")

    print(f"types remaining: {n_pp} (post-proc) -> {n_sr} (split) -> {n_merged} (merged) "
          f"-> {n_final} (final, superposition-checked)")

    # --- Voronoi: tessellation built ONCE from the split-refined type centroids;
    #     cells sharing a colour were fused by the merge step; grey 'X' cells were
    #     dropped by the final superposition re-check ---
    centroids = np.stack([k_emb[sr_labels == c].mean(axis=0) for c in range(n_sr)])
    radius = float(np.ptp(k_emb, axis=0).max()) * 3.0
    lo, hi = k_emb.min(axis=0), k_emb.max(axis=0)
    pad = 0.08 * (hi - lo)

    fig, ax = plt.subplots(figsize=(13, 9))
    draw_merged_voronoi(
        ax, centroids, k_emb, sr_labels, final_of, radius,
        f"n={args.n} MUAPs | tsne + {args.method} + post-processing"
        f"{'' if args.split == 'none' else f' + {args.split}-split'}"
        f"{'' if args.post_superposition == 'off' else ' + final superposition drop'}\n"
        f"sequential anchor merge, {args.test} one-vs-two test:  "
        f"{n_pp} -> {n_sr} -> {n_merged} -> {n_final} types "
        f"(same colour = merged; grey X = dropped superposition; label = type→final id)")
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_xlabel("t-SNE dim 1 (standardized)")
    ax.set_ylabel("t-SNE dim 2 (standardized)")
    fig.tight_layout()
    out_dir = os.path.join(base.SIM_ROOT, "tsne")
    os.makedirs(out_dir, exist_ok=True)
    vpath = os.path.join(out_dir, f"seqmerge_n{args.n}_{tag}_voronoi.png")
    fig.savefig(vpath, dpi=150)
    plt.close(fig)
    print(f"Wrote {vpath}")

    # --- purity vs ground truth ---
    est_match, est_purity, est_stats = purity_vs_truth(pp_labels, k_t, true_units)
    sr_match, sr_purity, sr_stats = purity_vs_truth(sr_labels, k_t, true_units)
    mrg_match, mrg_purity, mrg_stats = purity_vs_truth(merged_labels, k_t, true_units)
    dropped_final = final_labels < 0
    fin_match, fin_purity, fin_stats = purity_vs_truth(
        final_labels[~dropped_final], k_t[~dropped_final], true_units)
    print(f"\nPurity vs {len(true_units)} ground-truth MUAPs (match tol {base.MATCH_TOL_MS} ms):")
    print(f"  {'':10}{'types':>7}{'mean purity':>14}{'spike-wtd':>12}{'true units rec.':>18}")
    print(_fmt_purity("estimated", n_pp, est_stats))
    if args.split != "none":
        print(_fmt_purity("split", n_sr, sr_stats))
    print(_fmt_purity("merged", n_merged, mrg_stats))
    if sup_dropped:
        print(_fmt_purity("final", n_final, fin_stats))

    # --- galleries: actual, estimated (post-proc), split-refined, final merged ---
    measured = base.bandpass(trace, fs, 20.0, 10000.0)
    gallery_true(true_units, measured, fs,
                 f"Actual (ground-truth) MUAPs, n={args.n}  ({len(true_units)} units)",
                 os.path.join(out_dir, f"seqmerge_n{args.n}_{tag}_gallery_true.png"))
    gallery(pp_labels, k_t, measured, fs,
            f"Estimated MUAPs -- tsne + {args.method} + post-processing, n={args.n}  "
            f"({n_pp} types, mean purity {est_stats['mean']:.0%}, "
            f"{est_stats['recovered']}/{est_stats['n_true']} true units)",
            os.path.join(out_dir, f"seqmerge_n{args.n}_{tag}_gallery_est.png"),
            est_match=est_match, est_purity=est_purity)
    if args.split != "none":
        gallery(sr_labels, k_t, measured, fs,
                f"After initial split-refinement ({args.split}), n={args.n}  "
                f"({n_sr} types, mean purity {sr_stats['mean']:.0%}, "
                f"{sr_stats['recovered']}/{sr_stats['n_true']} true units)",
                os.path.join(out_dir, f"seqmerge_n{args.n}_{tag}_gallery_split.png"),
                est_match=sr_match, est_purity=sr_purity)
    gallery(merged_labels, k_t, measured, fs,
            f"Merged MUAPs -- sequential merge ({args.test} test), n={args.n}  "
            f"({n_merged} types, mean purity {mrg_stats['mean']:.0%}, "
            f"{mrg_stats['recovered']}/{mrg_stats['n_true']} true units)",
            os.path.join(out_dir, f"seqmerge_n{args.n}_{tag}_gallery_merged.png"),
            est_match=mrg_match, est_purity=mrg_purity)
    if sup_dropped:
        fin_dense, _ = relabel_dense(final_labels[~dropped_final])
        gallery(fin_dense, k_t[~dropped_final], measured, fs,
                f"Final MUAPs -- after superposition re-check, n={args.n}  "
                f"({n_final} types, {len(sup_dropped)} dropped; mean purity "
                f"{fin_stats['mean']:.0%}, {fin_stats['recovered']}/{fin_stats['n_true']} "
                f"true units)",
                os.path.join(out_dir, f"seqmerge_n{args.n}_{tag}_gallery_final.png"),
                est_match=fin_match, est_purity=fin_purity)

    # --- data dump ---
    dpath = os.path.join(out_dir, f"seqmerge_n{args.n}_{tag}_data.npz")
    gap1 = np.array([np.nan if e.get("gap1") is None else e["gap1"] for e in log])
    gap2 = np.array([np.nan if e.get("gap2") is None else e["gap2"] for e in log])
    sil = np.array([np.nan if e.get("silhouette") is None else e["silhouette"] for e in log])
    np.savez_compressed(
        dpath,
        peaks_t=k_t, pp_labels=pp_labels, sr_labels=sr_labels,
        merged_labels=merged_labels, merged_of=merged_of,
        pp_templates=np.stack([k_wf[pp_labels == c].mean(axis=0) for c in range(n_pp)]),
        sr_templates=np.stack([k_wf[sr_labels == c].mean(axis=0) for c in range(n_sr)]),
        merged_templates=np.stack([k_wf[merged_labels == c].mean(axis=0)
                                    for c in range(n_merged)]),
        split_log_round=np.array([r["round"] for r in split_report]),
        split_log_type=np.array([r["type"] for r in split_report]),
        split_log_split=np.array([r["split"] for r in split_report]),
        split_log_gap1=np.array([np.nan if r["gap1"] is None else r["gap1"]
                                  for r in split_report]),
        split_log_gap2=np.array([np.nan if r["gap2"] is None else r["gap2"]
                                  for r in split_report]),
        split_log_s2=np.array([np.nan if r.get("s2") is None else r["s2"]
                                for r in split_report]),
        merge_log_anchor=np.array([e["anchor"] for e in log]),
        merge_log_other=np.array([e["other"] for e in log]),
        merge_log_verdict=np.array([e["verdict"] for e in log]),
        merge_log_gap1=gap1, merge_log_gap2=gap2, merge_log_silhouette=sil,
        final_labels=final_labels, final_of=final_of,
        superposition_dropped=np.array(sup_dropped, dtype=int),
        superposition_pairs=np.array([(c["type_c"], c["type_i"], c["type_j"])
                                      for c in sup_cands], dtype=int).reshape(-1, 3),
        superposition_pair_residual=np.array([c["pair_residual_frac"] for c in sup_cands]),
        test=args.test, split=args.split, post_superposition=args.post_superposition,
        n_raw=n_raw, n_pp=n_pp, n_sr=n_sr, n_merged=n_merged, n_final=n_final, fs=fs,
    )
    print(f"Wrote {dpath}")


if __name__ == "__main__":
    main()
