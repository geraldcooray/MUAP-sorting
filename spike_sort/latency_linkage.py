"""Identify sorted spike "types" that are probably the same MUAP's
different phases, not different motor units.

Detection collapses each MUAP's lobes into one broad envelope bump (see
detect_spikes), but that collapse isn't perfect -- a sharp secondary
lobe/turn can still cross threshold on its own and get extracted as its
own "spike", with its own shape. Downstream clustering then has no way to
know that two of its types are really one waveform's two parts: it just
sees two consistent shapes and calls them two types.

The giveaway is timing: if type A is the main phase and type B is a
secondary lobe of the *same* underlying discharge, then (almost) every
occurrence of A is accompanied by an occurrence of B a few ms later (or
earlier), and vice versa -- because they fire together by construction,
not by coincidence. Two genuinely different motor units, in contrast,
have no reason to keep a fixed short latency to each other (their firing
is independent), so this signature is unlikely to arise from two real
separate units unless they happen to be firing in a fixed pattern.

Core functions:
    pairwise_latency_stats  -- for every ordered pair of types, how often
                                (and at what latency) a type-A spike has a
                                type-B spike nearby.
    find_linked_type_pairs  -- flags pairs with high, ~symmetric,
                                low-jitter co-occurrence as same-MUAP
                                candidates.
    group_linked_types      -- unions linked pairs into groups (so if A-B
                                and B-C are both linked, {A, B, C} come
                                out as one group).
    merge_linked_types      -- collapses each group to one canonical
                                label and drops the resulting near-
                                duplicate detections (reuses
                                isi_cleanup.enforce_min_isi_per_type).
"""
import numpy as np

from spike_sort.isi_cleanup import enforce_min_isi_per_type


def pairwise_latency_stats(peaks_t, labels, max_latency_ms):
    """For every ordered pair of distinct types (i, j), match each type-i
    spike to its nearest type-j spike and keep the match if it's within
    max_latency_ms. Returns {(i, j): stats}, where stats has:

        n_i               -- number of type-i spikes
        n_matched         -- how many of them had a type-j spike within
                              max_latency_ms
        co_occurrence_frac -- n_matched / n_i (1.0 = every type-i spike
                              has a type-j companion)
        latency_mean_ms, latency_std_ms, latency_median_ms
                          -- signed latency (type-j time minus type-i
                             time) over the matched pairs only; NaN if
                             n_matched == 0

    Matching is nearest-neighbor, not one-to-one (same convention as
    spikesort_simulated_traces.match_to_ground_truth) -- fine here since
    what matters is whether *a* companion exists nearby, not a strict
    pairing.
    """
    max_latency_s = max_latency_ms * 1e-3
    types = np.unique(labels)
    times_by_type = {int(c): np.sort(peaks_t[labels == c]) for c in types}

    stats = {}
    for i in times_by_type:
        ti = times_by_type[i]
        if ti.size == 0:
            continue
        for j in times_by_type:
            if j == i:
                continue
            tj = times_by_type[j]
            if tj.size == 0:
                continue

            idx = np.searchsorted(tj, ti)
            idx_lo = np.clip(idx - 1, 0, tj.size - 1)
            idx_hi = np.clip(idx, 0, tj.size - 1)
            cand_lo, cand_hi = tj[idx_lo], tj[idx_hi]
            d_lo, d_hi = np.abs(ti - cand_lo), np.abs(ti - cand_hi)
            nearest = np.where(d_hi < d_lo, cand_hi, cand_lo)

            latency = nearest - ti  # signed: positive = j fires after i
            matched = np.abs(latency) <= max_latency_s
            n_matched = int(matched.sum())

            stats[(i, j)] = dict(
                n_i=int(ti.size), n_matched=n_matched,
                co_occurrence_frac=n_matched / ti.size,
                latency_mean_ms=float(np.mean(latency[matched]) * 1e3) if n_matched else np.nan,
                latency_std_ms=float(np.std(latency[matched]) * 1e3) if n_matched else np.nan,
                latency_median_ms=float(np.median(latency[matched]) * 1e3) if n_matched else np.nan,
            )
    return stats


def find_linked_type_pairs(peaks_t, labels, max_latency_ms=8.0, min_co_occurrence=0.6,
                            max_latency_std_ms=1.0, require_symmetric=False):
    """Flag unordered type pairs {i, j} as same-MUAP candidates.

    The real signature is usually ASYMMETRIC, not both-ways: a secondary
    lobe's detections almost always have a main-phase companion nearby
    (its whole reason for existing is riding along on that discharge),
    but the main phase doesn't need a secondary-lobe companion every
    time -- the lobe might not clear threshold on a smaller-amplitude
    occurrence, or the "main" type might itself be a mix of several
    causes. So by default (require_symmetric=False) a pair qualifies if
    EITHER direction's co-occurrence fraction clears min_co_occurrence.
    Pass require_symmetric=True for the stricter both-ways test instead.

    Either way, what actually discriminates a real linked pair from two
    independently-firing units that coincidentally land near each other
    sometimes is max_latency_std_ms: two real phases of one waveform
    fire at a near-fixed offset (std of a few tenths of a ms), while
    coincidental proximity between independent units' spikes has latency
    spread roughly uniformly across the whole +/-max_latency_ms window
    (std on the order of max_latency_ms itself) -- so keep
    max_latency_std_ms tight (order ~1 ms or less) even if
    min_co_occurrence is relaxed.

    max_latency_ms should be around one MUAP's own duration (a few ms).

    Returns a list of dicts, one per flagged pair, most-supported
    direction's fraction first:
        type_a, type_b, n_a, n_b, co_occurrence_a_to_b, co_occurrence_b_to_a,
        latency_mean_ms (b relative to a), latency_std_ms,
        dependent_type -- whichever of a/b has the higher co-occurrence
                          fraction (the one that "almost always" has a
                          companion -- the more likely secondary lobe)
    sorted by that dependent-side co-occurrence (most confident first).
    """
    stats = pairwise_latency_stats(peaks_t, labels, max_latency_ms)
    types = sorted({c for pair in stats for c in pair})

    linked = []
    for pos, i in enumerate(types):
        for j in types[pos + 1:]:
            fwd, rev = stats.get((i, j)), stats.get((j, i))
            if fwd is None or rev is None:
                continue
            co_fwd, co_rev = fwd["co_occurrence_frac"], rev["co_occurrence_frac"]
            qualifies = (min(co_fwd, co_rev) >= min_co_occurrence if require_symmetric
                         else max(co_fwd, co_rev) >= min_co_occurrence)
            if qualifies and fwd["latency_std_ms"] <= max_latency_std_ms:
                linked.append(dict(
                    type_a=i, type_b=j, n_a=fwd["n_i"], n_b=rev["n_i"],
                    co_occurrence_a_to_b=co_fwd, co_occurrence_b_to_a=co_rev,
                    latency_mean_ms=fwd["latency_mean_ms"],
                    latency_std_ms=fwd["latency_std_ms"],
                    dependent_type=i if co_fwd >= co_rev else j,
                ))
    linked.sort(key=lambda p: max(p["co_occurrence_a_to_b"], p["co_occurrence_b_to_a"]),
                reverse=True)
    return linked


def group_linked_types(linked_pairs):
    """Union pairwise links into groups via union-find, so a chain (A-B
    linked, B-C linked) comes out as one group {A, B, C} even though A-C
    was never itself tested/flagged directly. Returns a list of sorted
    tuples, singletons excluded (only actual groups of >=2 types)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for p in linked_pairs:
        union(p["type_a"], p["type_b"])

    groups = {}
    for node in parent:
        groups.setdefault(find(node), set()).add(node)
    return [tuple(sorted(g)) for g in groups.values() if len(g) > 1]


def merge_linked_types(peaks_t, labels, peak_amps, groups, max_latency_ms=8.0):
    """Collapse each linked group of types to one canonical label (its
    smallest original label), then drop the resulting near-duplicate
    detections -- the same physical discharge, previously split across
    two types, would otherwise show up twice in the merged type a few ms
    apart. Reuses isi_cleanup.enforce_min_isi_per_type (keeping the
    larger-|amplitude| detection of each too-close pair) with
    max_latency_ms as the minimum-ISI window, since that's exactly the
    same-discharge time scale established by find_linked_type_pairs.

    Returns (new_labels, keep) -- new_labels is the same length as labels
    (merged, but NOT yet relabeled dense; pass through
    spike_sort.clustering.relabel_dense if a contiguous 0..k-1 range is
    needed), keep is a boolean mask selecting which original spikes
    survive the dedup (apply to peaks_t/peak_amps/waveforms/features
    together, same convention as enforce_min_isi_per_type)."""
    remap = {t: min(group) for group in groups for t in group}
    new_labels = np.array([remap.get(label, label) for label in labels])
    keep = enforce_min_isi_per_type(peaks_t, new_labels, peak_amps, max_latency_ms * 1e-3)
    return new_labels, keep


def format_linked_pairs_report(linked_pairs, type_names=None):
    """Human-readable one-line-per-pair summary of find_linked_type_pairs'
    output, most confident link first. type_names optionally maps a type
    label to a display name (e.g. "est 3"); defaults to str(label)."""
    if not linked_pairs:
        return "No linked (same-MUAP-candidate) type pairs found."
    name = (lambda c: type_names.get(c, str(c))) if type_names else str
    lines = []
    for p in linked_pairs:
        lines.append(
            f"type {name(p['type_a'])} (n={p['n_a']}) <-> type {name(p['type_b'])} "
            f"(n={p['n_b']}): co-occurrence {p['co_occurrence_a_to_b']:.0%}/"
            f"{p['co_occurrence_b_to_a']:.0%}, latency {p['latency_mean_ms']:+.2f} "
            f"+/- {p['latency_std_ms']:.2f} ms (type {name(p['dependent_type'])} is the "
            f"likely secondary lobe)"
        )
    return "\n".join(lines)
