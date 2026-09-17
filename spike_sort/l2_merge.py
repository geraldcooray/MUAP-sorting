"""Post-clustering shape-similarity merge.

t-SNE's embedding preserves *local* neighborhood structure, not global
distances (see shape_features_tsne) -- so it can still split one true
MUAP shape into two nearby "types" if the local point density happens to
vary across it, even when the underlying waveforms are practically
identical. spike_sort.latency_linkage catches one specific cause of that
(a secondary lobe extracted as its own detection); this module catches
it more directly and generally, by looking straight at each type's own
averaged (mean) waveform -- the actual physical MUAP shape -- and merging
any two types whose averaged waveforms are close in raw L2 (Euclidean)
distance.

Deliberately NOT normalized/standardized before the distance is computed
(unlike the shape features built in spike_sort.features, which
standardize every feature to unit variance so no one dimension dominates
by scale alone) -- here the raw physical amplitude scale is exactly what
should discriminate two types: an averaged MUAP with peak amplitude 900
should never register as "close" to one with peak amplitude 90 just
because both happen to have a similar normalized shape. Two genuinely
different MUAPs practically always differ enough in raw amplitude/shape
to sit far apart in L2, even if t-SNE's local structure placed their
spikes near each other.

l2_threshold is deliberately left to the caller (movable, no shipped
default) -- what counts as "close" depends entirely on the recording's
own raw amplitude units/scale, which nothing here can guess.
"""
import numpy as np


def type_mean_waveforms(waveforms, labels):
    """{type: mean waveform} for every type, raw units, unnormalized."""
    return {int(c): waveforms[labels == c].mean(axis=0) for c in np.unique(labels)}


def normalize_area(waveform):
    """Scale a waveform so the area under |waveform| (its L1 norm) is 1 --
    i.e. divide out its own total amplitude "mass" before comparing
    shape. A biphasic/multiphasic MUAP's raw sum isn't a useful area
    (positive and negative lobes partly cancel), so this normalizes
    against sum(|waveform|) instead. Leaves an all-zero waveform
    untouched (nothing to normalize by)."""
    area = np.sum(np.abs(waveform))
    return waveform / area if area > 0 else waveform


def pairwise_l2_distances(waveforms, labels, normalize=False):
    """{(i, j): L2 distance} between every unordered pair of types' mean
    waveforms, i < j.

    normalize=False (default): raw units, no standardization/
    normalization -- an averaged MUAP with peak amplitude 900 never
    registers as "close" to one with peak amplitude 90 just because both
    have a similar normalized shape (see module docstring).

    normalize=True: each mean waveform is first scaled to unit area (see
    normalize_area) before the distance is computed -- an explicit,
    opt-in alternative that compares normalized SHAPE only, blind to
    overall amplitude. Useful to check whether two types differ mainly
    in amplitude (raw L2 far apart, area-normalized L2 close -- likely
    the same MUAP shape at different gain/interference level) or in
    genuine waveform shape (still far apart even area-normalized)."""
    means = type_mean_waveforms(waveforms, labels)
    if normalize:
        means = {c: normalize_area(m) for c, m in means.items()}
    types = sorted(means)
    dist = {}
    for pos, i in enumerate(types):
        for j in types[pos + 1:]:
            dist[(i, j)] = float(np.linalg.norm(means[i] - means[j]))
    return dist


def sorted_distance_array(l2_distances):
    """Ascending 1-D array of every pairwise L2 distance value in the
    {(i, j): distance} dict from pairwise_l2_distances -- e.g. for
    plotting the sorted-distance curve used to eyeball a percentile-based
    cutoff (see percentile_threshold)."""
    return np.sort(np.fromiter(l2_distances.values(), dtype=float))


def percentile_threshold(l2_distances, percentile):
    """The distance value at the given percentile (0-100) of every
    pairwise L2 distance -- e.g. percentile=20 gives the value below
    which the closest 20% of type pairs fall, for a "merge the smallest
    20% of distances" rule instead of a fixed absolute threshold. Returns
    NaN if there are fewer than 2 types (no pairs to compute)."""
    values = sorted_distance_array(l2_distances)
    return float(np.percentile(values, percentile)) if values.size else float("nan")


def elbow_threshold(l2_distances):
    """The distance value at the largest gap between consecutive sorted
    pairwise distances -- an automatic "elbow" cutoff attempt: below the
    gap are presumably same-MUAP duplicate types, above it genuinely
    different types.

    VALIDATED AGAINST GROUND TRUTH ON THE SIMULATED TRACES AND FOUND TO
    FAIL BADLY: real type-pair distance distributions here are roughly
    continuous/gently-accelerating, not bimodal, so the single largest
    gap usually sits between the two *largest* distances (naturally more
    spread apart in absolute terms) rather than at any meaningful
    same-unit/different-unit boundary -- this collapsed every type into
    one on 3 of 4 test traces. Kept for reference/comparison; prefer
    outlier_fence_threshold, which matches this problem's actual shape
    (a few genuine duplicates as low outliers, not two equal clusters)
    and validated much better.

    Returns the midpoint of the largest gap (NaN if fewer than 2 pairs
    exist, i.e. fewer than 3 types)."""
    values = sorted_distance_array(l2_distances)
    if values.size < 2:
        return float("nan")
    gaps = np.diff(values)
    i = int(np.argmax(gaps))
    return float((values[i] + values[i + 1]) / 2.0)


def outlier_fence_threshold(l2_distances, k=2.0):
    """A robust *low-outlier* cutoff: median(distances) - k * MAD(distances)
    (MAD = median absolute deviation, unscaled). Matches this problem's
    actual expected shape better than elbow_threshold's gap search: a
    small few genuine duplicate-type pairs should sit as low outliers
    below a broad spread of genuinely-different-type distances, not as
    one side of a clean two-cluster gap.

    Lower k = fence closer to the median = merges more (more aggressive);
    higher k = fence further below the median = merges less. k=2.0
    validated well on the simulated traces: it made the one clearly-safe
    merge where a real low outlier existed (n=3 trace) and correctly
    merged nothing where it didn't (n=5/10/20 traces, where a smaller k
    like 1.5 already started causing significant cross-unit contamination
    -- see spikesort_simulated_traces.py's L2-merge validation). Still no
    guarantee against all false merges -- check the sorted-distance plot.

    Returns NaN if fewer than 2 pairs exist."""
    values = sorted_distance_array(l2_distances)
    if values.size < 2:
        return float("nan")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(median - k * mad, 0.0)


def find_close_type_pairs(waveforms, labels, l2_threshold, normalize=False):
    """Flag unordered type pairs {i, j} whose mean waveforms' L2 distance
    is <= l2_threshold (raw units, or area-normalized -- see
    pairwise_l2_distances' normalize). Returns a list of dicts: type_a,
    type_b, l2_distance, n_a, n_b, sorted closest-first."""
    dist = pairwise_l2_distances(waveforms, labels, normalize=normalize)
    counts = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
    close = [dict(type_a=i, type_b=j, l2_distance=d, n_a=counts[i], n_b=counts[j])
             for (i, j), d in dist.items() if d <= l2_threshold]
    close.sort(key=lambda p: p["l2_distance"])
    return close


def group_close_types(close_pairs):
    """Union pairwise close-matches into groups via union-find (same
    convention as spike_sort.latency_linkage.group_linked_types), so a
    chain (A close to B, B close to C) comes out as one group {A, B, C}.
    Returns a list of sorted tuples, singletons excluded."""
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

    for p in close_pairs:
        union(p["type_a"], p["type_b"])

    groups = {}
    for node in parent:
        groups.setdefault(find(node), set()).add(node)
    return [tuple(sorted(g)) for g in groups.values() if len(g) > 1]


def merge_close_types(labels, groups):
    """Collapse each group of shape-close types to one canonical label
    (its smallest original label). Unlike
    latency_linkage.merge_linked_types, this never drops any spikes -- an
    L2 shape match says two types ARE the same MUAP, not that any
    individual detection is a duplicate of another, so every spike keeps
    its place, just relabeled. Returns new_labels, same length as labels,
    NOT yet relabeled dense -- pass through
    spike_sort.clustering.relabel_dense for a contiguous 0..k-1 range."""
    remap = {t: min(group) for group in groups for t in group}
    return np.array([remap.get(label, label) for label in labels])


def format_close_pairs_report(close_pairs, type_names=None):
    """Human-readable one-line-per-pair summary, closest first."""
    if not close_pairs:
        return "No shape-close (L2) type pairs found."
    name = (lambda c: type_names.get(c, str(c))) if type_names else str
    lines = []
    for p in close_pairs:
        lines.append(
            f"type {name(p['type_a'])} (n={p['n_a']}) <-> type {name(p['type_b'])} "
            f"(n={p['n_b']}): L2 distance {p['l2_distance']:.2f}"
        )
    return "\n".join(lines)
