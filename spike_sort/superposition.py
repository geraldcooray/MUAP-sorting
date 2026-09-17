"""Detect estimated MUAP types that are actually superpositions (sums)
of two OTHER estimated types, rather than genuinely distinct units.

This is a third, distinct failure mode alongside spike_sort.latency_linkage
(same discharge detected twice -- a secondary lobe split off as its own
type) and spike_sort.l2_merge (same shape over-fragmented into two nearby
types): here, type C is a genuinely different-LOOKING waveform, but only
because it's really type A and type B's waveforms summed together (their
spikes consistently landing close enough in time to superimpose), not
because a third motor unit exists.

Two complementary pieces:

    effective_rank              -- "how many genuinely independent MUAP
                                    shapes are actually present", via the
                                    singular-value spectrum of the
                                    stacked mean-waveform matrix against
                                    a noise-floor threshold. Noiselessly,
                                    a composite type is an exact linear
                                    combination of others, so
                                    rank(templates) < n_types whenever
                                    superposition artifacts exist; real
                                    averaging noise makes the matrix
                                    numerically full-rank regardless, so
                                    this asks the same question in a
                                    noise-robust way (numerical/effective
                                    rank via a random-matrix-theory
                                    noise ceiling) instead of exact rank.

    find_superposition_candidates -- "which specific type is a
                                    superposition of which two others, at
                                    what relative delay" -- a small
                                    two-stage matching-pursuit search
                                    (best single-template fit, then
                                    matched-filter the residual against
                                    every other template/shift) run on
                                    the already-estimated per-type mean
                                    waveforms themselves, not the raw
                                    trace.
"""
import numpy as np


def effective_rank(templates, sigma, n_trials):
    """How many singular values of the (n_types x n_samples) stacked
    mean-waveform matrix `templates` sit above the expected noise floor,
    given `sigma` (raw per-sample noise standard deviation, e.g. from the
    trace's own spike-free baseline segment) and `n_trials` (how many
    spikes were averaged into each row -- a mean of n trials has residual
    noise variance sigma^2 / n, not sigma^2 itself).

    Each row is first scaled by sqrt(n_trials) so every row's residual
    averaging-noise variance becomes the same (sigma^2), regardless of
    how many trials went into it -- otherwise a type with few trials
    (noisier mean) and one with many (cleaner mean) aren't comparable on
    the same noise-floor threshold. Against that homogenized noise level,
    a pure-noise (n_types x n_samples) matrix has singular values
    clustering near sigma * (sqrt(n_types) + sqrt(n_samples)) -- the
    Marchenko-Pastur bulk-edge bound for an i.i.d. Gaussian random
    matrix of that shape and per-entry variance sigma^2 -- so singular
    values above that ceiling reflect real signal structure, not noise.

    Returns (n_significant, singular_values, noise_ceiling).
    n_significant < n_types is direct evidence that some estimated types
    are linear combinations of others (duplicates or superpositions) --
    exactly the noiseless rank-deficiency argument, made noise-robust.
    Doesn't say WHICH types are redundant -- see
    find_superposition_candidates for that."""
    templates = np.asarray(templates, dtype=float)
    n_types, n_samples = templates.shape
    n_trials = np.asarray(n_trials, dtype=float)
    scaled = templates * np.sqrt(np.maximum(n_trials, 1.0))[:, None]
    singular_values = np.linalg.svd(scaled, compute_uv=False)
    noise_ceiling = sigma * (np.sqrt(n_types) + np.sqrt(n_samples))
    n_significant = int(np.sum(singular_values > noise_ceiling))
    return n_significant, singular_values, noise_ceiling


def _shift_waveform(wf, shift):
    """Shift wf by `shift` samples (positive = later in time), zero-
    padding the exposed edge (not circular -- a circularly-wrapped
    shift would smear the far end of the waveform into the near end,
    fabricating structure that was never there)."""
    n = len(wf)
    out = np.zeros_like(wf)
    if shift == 0:
        return wf.copy()
    if shift > 0:
        if shift < n:
            out[shift:] = wf[:n - shift]
    else:
        if -shift < n:
            out[:n + shift] = wf[-shift:]
    return out


def _best_single_fit(target, templates, exclude):
    """Best-fitting single OTHER template for `target`: a non-negative
    least-squares scalar (unshifted) per candidate, keep the lowest
    relative residual. Non-negative because a "fit" via a negative
    coefficient isn't a physically meaningful match -- it's just
    cancellation, not the same MUAP recurring. Returns (idx, coef,
    residual_vector, residual_frac), or None if no candidates."""
    best = None
    for i, t in enumerate(templates):
        if i == exclude:
            continue
        denom = float(np.dot(t, t))
        coef = max(float(np.dot(target, t) / denom), 0.0) if denom > 0 else 0.0
        residual = target - coef * t
        rf = float(np.linalg.norm(residual) / (np.linalg.norm(target) + 1e-12))
        if best is None or rf < best[3]:
            best = (i, coef, residual, rf)
    return best


def find_superposition_candidates(templates, max_shift_samples, min_pair_improvement=0.15,
                                   max_pair_residual_frac=0.25):
    """Two-stage matching-pursuit search, run on the estimated per-type
    mean waveforms themselves (not the raw trace): for each candidate
    composite type c,
        1. find its best-fitting single OTHER type i (non-negative
           scalar fit) -- this is the "obvious" explanation if c were
           really just a noisy copy of one real unit;
        2. matched-filter the LEFTOVER residual against every other
           type j at every shift within +/-max_shift_samples, to find
           whichever (j, shift) most reduces what's left unexplained;
        3. refine both amplitudes jointly (non-negative least squares
           over [template_i, shift(template_j, shift)]).
    Flags c as a likely superposition of (i, j) only if BOTH:
      - the joint 2-template fit's residual fraction is at least
        min_pair_improvement (relative, e.g. 0.15 = 15 percentage
        points) lower than the best single-template fit's residual
        fraction -- the pair genuinely explains more than either
        component alone, not just fits marginally better by having an
        extra free parameter -- AND
      - the pair fit's own absolute residual fraction is at most
        max_pair_residual_frac -- a GENUINE superposition should be
        explained almost entirely by its two components (see the
        validated t-SNE n=3 example: 80% -> 14% residual), not just
        "improved somewhat from an already-bad single-template fit"
        (e.g. 88% -> 70% clears a relative-improvement bar but the pair
        still leaves 70% of the waveform unexplained -- not a real
        match). Without this second check, a handful of genuinely clean,
        distinct types can all flag each other as pairwise
        "explanations" purely because no single OTHER template fits any
        of them particularly well to begin with (observed empirically:
        every type in an otherwise-clean 5-type run got flagged before
        this guard was added).

    max_shift_samples should cover the range of relative discharge
    timing you'd plausibly see between two independently-firing units
    landing in the same detection window -- a few ms in samples.

    Returns a list of dicts, best-explained (lowest pair_residual_frac)
    first: type_c, type_i, type_j, shift_samples, coef_i, coef_j,
    single_residual_frac, pair_residual_frac, improvement."""
    templates = np.asarray(templates, dtype=float)
    n_types, n_samples = templates.shape
    results = []

    for c in range(n_types):
        target = templates[c]
        single = _best_single_fit(target, templates, exclude=c)
        if single is None:
            continue
        i, coef_i0, residual, single_rf = single

        best_pair = None
        for j in range(n_types):
            if j == c or j == i:
                continue
            tj = templates[j]
            corr = np.correlate(residual, tj, mode="full")
            shifts = np.arange(-(n_samples - 1), n_samples)
            valid = np.abs(shifts) <= max_shift_samples
            if not valid.any():
                continue
            masked = np.where(valid, corr, -np.inf)
            shift = int(shifts[int(np.argmax(masked))])
            tj_shifted = _shift_waveform(tj, shift)

            A = np.column_stack([templates[i], tj_shifted])
            coef, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
            coef = np.clip(coef, 0.0, None)
            pred = A @ coef
            resid2 = target - pred
            pair_rf = float(np.linalg.norm(resid2) / (np.linalg.norm(target) + 1e-12))

            if best_pair is None or pair_rf < best_pair["pair_residual_frac"]:
                best_pair = dict(type_c=c, type_i=i, type_j=j, shift_samples=shift,
                                  coef_i=float(coef[0]), coef_j=float(coef[1]),
                                  single_residual_frac=single_rf, pair_residual_frac=pair_rf)

        if best_pair is not None:
            best_pair["improvement"] = best_pair["single_residual_frac"] - best_pair["pair_residual_frac"]
            if (best_pair["improvement"] >= min_pair_improvement
                    and best_pair["pair_residual_frac"] <= max_pair_residual_frac):
                results.append(best_pair)

    results.sort(key=lambda r: r["pair_residual_frac"])
    return results


def format_superposition_report(candidates, type_names=None, fs=None):
    """Human-readable one-line-per-candidate summary, best-explained
    first. fs (Hz), if given, reports shift_samples as milliseconds too."""
    if not candidates:
        return "No superposition candidates found."
    name = (lambda c: type_names.get(c, str(c))) if type_names else str
    lines = []
    for r in candidates:
        shift_note = f"{r['shift_samples']} samples"
        if fs:
            shift_note += f" ({r['shift_samples'] / fs * 1000:+.2f} ms)"
        lines.append(
            f"type {name(r['type_c'])} ~= {r['coef_i']:.2f}*type{name(r['type_i'])} + "
            f"{r['coef_j']:.2f}*type{name(r['type_j'])} shifted {shift_note}: "
            f"residual {r['single_residual_frac']:.0%} (single) -> "
            f"{r['pair_residual_frac']:.0%} (pair), improvement {r['improvement']:.0%}"
        )
    return "\n".join(lines)
