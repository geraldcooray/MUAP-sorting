"""Clustering algorithms (k-means, GMM, Student's-t mixture, HDBSCAN, SPC)
and post-clustering label utilities."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import gammaln, logsumexp
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


def relabel_dense(labels):
    """Renumber cluster labels to a contiguous 0..k-1 range, dropping any
    that ended up with zero members."""
    used = sorted(set(labels.tolist()))
    remap = {old: new for new, old in enumerate(used)}
    return np.array([remap[label] for label in labels]), len(used)


def refine_labels_by_firing_regularity(labels, peaks_t, features, n_clusters,
                                        isi_weight, isi_window_s, n_iters):
    """Nudge spikes toward whichever cluster keeps its firing pattern
    locally regular, not just whichever is closest in shape/amplitude/
    duration feature space.

    Each iteration re-scores every spike against every cluster as
    (feature distance to that cluster's centroid) + isi_weight *
    (|instantaneous rate this spike would create - that cluster's own
    local median rate over +/- isi_window_s|, in Hz). Using an absolute
    Hz difference (rather than a ratio to the local ISI) keeps the cost
    bounded and well-conditioned even when a cluster is firing sparsely
    nearby, unlike a relative-ISI-deviation formulation, which blows up
    in exactly that case. Reassigning to minimize this combined cost lets
    each unit's rate drift slowly over isi_window_s-scale epochs (as real
    motor units do) while penalizing spike-to-spike jitter in its
    instantaneous rate.

    Since re-scoring after reassignment can occasionally make the fit
    worse (this is coordinate descent on a moving target, not a
    guaranteed-convergent optimization), only the iteration with the
    lowest realized total cost is kept -- so this refinement can only
    match or improve on the starting (shape/amplitude/duration-only)
    clustering, never silently degrade it."""
    if isi_weight <= 0 or n_iters <= 0:
        return labels

    labels = labels.copy()
    n_spikes = len(peaks_t)
    order = np.argsort(peaks_t)  # searchsorted needs sorted times
    peaks_sorted = peaks_t[order]

    best_labels = labels.copy()
    best_cost = np.inf

    for _ in range(n_iters):
        centroids = np.stack([
            features[labels == c].mean(axis=0) if np.any(labels == c)
            else features.mean(axis=0)
            for c in range(n_clusters)
        ])
        feature_cost = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)

        temporal_cost = np.zeros((n_spikes, n_clusters))
        labels_sorted = labels[order]
        for c in range(n_clusters):
            times_c = peaks_sorted[labels_sorted == c]
            for i in range(n_spikes):
                t_i = peaks_t[i]
                other = np.delete(times_c, np.searchsorted(times_c, t_i)) \
                    if labels[i] == c and len(times_c) else times_c
                if len(other) == 0:
                    continue
                idx = np.searchsorted(other, t_i)
                candidates = []
                if idx > 0:
                    candidates.append(t_i - other[idx - 1])
                if idx < len(other):
                    candidates.append(other[idx] - t_i)
                candidate_isi = min(candidates)
                candidate_rate = 1.0 / candidate_isi if candidate_isi > 0 else 0.0

                local = other[np.abs(other - t_i) <= isi_window_s]
                if len(local) >= 2:
                    local_isi = np.median(np.diff(np.sort(local)))
                    local_rate = 1.0 / local_isi if local_isi > 0 else 0.0
                    temporal_cost[i, c] = abs(candidate_rate - local_rate)  # Hz

        total_cost = feature_cost + isi_weight * temporal_cost
        new_labels = np.argmin(total_cost, axis=1)
        realized_cost = total_cost[np.arange(n_spikes), new_labels].sum()

        if realized_cost < best_cost:
            best_cost = realized_cost
            best_labels = new_labels.copy()

        labels = new_labels

    return best_labels


def _multivariate_t_logpdf(X, mean, cov, dof):
    """Log density of a multivariate Student's t distribution at each row
    of X, via a Cholesky factorization of `cov` for numerical stability."""
    d = X.shape[1]
    diff = X - mean
    L = np.linalg.cholesky(cov)
    sol = np.linalg.solve(L, diff.T).T
    maha = np.sum(sol ** 2, axis=1)
    log_det = 2 * np.sum(np.log(np.diag(L)))
    log_norm = (gammaln((dof + d) / 2) - gammaln(dof / 2)
                - 0.5 * d * np.log(dof * np.pi) - 0.5 * log_det)
    log_kernel = -0.5 * (dof + d) * np.log1p(maha / dof)
    return log_norm + log_kernel


def cluster_tmixture(features, n_clusters, seed, dof=4.0, n_iters=100, tol=1e-4):
    """Cluster via a mixture of multivariate Student's t distributions
    (fixed degrees of freedom `dof`), fit with EM.

    A t-distribution's heavier tails (vs. a Gaussian mixture / k-means'
    implicit spherical-Gaussian assumption) make cluster fitting more
    robust to outlier waveforms -- e.g. from partially overlapping spikes
    or detection/alignment artifacts -- which would otherwise drag a
    Gaussian cluster's mean and covariance toward them. This robustness
    argument for t-mixtures is the same one used in the spike-sorting
    literature (e.g. Shoham, Fellows & Normann 2003).

    `dof` is fixed rather than estimated per-cluster (a common
    simplification): jointly estimating degrees of freedom in t-mixture EM
    is a known-unstable sub-problem, and a fixed, user-chosen value already
    captures the main practical benefit -- down-weighting outliers via the
    per-point weight `u` in the E-step -- without that instability.
    """
    n, d = features.shape
    eps = 1e-6 * np.eye(d)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(features)
    means = km.cluster_centers_.copy()
    init_labels = km.labels_
    covs = np.stack([
        np.cov(features[init_labels == k], rowvar=False) + eps
        if np.sum(init_labels == k) > d else np.eye(d)
        for k in range(n_clusters)
    ])
    weights = np.array([max(np.mean(init_labels == k), 1e-3) for k in range(n_clusters)])
    weights /= weights.sum()

    prev_ll = -np.inf
    for _ in range(n_iters):
        log_probs = np.stack([
            np.log(weights[k]) + _multivariate_t_logpdf(features, means[k], covs[k], dof)
            for k in range(n_clusters)
        ], axis=1)
        log_norm = logsumexp(log_probs, axis=1, keepdims=True)
        resp = np.exp(log_probs - log_norm)
        ll = log_norm.sum()

        for k in range(n_clusters):
            diff = features - means[k]
            L = np.linalg.cholesky(covs[k])
            sol = np.linalg.solve(L, diff.T).T
            maha = np.sum(sol ** 2, axis=1)
            u = (dof + d) / (dof + maha)  # down-weights points far from the mean
            ru = resp[:, k] * u
            Nk = resp[:, k].sum()

            means[k] = (ru[:, None] * features).sum(axis=0) / ru.sum()
            diff2 = features - means[k]
            covs[k] = (ru[:, None, None] * (diff2[:, :, None] * diff2[:, None, :])).sum(axis=0) / Nk + eps
            weights[k] = Nk / n

        if abs(ll - prev_ll) < tol * max(abs(prev_ll), 1.0):
            break
        prev_ll = ll

    log_probs = np.stack([
        np.log(weights[k]) + _multivariate_t_logpdf(features, means[k], covs[k], dof)
        for k in range(n_clusters)
    ], axis=1)
    return np.argmax(log_probs, axis=1)


def cluster_hdbscan(features, min_cluster_size=5, min_samples=None):
    """Cluster via HDBSCAN (density-based): infers the number of clusters
    directly from the data instead of taking it as an input, and explicitly
    labels low-density points as noise (-1) rather than forcing them into
    the nearest cluster -- useful for rejecting detection/alignment
    artifacts or rare overlapping-spike waveforms that don't match any real
    MUAP type."""
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    return model.fit_predict(features)


def _potts_knn_graph(features, n_neighbors):
    """K-nearest-neighbor graph with Gaussian interaction weights -- the
    graph SPC's Potts-model simulation runs on. Weight scale is set from
    the mean k-th-neighbor distance, so it adapts to the feature space's
    own density/spread instead of needing an absolute distance cutoff."""
    n = features.shape[0]
    k = min(n_neighbors, n - 1)
    dist, idx = NearestNeighbors(n_neighbors=k + 1).fit(features).kneighbors(features)
    local_scale = max(dist[:, -1].mean(), 1e-9)

    weights = {}
    for i in range(n):
        for jj in range(1, k + 1):
            j = int(idx[i, jj])
            edge = (i, j) if i < j else (j, i)
            w = np.exp(-(dist[i, jj] ** 2) / (2 * local_scale ** 2))
            weights[edge] = max(weights.get(edge, 0.0), w)

    edge_list = np.array(list(weights.keys()), dtype=int).reshape(-1, 2)
    edge_w = np.array(list(weights.values()))
    return edge_list, edge_w


def _wolff_step(spins, adj, q_states, temperature, rng):
    """One Wolff cluster-flip update of the q-state Potts model on the
    interaction graph `adj` (adjacency list of (neighbor, weight) pairs),
    at the given temperature. Grows a cluster of same-spin neighbors from a
    random seed (bond-activation probability 1 - exp(-w/T)), then flips the
    whole cluster to a new state at once -- this avoids the critical
    slowing-down of single-spin-flip updates near a phase transition, which
    is exactly the regime SPC's temperature sweep needs to resolve."""
    n = len(spins)
    seed = rng.integers(n)
    old_state = spins[seed]
    new_state = rng.integers(q_states - 1)
    if new_state >= old_state:
        new_state += 1

    in_cluster = np.zeros(n, dtype=bool)
    in_cluster[seed] = True
    stack = [seed]
    while stack:
        i = stack.pop()
        for j, w in adj[i]:
            if not in_cluster[j] and spins[j] == old_state and rng.random() < 1.0 - np.exp(-w / temperature):
                in_cluster[j] = True
                stack.append(j)
    spins[in_cluster] = new_state


def cluster_spc(features, seed, n_neighbors=10, q_states=20,
                 t_min=0.05, t_max=1.5, n_temps=20,
                 mc_steps=100, min_clus_frac=0.05):
    """Superparamagnetic clustering (Blatt, Wiseman & Domany 1996), the
    method WaveClus (Quiroga et al. 2004) pairs with wavelet-coefficient
    features -- see shape_features_wavelet. Spikes are nodes of a
    k-nearest-neighbor graph with Potts-model couplings; simulating the
    model (Wolff algorithm) across a temperature sweep and thresholding
    time-averaged spin-spin correlations at 0.5 yields clusters that are
    physically "stable" over a temperature range, without having to choose
    the number of clusters in advance -- as temperature rises, one giant
    cluster (T near 0, "ferromagnetic") gradually splits into meaningful
    sub-clusters (the "superparamagnetic" phase) before dissolving into all
    singletons (T large, "paramagnetic"). The temperature used is the
    lowest one (sweeping up from `t_min`) at which the second-largest
    cluster first grows past `min_clus_frac` of all spikes -- i.e. just as
    the superparamagnetic phase begins. Points left outside any
    sufficiently large cluster at that temperature are labeled -1 (noise).
    """
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    edge_list, edge_w = _potts_knn_graph(features, n_neighbors)
    adj = [[] for _ in range(n)]
    for (i, j), w in zip(edge_list, edge_w):
        adj[i].append((j, w))
        adj[j].append((i, w))
    min_clus = max(2, int(round(min_clus_frac * n)))

    labels = np.zeros(n, dtype=int)  # fallback if the sweep never transitions: one giant cluster
    transitioned = False
    for temperature in np.linspace(t_min, t_max, n_temps):
        spins = rng.integers(0, q_states, size=n)
        for _ in range(mc_steps // 2):  # equilibration
            _wolff_step(spins, adj, q_states, temperature, rng)

        same_count = np.zeros(len(edge_list))
        n_measure = max(1, mc_steps - mc_steps // 2)
        for _ in range(n_measure):
            _wolff_step(spins, adj, q_states, temperature, rng)
            same_count += spins[edge_list[:, 0]] == spins[edge_list[:, 1]]
        corr = same_count / n_measure

        strong_edges = edge_list[corr > 0.5]
        rows, cols = (strong_edges[:, 0], strong_edges[:, 1]) if len(strong_edges) else ([], [])
        graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        _, candidate_labels = connected_components(graph, directed=False)

        sizes = np.bincount(candidate_labels)
        sizes_sorted = np.sort(sizes)[::-1]
        labels = candidate_labels
        if len(sizes_sorted) > 1 and sizes_sorted[1] >= min_clus:
            transitioned = True
            break

    if not transitioned:
        print(f"SPC: no cluster transition found in T=[{t_min}, {t_max}] -- using the "
              f"highest temperature tried; consider raising --spc-t-max or lowering "
              f"--spc-min-clus-frac")

    sizes = np.bincount(labels)
    return np.where(sizes[labels] >= min_clus, labels, -1)


def cluster_spikes(features, n_clusters, seed, method="kmeans",
                    tmix_dof=4.0, tmix_iters=100,
                    hdbscan_min_cluster_size=5, hdbscan_min_samples=None,
                    spc_knn=10, spc_q_states=20,
                    spc_t_min=0.05, spc_t_max=1.5, spc_n_temps=20,
                    spc_mc_steps=100, spc_min_clus_frac=0.05):
    """Dispatch to one of the clustering algorithms above by name.
    "hdbscan"/"spc" ignore `n_clusters` (they pick their own cluster count)
    and can return -1 for spikes left unclustered as noise."""
    if method == "gmm":
        gmm = GaussianMixture(n_components=n_clusters, random_state=seed, n_init=10)
        return gmm.fit_predict(features)
    if method == "tmixture":
        return cluster_tmixture(features, n_clusters, seed, dof=tmix_dof, n_iters=tmix_iters)
    if method == "hdbscan":
        return cluster_hdbscan(features, min_cluster_size=hdbscan_min_cluster_size,
                                min_samples=hdbscan_min_samples)
    if method == "spc":
        return cluster_spc(features, seed, n_neighbors=spc_knn, q_states=spc_q_states,
                            t_min=spc_t_min, t_max=spc_t_max, n_temps=spc_n_temps,
                            mc_steps=spc_mc_steps, min_clus_frac=spc_min_clus_frac)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    return km.fit_predict(features)


def silhouette_sweep(features, n_values, seed, method="kmeans", **cluster_kwargs):
    """Fit `method` clustering at each candidate cluster count in
    `n_values` and score each with the silhouette coefficient (mean, over
    all spikes, of (b - a) / max(a, b), where a = mean distance to other
    points in the same cluster and b = mean distance to points in the
    nearest other cluster; ranges -1 to 1, higher = better-separated,
    more internally-consistent clusters) -- a data-driven check of which
    cluster count best matches the actual structure in feature space,
    instead of trusting a single firing-rate-based estimate (e.g.
    N_CLUSTERS = ceil(n_spikes / (MAX_RATE_HZ x duration))) on its own.

    Only meaningful for a fixed-n_clusters method (kmeans, gmm, tmixture);
    hdbscan/spc pick their own cluster count and shouldn't be swept here.

    Returns a dict {n_clusters: silhouette_score}, using nan for any n
    that's infeasible (< 2, >= n_spikes) or that the clustering collapses
    to fewer than 2 populated clusters (silhouette is undefined for a
    single cluster)."""
    scores = {}
    for n in n_values:
        if n < 2 or n >= features.shape[0]:
            scores[n] = float("nan")
            continue
        labels = cluster_spikes(features, n, seed, method=method, **cluster_kwargs)
        mask = labels != -1  # exclude any noise label, though fixed-n methods shouldn't emit one
        if mask.sum() < 2 or len(set(labels[mask].tolist())) < 2:
            scores[n] = float("nan")
            continue
        scores[n] = float(silhouette_score(features[mask], labels[mask]))
    return scores
