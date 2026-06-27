"""Density & Coverage, vendored verbatim (logic) from
clovaai/generative-evaluation-prdc (MIT licence) -- Naeem et al., ICML 2020,
"Reliable Fidelity and Diversity Metrics for Generative Models".

These operate on feature vectors (Inception 2048-d by convention, or the cheap
`phi` proxy here).  Density can exceed 1; Coverage is in [0,1]; higher is better.
"""
import numpy as np
import sklearn.metrics


def compute_pairwise_distance(data_x, data_y=None):
    if data_y is None:
        data_y = data_x
    return sklearn.metrics.pairwise_distances(data_x, data_y, metric="euclidean")


def get_kth_value(unsorted, k, axis=-1):
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    return k_smallests.max(axis=axis)


def compute_nearest_neighbour_distances(input_features, nearest_k):
    distances = compute_pairwise_distance(input_features)
    return get_kth_value(distances, k=nearest_k + 1, axis=-1)  # +1 skips self


def compute_prdc(real_features, fake_features, nearest_k):
    real_nn = compute_nearest_neighbour_distances(real_features, nearest_k)
    fake_nn = compute_nearest_neighbour_distances(fake_features, nearest_k)
    d_rf = compute_pairwise_distance(real_features, fake_features)

    precision = (d_rf < np.expand_dims(real_nn, 1)).any(0).mean()
    recall = (d_rf < np.expand_dims(fake_nn, 0)).any(1).mean()
    density = (1.0 / nearest_k) * (d_rf < np.expand_dims(real_nn, 1)).sum(0).mean()
    coverage = (d_rf.min(1) < real_nn).mean()
    return dict(precision=float(precision), recall=float(recall),
                density=float(density), coverage=float(coverage))
