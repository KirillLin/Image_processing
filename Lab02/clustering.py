import numpy as np

from config import KMEANS_MAX_ITER, KMEANS_SEED

def normalize_features(features_matrix: np.ndarray) -> np.ndarray:
    mean = features_matrix.mean(axis=0)
    std = features_matrix.std(axis=0)
    std[std == 0] = 1
    return (features_matrix - mean) / std


def kmeans_manual(X: np.ndarray, k: int,
                  max_iter: int = KMEANS_MAX_ITER,
                  seed: int = KMEANS_SEED):
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]

    centers = X[rng.choice(n_samples, size=k, replace=False)].copy()
    labels = np.zeros(n_samples, dtype=int)

    for iteration in range(max_iter):
        distances = np.zeros((n_samples, k))
        for i in range(n_samples):
            for j in range(k):
                distances[i, j] = np.linalg.norm(X[i] - centers[j])
        new_labels = np.argmin(distances, axis=1)

        if iteration > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for j in range(k):
            points = X[labels == j]
            if len(points) > 0:
                centers[j] = points.mean(axis=0)

    return labels, centers