"""
ЭТАП 4: Кластеризация.

Три алгоритма из методички:
  - k-means    (стр. 34)
  - k-medians  (стр. 35)
  - k-medoids  (стр. 35)
"""

import numpy as np

from config import KMEANS_MAX_ITER, KMEANS_SEED


def normalize_features(features_matrix: np.ndarray) -> np.ndarray:
    """Z-нормализация: (x - mean) / std."""
    mean = features_matrix.mean(axis=0)
    std = features_matrix.std(axis=0)
    std[std == 0] = 1
    return (features_matrix - mean) / std


# ==========================================================
# K-MEANS (стр. 34)
# ==========================================================
def kmeans_manual(X: np.ndarray, k: int,
                  max_iter: int = KMEANS_MAX_ITER,
                  seed: int = KMEANS_SEED):
    """
    K-means: центр кластера — СРЕДНЕЕ объектов.

    Формула (методичка, стр. 34):
        1. Случайно выбрать k средних.
        2. Отнести каждую точку к ближайшему среднему.
        3. Пересчитать средние.
        4. Повторять, пока кластеры не перестанут меняться.
    """
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
                centers[j] = points.mean(axis=0)   # ← СРЕДНЕЕ

    return labels, centers


# ==========================================================
# K-MEDIANS (стр. 35) — НОВОЕ
# ==========================================================
def kmedians_manual(X: np.ndarray, k: int,
                    max_iter: int = KMEANS_MAX_ITER,
                    seed: int = KMEANS_SEED):
    """
    K-medians: центр кластера — МЕДИАНА объектов.
    Устойчивее к выбросам, чем k-means.

    Отличие от k-means — одна строка: median вместо mean.
    """
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
                centers[j] = np.median(points, axis=0)   # ← МЕДИАНА

    return labels, centers


# ==========================================================
# K-MEDOIDS (стр. 35) — НОВОЕ
# ==========================================================
def kmedoids_manual(X: np.ndarray, k: int,
                    max_iter: int = KMEANS_MAX_ITER,
                    seed: int = KMEANS_SEED):
    """
    K-medoids: центр кластера — ОДИН ИЗ ОБЪЕКТОВ (медоид).
    Медоид = объект с минимальной суммой расстояний до остальных
    объектов того же кластера.

    Возвращает:
        labels        — метки кластеров
        medoid_indices — индексы объектов-медоидов в X
    """
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]

    medoid_indices = rng.choice(n_samples, size=k, replace=False)
    labels = np.zeros(n_samples, dtype=int)

    for iteration in range(max_iter):
        # Отнести к ближайшему медоиду
        distances = np.zeros((n_samples, k))
        for i in range(n_samples):
            for j in range(k):
                distances[i, j] = np.linalg.norm(X[i] - X[medoid_indices[j]])
        new_labels = np.argmin(distances, axis=1)

        if iteration > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Пересчёт медоидов
        for j in range(k):
            points_idx = np.where(labels == j)[0]
            if len(points_idx) == 0:
                continue

            best_cost = float('inf')
            best_idx = medoid_indices[j]

            for candidate in points_idx:
                cost = sum(
                    np.linalg.norm(X[candidate] - X[other])
                    for other in points_idx
                )
                if cost < best_cost:
                    best_cost = cost
                    best_idx = candidate
            medoid_indices[j] = best_idx

    return labels, medoid_indices