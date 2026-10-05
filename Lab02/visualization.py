"""
ЭТАП 5: Визуализация.

Содержит:
  - draw_labeled_objects   — раскраска каждой компоненты своим цветом
  - draw_clustered_objects — раскраска по классам кластеризации
  - show_images            — отображение нескольких картинок
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import CLASS_COLORS


def draw_labeled_objects(image_bgr: np.ndarray,
                         labels: np.ndarray,
                         components) -> np.ndarray:
    """
    Раскрашивает каждую компоненту своим уникальным цветом.
    Используется для визуализации «все объекты разные».
    """
    rng = np.random.default_rng(42)
    palette = rng.integers(50, 255, size=(len(components) + 1, 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)  # фон — чёрный

    result = image_bgr.copy()
    for label_id in range(1, len(components) + 1):
        mask = (labels == label_id)
        color = tuple(int(c) for c in palette[label_id])
        result[mask] = color
    return result


def draw_clustered_objects(image_bgr: np.ndarray,
                           labels: np.ndarray,
                           components,
                           cluster_labels: np.ndarray) -> np.ndarray:
    """
    Раскрашивает объекты по классам кластеризации.
    Класс 0 → красный, класс 1 → синий, класс 2 → зелёный.
    """
    result = image_bgr.copy()
    for i, label_id in enumerate(range(1, len(components) + 1)):
        mask = (labels == label_id)
        cls = cluster_labels[i]
        color = CLASS_COLORS[cls % len(CLASS_COLORS)]
        result[mask] = color
    return result


def show_images(images, titles, save_path=None, cols=3):
    """Отображение нескольких изображений в сетке."""
    n = len(images)
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(5 * cols, 5 * rows))
    for idx, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, idx + 1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()