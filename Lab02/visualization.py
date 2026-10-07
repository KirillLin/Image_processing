import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import CLASS_COLORS


def draw_labeled_objects(image_bgr: np.ndarray,
                         labels: np.ndarray,
                         components) -> np.ndarray:

    rng = np.random.default_rng(42)
    palette = rng.integers(80, 255, size=(len(components) + 1, 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)

    result = image_bgr.copy()

    for label_id in range(1, len(components) + 1):
        mask = (labels == label_id)
        color = tuple(int(c) for c in palette[label_id])
        result[mask] = color

    for comp_id, pixels in enumerate(components, 1):
        ys = [p[0] for p in pixels]
        xs = [p[1] for p in pixels]
        cy = int(sum(ys) / len(ys))
        cx = int(sum(xs) / len(xs))

        text = str(comp_id)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        cv2.rectangle(result,
                      (cx - tw // 2 - 4, cy - th // 2 - 4),
                      (cx + tw // 2 + 4, cy + th // 2 + 4),
                      (0, 0, 0), -1)
        cv2.putText(result, text, (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return result


def draw_clustered_objects(image_bgr: np.ndarray,
                           labels: np.ndarray,
                           components,
                           cluster_labels: np.ndarray) -> np.ndarray:
    result = image_bgr.copy()
    for i, label_id in enumerate(range(1, len(components) + 1)):
        mask = (labels == label_id)
        cls = int(cluster_labels[i])
        color = CLASS_COLORS[cls % len(CLASS_COLORS)]
        result[mask] = color
    return result


def show_images(images, titles, save_path=None, cols=3):
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