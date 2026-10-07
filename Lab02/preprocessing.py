import cv2
import numpy as np
from collections import deque

from config import (
    BLUE_RANGES, GREEN_RANGES,
    MIN_AREA, MAX_AREA_RATIO,
)


def build_color_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for low, high in BLUE_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))
    for low, high in GREEN_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))

    return mask


def clean_mask(mask: np.ndarray) -> np.ndarray:
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)

def find_components_scan(mask: np.ndarray):

    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    equivalences = {}   # {label: root_label}
    next_label = 1

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0:
                continue

            B = labels[y - 1, x] if y > 0 else 0
            C = labels[y, x - 1] if x > 0 else 0

            if B == 0 and C == 0:
                labels[y, x] = next_label
                next_label += 1
            elif B != 0 and C == 0:
                labels[y, x] = B
            elif B == 0 and C != 0:
                labels[y, x] = C
            else:
                if B == C:
                    labels[y, x] = B
                else:
                    labels[y, x] = min(B, C)
                    equivalences[max(B, C)] = min(B, C)

    def find_root(label):
        path = []
        while label in equivalences:
            path.append(label)
            label = equivalences[label]
        for p in path:
            equivalences[p] = label
        return label

    for y in range(h):
        for x in range(w):
            if labels[y, x] != 0:
                labels[y, x] = find_root(labels[y, x])

    components_dict = {}
    for y in range(h):
        for x in range(w):
            lbl = labels[y, x]
            if lbl != 0:
                components_dict.setdefault(lbl, []).append((y, x))

    components = list(components_dict.values())

    new_labels = np.zeros_like(labels)
    for new_id, pixels in enumerate(components, 1):
        for (y, x) in pixels:
            new_labels[y, x] = new_id

    return new_labels, components

def find_components_recursive(mask: np.ndarray):
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    components = []
    current_label = [0]

    def fill(y, x, lbl):

        if y < 0 or y >= h or x < 0 or x >= w:
            return

        if mask[y, x] == 0 or labels[y, x] != 0:
            return

        labels[y, x] = lbl
        pixels.append((y, x))

        fill(y - 1, x, lbl)
        fill(y + 1, x, lbl)
        fill(y, x - 1, lbl)
        fill(y, x + 1, lbl)

    for start_y in range(h):
        for start_x in range(w):
            if mask[start_y, start_x] > 0 and labels[start_y, start_x] == 0:
                current_label[0] += 1
                pixels = []
                fill(start_y, start_x, current_label[0])
                components.append(pixels)

    return labels, components

def filter_by_area(mask: np.ndarray,
                   min_area: int = MIN_AREA,
                   max_area_ratio: float = MAX_AREA_RATIO) -> np.ndarray:

    h, w = mask.shape
    max_area = int(h * w * max_area_ratio)

    _, components = find_components_scan(mask)

    out = np.zeros_like(mask)
    for pixels in components:
        area = len(pixels)
        if min_area <= area <= max_area:
            for (y, x) in pixels:
                out[y, x] = 255
    return out


def filter_small_components(components, min_area: int = MIN_AREA):
    return [c for c in components if len(c) >= min_area]