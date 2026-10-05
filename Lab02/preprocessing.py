"""
ЭТАПЫ 1-2: Удаление фона и выделение связных областей.

Содержит:
  - build_color_mask            — HSV-сегментация
  - clean_mask                  — морфология
  - filter_by_area              — фильтр по площади
  - erode_to_split              — эрозия
  - find_components_bfs         — авторский BFS
  - find_components_scan        — последовательное сканирование (методичка)
  - filter_small_components     — удаление мелких
"""

import cv2
import numpy as np
from collections import deque

from config import (
    BLUE_RANGES, GREEN_RANGES,
    MIN_AREA, MAX_AREA_RATIO,
)


def build_color_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Строит HSV-маску синего/зелёного."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for low, high in BLUE_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))
    for low, high in GREEN_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))

    return mask


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Морфология: только close, без open.
    open удаляет тонкие линии — утку.
    close заливает дырки.
    """
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
    return mask


def filter_by_area(mask: np.ndarray,
                   min_area: int = MIN_AREA,
                   max_area_ratio: float = MAX_AREA_RATIO) -> np.ndarray:
    """Оставляет только компоненты с площадью в [min_area, max_area]."""
    h, w = mask.shape
    total_pixels = h * w
    max_area = int(total_pixels * max_area_ratio)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = np.zeros_like(mask)

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            out[labels == i] = 255

    return out


def erode_to_split(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Эрозия — разъединение слипшихся объектов."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.erode(mask, k, iterations=iterations)


# ==========================================================
# ВЫДЕЛЕНИЕ СВЯЗНЫХ ОБЛАСТЕЙ — BFS
# ==========================================================
def find_components_bfs(mask: np.ndarray):
    """
    Авторский BFS (обход в ширину).
    Вариация рекурсивного алгоритма из методички (стр. 26) с очередью.
    """
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    components = []
    current_label = 0

    for start_y in range(h):
        for start_x in range(w):
            if mask[start_y, start_x] > 0 and labels[start_y, start_x] == 0:
                current_label += 1
                pixels = []
                queue = deque()
                queue.append((start_y, start_x))
                labels[start_y, start_x] = current_label

                while queue:
                    y, x = queue.popleft()
                    pixels.append((y, x))

                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if mask[ny, nx] > 0 and labels[ny, nx] == 0:
                                labels[ny, nx] = current_label
                                queue.append((ny, nx))

                components.append(pixels)

    return labels, components


# ==========================================================
# ВЫДЕЛЕНИЕ СВЯЗНЫХ ОБЛАСТЕЙ — ПОСЛЕДОВАТЕЛЬНОЕ СКАНИРОВАНИЕ (НОВОЕ)
# ==========================================================
def find_components_scan(mask: np.ndarray):
    """
    Метод последовательного сканирования (методичка, стр. 27).

    ПЕРВЫЙ ПРОХОД: помечаем пиксели по соседям B (сверху) и C (слева).
    Записываем эквивалентности, если B и C разные.

    ВТОРОЙ ПРОХОД: разрешаем эквивалентности, переразмечаем.
    """
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    equivalences = {}   # {label: root_label}
    next_label = 1

    # ---------- ПЕРВЫЙ ПРОХОД ----------
    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0:
                continue

            B = labels[y - 1, x] if y > 0 else 0     # сверху
            C = labels[y, x - 1] if x > 0 else 0     # слева

            if B == 0 and C == 0:
                # Новая компонента
                labels[y, x] = next_label
                next_label += 1
            elif B != 0 and C == 0:
                labels[y, x] = B
            elif B == 0 and C != 0:
                labels[y, x] = C
            else:
                # Оба != 0
                if B == C:
                    labels[y, x] = B
                else:
                    # Разные метки — берём меньшую, записываем эквивалентность
                    labels[y, x] = min(B, C)
                    equivalences[max(B, C)] = min(B, C)

    # ---------- РАЗРЕШЕНИЕ ЭКВИВАЛЕНТНОСТЕЙ ----------
    def find_root(label):
        """Итеративно идёт к корню цепочки эквивалентностей."""
        path = []
        while label in equivalences:
            path.append(label)
            label = equivalences[label]
        # Сжатие пути — все элементы указывают сразу на корень
        for p in path:
            equivalences[p] = label
        return label

    # ---------- ВТОРОЙ ПРОХОД ----------
    for y in range(h):
        for x in range(w):
            if labels[y, x] != 0:
                labels[y, x] = find_root(labels[y, x])

    # ---------- СБОРКА КОМПОНЕНТ ----------
    components_dict = {}
    for y in range(h):
        for x in range(w):
            lbl = labels[y, x]
            if lbl != 0:
                components_dict.setdefault(lbl, []).append((y, x))

    # Перенумеровать от 1 (после разрешения эквивалентностей могут быть дырки)
    components = list(components_dict.values())

    # Перестроить labels с непрерывной нумерацией
    new_labels = np.zeros_like(labels)
    for new_id, pixels in enumerate(components, 1):
        for (y, x) in pixels:
            new_labels[y, x] = new_id

    return new_labels, components


def filter_small_components(components, min_area: int = MIN_AREA):
    """Убирает компоненты меньше min_area."""
    return [c for c in components if len(c) >= min_area]