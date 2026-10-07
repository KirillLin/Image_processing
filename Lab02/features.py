import numpy as np
from collections import deque

def compute_area(pixels) -> int:
    return len(pixels)


def compute_perimeter(pixels) -> int:
    pixel_set = set(pixels)
    perimeter = 0

    for (y, x) in pixels:
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (y + dy, x + dx) not in pixel_set:
                perimeter += 1
                break
    return perimeter


def compute_center_mass(pixels):
    ys = [p[0] for p in pixels]
    xs = [p[1] for p in pixels]
    return sum(ys) / len(ys), sum(xs) / len(xs)


def compute_central_moments(pixels, cy, cx):
    m20 = m02 = m11 = 0

    for (y, x) in pixels:
        dy = y - cy
        dx = x - cx
        m20 += dx * dx
        m02 += dy * dy
        m11 += dx * dy

    n = len(pixels)
    return m20 / n, m02 / n, m11 / n


def compute_compactness(area, perimeter):
    if area == 0:
        return 0.0
    return (perimeter ** 2) / area


def compute_elongation(m20, m02, m11):
    diff = m20 - m02
    disc = np.sqrt(diff ** 2 + 4 * m11 ** 2)
    num = m20 + m02 + disc
    den = m20 + m02 - disc
    if den <= 0:
        return 1.0
    return num / den


def compute_orientation(m20, m02, m11):
    if m20 == m02:
        return 0.0
    return 0.5 * np.arctan(2 * m11 / (m20 - m02))

def count_holes(pixels, image_shape):
    h, w = image_shape

    comp_mask = np.zeros((h, w), dtype=np.uint8)
    for (y, x) in pixels:
        comp_mask[y, x] = 1

    inv = (comp_mask == 0).astype(np.uint8)

    visited = np.zeros((h, w), dtype=bool)
    holes = 0

    for start_y in range(h):
        for start_x in range(w):
            if inv[start_y, start_x] == 1 and not visited[start_y, start_x]:
                queue = deque()
                queue.append((start_y, start_x))
                visited[start_y, start_x] = True
                touches_border = False

                while queue:
                    y, x = queue.popleft()
                    if y == 0 or y == h - 1 or x == 0 or x == w - 1:
                        touches_border = True

                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if inv[ny, nx] == 1 and not visited[ny, nx]:
                                visited[ny, nx] = True
                                queue.append((ny, nx))

                if not touches_border:
                    holes += 1

    return holes

def compute_mean_brightness(pixels, image_gray):
    vals = [image_gray[y, x] for (y, x) in pixels]
    return float(np.mean(vals))

def compute_features_for_component(pixels, image_shape, image_gray=None) -> dict:
    area = compute_area(pixels)
    perimeter = compute_perimeter(pixels)
    cy, cx = compute_center_mass(pixels)
    m20, m02, m11 = compute_central_moments(pixels, cy, cx)

    compactness = compute_compactness(area, perimeter)
    elongation = compute_elongation(m20, m02, m11)
    orientation = compute_orientation(m20, m02, m11)
    holes = count_holes(pixels, image_shape)

    feats = {
        "area": area,
        "perimeter": perimeter,
        "cy": cy,
        "cx": cx,
        "m20": m20, "m02": m02, "m11": m11,
        "compactness": compactness,
        "elongation": elongation,
        "orientation": orientation,
        "holes": holes,
    }

    if image_gray is not None:
        feats["brightness"] = compute_mean_brightness(pixels, image_gray)

    return feats

def features_to_matrix(features_list, keys) -> np.ndarray:
    return np.array([
        [f[k] for k in keys]
        for f in features_list
    ], dtype=np.float32)