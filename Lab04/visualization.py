import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from config import CLASS_COLORS


# ==========================================================
# ОТРИСОВКА
# ==========================================================
def draw_labeled_objects(image_bgr, labels, components):
    rng = np.random.default_rng(42)
    palette = rng.integers(80, 255, size=(len(components) + 1, 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)
    result = image_bgr.copy()
    for label_id in range(1, len(components) + 1):
        mask = (labels == label_id)
        result[mask] = tuple(int(c) for c in palette[label_id])
    return result


def draw_clustered_objects(image_bgr, labels, components, cluster_labels):
    result = image_bgr.copy()
    for i, label_id in enumerate(range(1, len(components) + 1)):
        mask = (labels == label_id)
        cls = int(cluster_labels[i])
        result[mask] = CLASS_COLORS[cls % len(CLASS_COLORS)]
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


# ==========================================================
# ЛР №3: РУЧНОЙ ПОВОРОТ КАЖДОЙ ЦИФРЫ (ПО ГЛАВНОЙ ОСИ)
# ==========================================================

# Если |асимметрия| по вертикали меньше этого значения, фигура считается
# вертикально-симметричной (3) и 180° разрешается по горизонтали.
SYMMETRY_EPS = 0.15

# Уточнение угла: диапазон и шаг поиска (градусы)
ANGLE_SEARCH_DEG = 25.0
ANGLE_STEP_DEG = 0.5


def _principal_axis(pixels):
    """
    Центр масс, смещения пикселей от центра и угол главной оси
    (координаты изображения: x вправо, y вниз).
    """
    ys = np.array([p[0] for p in pixels], dtype=np.float64)
    xs = np.array([p[1] for p in pixels], dtype=np.float64)
    cy, cx = ys.mean(), xs.mean()
    dy, dx = ys - cy, xs - cx

    m20 = np.mean(dx * dx)
    m02 = np.mean(dy * dy)
    m11 = np.mean(dx * dy)

    # arctan2 -> всегда именно ГЛАВНАЯ ось
    theta = 0.5 * np.arctan2(2.0 * m11, m20 - m02)
    return ys, xs, cy, cx, dy, dx, theta


def _project(dx, dy, theta):
    """Координаты точек в повёрнутой системе: a - горизонталь, b - вертикаль (вниз)."""
    c, s = np.cos(theta), np.sin(theta)
    a = dx * s - dy * c
    b = dx * c + dy * s
    return a, b


def _refine_angle(dx, dy, theta):
    """
    Ручной поиск угла вокруг главной оси: выбираем поворот, при котором
    ограничивающий прямоугольник фигуры имеет минимальную площадь.
    """
    best_theta = theta
    best_area = np.inf
    for d in np.arange(-ANGLE_SEARCH_DEG, ANGLE_SEARCH_DEG + 1e-9, ANGLE_STEP_DEG):
        t = theta + np.deg2rad(d)
        a, b = _project(dx, dy, t)
        area = (a.max() - a.min()) * (b.max() - b.min())
        if area < best_area - 1e-9:
            best_area = area
            best_theta = t
    return best_theta


def _skewness(values):
    std = values.std()
    if std < 1e-9:
        return 0.0
    return float(np.mean(values ** 3) / std ** 3)


def _need_flip_180(a, b):
    """
    Нужно ли довернуть фигуру на 180° (для всех, кроме 8).

    - 1, 7: тяжёлая часть (перекладина, флажок) сверху -> skew_b > 0.
    - 3 (почти симметрична по вертикали): "спинка" справа -> skew_a < 0.
    """
    skew_b = _skewness(b)
    if abs(skew_b) >= SYMMETRY_EPS:
        return skew_b < 0
    skew_a = _skewness(a)
    return skew_a > 0


def _find_holes(occ):
    """
    Дырки фигуры на бинарной маске occ (True = фигура).
    Фон, достижимый из угла (он гарантированно пустой), - внешний;
    остальные области фона - дырки. Возвращает [(площадь, средняя строка)].
    """
    h, w = occ.shape
    seen = np.zeros((h, w), dtype=bool)

    def bfs(sy, sx):
        q = deque([(sy, sx)])
        seen[sy, sx] = True
        cells = []
        while q:
            y, x = q.popleft()
            cells.append((y, x))
            for ddy, ddx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + ddy, x + ddx
                if 0 <= ny < h and 0 <= nx < w and not occ[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        return cells

    bfs(0, 0)
    holes = []
    for y in range(h):
        for x in range(w):
            if not occ[y, x] and not seen[y, x]:
                cells = bfs(y, x)
                holes.append((len(cells), sum(c[0] for c in cells) / len(cells)))
    return holes


def _eight_needs_flip(occ):
    """
    8: большая петля должна быть снизу. Находим две самые большие дырки
    и сравниваем их положение по вертикали. Если дырок меньше двух,
    запасной вариант - сравнение числа пикселей сверху и снизу.
    """
    holes = sorted(_find_holes(occ), reverse=True)[:2]
    if len(holes) == 2:
        (_, row_big), (_, row_small) = holes
        return row_big < row_small          # большая выше -> переворачиваем
    rows = np.where(occ.any(axis=1))[0]
    mid = (rows.min() + rows.max()) / 2.0
    return occ[:int(mid)].sum() > occ[int(mid) + 1:].sum()


def _bilinear_sample(mask, xs, ys):
    """Билинейная интерполяция маски (float) в точках (xs, ys)."""
    h, w = mask.shape
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx = xs - x0
    fy = ys - y0

    def px(yy, xx):
        ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        out = np.zeros(xx.shape, dtype=np.float64)
        out[ok] = mask[yy[ok], xx[ok]]
        return out

    return (px(y0, x0) * (1 - fx) * (1 - fy) +
            px(y0, x0 + 1) * fx * (1 - fy) +
            px(y0 + 1, x0) * (1 - fx) * fy +
            px(y0 + 1, x0 + 1) * fx * fy)


def rotate_object_upright(image_bgr, result, pixels, holes=None):
    """
    Поворачивает одну фигуру так, чтобы её главная ось стала вертикальной,
    и рисует её в result на месте центра масс.
    Обратное отображение + билинейная интерполяция - всё вручную.
    """
    h, w = image_bgr.shape[:2]
    ys, xs, cy, cx, dy, dx, theta = _principal_axis(pixels)

    theta = _refine_angle(dx, dy, theta)
    c, s = np.cos(theta), np.sin(theta)

    a, b = _project(dx, dy, theta)

    iy = ys.astype(np.int64)
    ix = xs.astype(np.int64)
    color = np.round(image_bgr[iy, ix].mean(axis=0)).astype(np.uint8)

    src_mask = np.zeros((h, w), dtype=np.float64)
    src_mask[iy, ix] = 1.0

    radius = int(np.ceil(np.max(np.hypot(dx, dy)))) + 2
    cy_i, cx_i = int(round(cy)), int(round(cx))
    grid_a, grid_b = np.meshgrid(np.arange(-radius, radius + 1),
                                 np.arange(-radius, radius + 1))

    def render(flip):
        A = grid_a.astype(np.float64)
        B = grid_b.astype(np.float64)
        if flip:
            A, B = -A, -B
        src_dx = A * s + B * c
        src_dy = -A * c + B * s
        return _bilinear_sample(src_mask, cx + src_dx, cy + src_dy) > 0.5

    if holes == 2:
        flip = _eight_needs_flip(render(False))
    else:
        flip = _need_flip_180(a, b)

    occ = render(flip)

    gy = cy_i + grid_b
    gx = cx_i + grid_a
    valid = occ & (gy >= 0) & (gy < h) & (gx >= 0) & (gx < w)
    result[gy[valid], gx[valid]] = color


def rotate_each_object_manual(image_bgr, components, elongations,
                              round_threshold=1.5, holes=None):
    """
    Каждая цифра поворачивается отдельно: главная ось -> вертикаль,
    затем выбирается правильное направление (не вверх ногами).
    Круглые объекты (elongation < round_threshold) не трогаем.
    """
    result = np.zeros_like(image_bgr)

    for i, pixels in enumerate(components):
        if elongations[i] < round_threshold or len(pixels) < 10:
            for (py, px) in pixels:
                result[py, px] = image_bgr[py, px]
            continue

        n_holes = holes[i] if holes is not None else None
        rotate_object_upright(image_bgr, result, pixels, n_holes)

    return result