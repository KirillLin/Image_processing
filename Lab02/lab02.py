"""
Лабораторная работа №3: Манипуляции с объектами изображения.

Задача: повернуть каждую цифру так, чтобы она была в удобном
человекочитаемом формате (главная ось — вертикальная).
"""

import os
import glob
import cv2
import numpy as np

from config import (
    INPUT_DIR, OUTPUT_DIR, SUPPORTED_EXT, MAX_SIDE, SHOW_PLOTS,
    MIN_AREA, N_CLASSES, KMEANS_MAX_ITER, KMEANS_SEED, FEATURE_KEYS,
    ROUND_ELONGATION_THRESHOLD,
)
from preprocessing import (
    build_color_mask, clean_mask, filter_by_area,
    find_components_recursive, filter_small_components,
)
from features import compute_features_for_component, features_to_matrix
from clustering import normalize_features, kmedoids_manual
from visualization import (
    draw_clustered_objects,
    show_images,
    rotate_each_object_manual
)


def process_one_image(image_path: str, output_subdir: str):
    """Полный пайплайн ЛР №3 для одного изображения."""
    os.makedirs(output_subdir, exist_ok=True)
    fname = os.path.basename(image_path)
    print(f"\n{'='*60}")
    print(f"Обработка: {fname}")
    print('='*60)

    # ---- Загрузка и ресайз ----
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"  [!] Не удалось прочитать")
        return

    h, w = image_bgr.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        image_bgr = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)
        print(f"  Ресайз: {w}x{h} -> {new_size[0]}x{new_size[1]}")

    image_shape = image_bgr.shape[:2]
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    def save(name, img):
        cv2.imwrite(os.path.join(output_subdir, name), img)

    save("01_original.png", image_bgr)

    # ===== ЭТАП 1. УДАЛЕНИЕ ФОНА =====
    print("\n  [ЭТАП 1] Удаление фона (HSV-сегментация)...")
    mask_raw = build_color_mask(image_bgr)
    mask_clean = clean_mask(mask_raw)
    mask_filtered = filter_by_area(mask_clean, MIN_AREA)

    # ===== ЭТАП 2. СВЯЗНЫЕ ОБЛАСТИ =====
    print("\n  [ЭТАП 2] Выделение связных областей (рекурсивный)...")
    labels, components = find_components_recursive(mask_filtered)
    components = filter_small_components(components, MIN_AREA)
    print(f"    Найдено объектов: {len(components)}")

    if len(components) == 0:
        print("    [!] Объектов не найдено")
        return

    labels_clean = np.zeros_like(labels)
    for new_id, pixels in enumerate(components, 1):
        for (y, x) in pixels:
            labels_clean[y, x] = new_id

    # ===== ЭТАП 3. ПРИЗНАКИ =====
    print("\n  [ЭТАП 3] Признаки объектов:")
    features_list = [
        compute_features_for_component(c, image_shape, image_gray)
        for c in components
    ]

    print(f"    {'#':>3} {'elong':>7} {'orient_deg':>10} {'holes':>6}")
    for i, f in enumerate(features_list, 1):
        orient_deg = np.degrees(f['orientation'])
        print(f"    {i:>3} {f['elongation']:>7.2f} {orient_deg:>10.2f} {f['holes']:>6}")

    # ===== ЭТАП 4. КЛАСТЕРИЗАЦИЯ =====
    print(f"\n  [ЭТАП 4] Кластеризация k-medoids (k={N_CLASSES})...")

    if len(components) >= N_CLASSES:
        X = features_to_matrix(features_list, keys=FEATURE_KEYS)
        X_norm = normalize_features(X)
        cluster_labels, medoid_indices = kmedoids_manual(
            X_norm, k=N_CLASSES, max_iter=KMEANS_MAX_ITER, seed=KMEANS_SEED
        )
        print(f"    Метки кластеров: {cluster_labels.tolist()}")
    else:
        print(f"    [!] Объектов ({len(components)}) < классов ({N_CLASSES}). "
              f"Кластеризация пропущена.")
        cluster_labels = np.zeros(len(components), dtype=int)

    # ===== ЭТАП 5. РАСКРАСКА (финал ЛР №2) =====
    print("\n  [ЭТАП 5] Раскраска по классам (финал ЛР №2)...")
    result_black_bg = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_filtered)
    result_clustered = draw_clustered_objects(
        result_black_bg, labels_clean, components, cluster_labels
    )
    save("02_result_lr2.png", result_clustered)

    print("\n  [ЭТАП 6] Поворот цифр (ЛР №3)...")
    print("    Поворот КАЖДОЙ цифры отдельно (полностью авторский код)...")

    elongations = [f['elongation'] for f in features_list]
    holes = [f['holes'] for f in features_list]

    result_rotated_each = rotate_each_object_manual(
        result_clustered,
        components,
        elongations,
        round_threshold=ROUND_ELONGATION_THRESHOLD,
        holes=holes,
    )
    save("04_rotated_each.png", result_rotated_each)

    print(f"\n  [OK] Результаты: {output_subdir}")

    # ===== ВИЗУАЛИЗАЦИЯ =====
    if SHOW_PLOTS:
        show_images(
            [image_bgr,
             result_clustered,
             result_rotated_each],
            ["1. Оригинал",
             "2. Финал ЛР №2 (кластеры)",
             "3. ЛР №3 (поворот по orientation)"],
            save_path=os.path.join(output_subdir, "steps_lr3.png"),
            cols=3,
        )


def main():
    files = []
    for ext in SUPPORTED_EXT:
        files.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext}")))
        files.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext.upper()}")))
    files = sorted(set(files))

    if not files:
        print(f"[!] В папке {INPUT_DIR} нет изображений")
        return

    print(f"Найдено изображений: {len(files)}")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    for idx, path in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        output_subdir = os.path.join(OUTPUT_DIR, stem)
        print(f"\n[{idx}/{len(files)}] {os.path.basename(path)}")
        try:
            process_one_image(path, output_subdir)
        except Exception as e:
            import traceback
            print(f"  [!] Ошибка: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Готово: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()