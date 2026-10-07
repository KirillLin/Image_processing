import os
import glob
import csv

import cv2
import numpy as np

from config import (
    INPUT_DIR, OUTPUT_DIR, SUPPORTED_EXT, MAX_SIDE, SHOW_PLOTS,
    MIN_AREA, N_CLASSES, KMEANS_MAX_ITER, KMEANS_SEED, FEATURE_KEYS,
)
from preprocessing import (
    build_color_mask, clean_mask, filter_by_area, find_components_recursive,
    find_components_scan, filter_small_components,
)
from features import compute_features_for_component, features_to_matrix
from clustering import normalize_features, kmedoids_manual
from visualization import (
    draw_labeled_objects, draw_clustered_objects, show_images,
)


def process_one_image(image_path: str, output_subdir: str):
    os.makedirs(output_subdir, exist_ok=True)
    fname = os.path.basename(image_path)
    print(f"\n{'='*60}")
    print(f"Обработка: {fname}")
    print('='*60)

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

    print("\n  [ЭТАП 1] Удаление фона (HSV-сегментация)...")
    mask_raw = build_color_mask(image_bgr)
    save("02_mask_raw.png", mask_raw)

    mask_clean = clean_mask(mask_raw)
    mask_filtered = filter_by_area(mask_clean, MIN_AREA)
    save("03_mask_clean.png", mask_filtered)

    print("\n  [ЭТАП 2] Выделение связных областей (РЕКУРСИВНЫЙ алгоритм)...")
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

    save("04_objects_labeled.png",
         draw_labeled_objects(image_bgr, labels_clean, components))

    if len(components) < N_CLASSES:
        print(f"    [!] Объектов ({len(components)}) < классов ({N_CLASSES}). "
              f"Кластеризация пропущена.")
        return

    print("\n  [ЭТАП 3] Признаки объектов:")
    features_list = [
        compute_features_for_component(c, image_shape, image_gray)
        for c in components
    ]

    print(f"    {'#':>3} {'area':>6} {'perim':>6} {'compact':>8} "
          f"{'elong':>7} {'holes':>6} {'bright':>7}")
    for i, f in enumerate(features_list, 1):
        print(f"    {i:>3} {f['area']:>6} {f['perimeter']:>6} "
              f"{f['compactness']:>8.2f} {f['elongation']:>7.2f} "
              f"{f['holes']:>6} {f.get('brightness', 0):>7.1f}")

    csv_path = os.path.join(output_subdir, "05_features.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "area", "perimeter", "cy", "cx",
                      "compactness", "elongation", "orientation",
                      "m20", "m02", "m11", "holes", "brightness"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, feats in enumerate(features_list, 1):
            row = {"id": i}
            row.update(feats)
            writer.writerow(row)

    print(f"\n  [ЭТАП 4] Кластеризация k-means (k={N_CLASSES})...")
    X = features_to_matrix(features_list, keys=FEATURE_KEYS)
    X_norm = normalize_features(X)

    cluster_labels, medoid_indices = kmedoids_manual(
        X_norm, k=N_CLASSES, max_iter=KMEANS_MAX_ITER, seed=KMEANS_SEED
    )
    print(f"    Метки кластеров: {cluster_labels.tolist()}")

    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"    Распределение: {dict(zip(unique.tolist(), counts.tolist()))}")

    print("\n  [ЭТАП 5] Финальная раскраска...")
    result_black_bg = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_filtered)
    result_clustered = draw_clustered_objects(
        result_black_bg, labels_clean, components, cluster_labels
    )
    save("06_result_clustered.png", result_clustered)

    print(f"\n  [OK] Результаты: {output_subdir}")

    if SHOW_PLOTS:
        show_images(
            [image_bgr, mask_raw, mask_filtered,
             draw_labeled_objects(image_bgr, labels_clean, components),
             result_clustered],
            ["1. Оригинал",
             "2. Маска (HSV)",
             "3. Маска чистая",
             "4. Объекты + номера (этап 2)",
             "5. Финал: кластеры (этап 5)"],
            save_path=os.path.join(output_subdir, "steps.png"),
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