"""
Лабораторная работа №2: Кластерный анализ в распознавании образов.

Главный скрипт — оркестратор. Связывает модули:
  - config         — настройки
  - preprocessing  — удаление фона, связные области
  - features       — признаки объектов
  - clustering     — k-means
  - visualization  — рисование и отображение
"""

import os
import glob
import csv

import cv2
import numpy as np

from config import (
    INPUT_DIR, OUTPUT_DIR, SUPPORTED_EXT, MAX_SIDE, SHOW_PLOTS,
    MIN_AREA, N_CLASSES, KMEANS_MAX_ITER, KMEANS_SEED,
)
from preprocessing import (
    build_color_mask, clean_mask, filter_by_area,
    erode_to_split, find_components_bfs, filter_small_components,
)
from features import compute_features_for_component, features_to_matrix
from clustering import normalize_features, kmeans_manual
from visualization import (
    draw_labeled_objects, draw_clustered_objects, show_images,
)


def process_one_image(image_path: str, output_subdir: str):
    """Полный пайплайн для одного изображения."""
    os.makedirs(output_subdir, exist_ok=True)
    fname = os.path.basename(image_path)
    print(f"\n=== Обработка: {fname} ===")

    # ---- Загрузка и ресайз ----
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"  [!] Не удалось прочитать {image_path}")
        return

    h, w = image_bgr.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        image_bgr = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)
        print(f"    Ресайз: {w}x{h} -> {new_size[0]}x{new_size[1]}")

    def save(name, img):
        cv2.imwrite(os.path.join(output_subdir, name), img)

    save("01_original.png", image_bgr)

    # ---- ЭТАП 1. Удаление фона ----
    print("  [1/5] Удаление фона...")
    mask_raw = build_color_mask(image_bgr)
    save("02_mask_raw.png", mask_raw)

    mask_clean = clean_mask(mask_raw)
    mask_filtered = filter_by_area(mask_clean, MIN_AREA)
    save("03_mask_clean.png", mask_filtered)

    # ---- ЭТАП 2. Разъединение объектов (эрозия) + BFS ----
    print("  [2/5] Разъединение объектов (эрозия)...")
    #mask_split = erode_to_split(mask_filtered, iterations=1)
    mask_split = mask_filtered

    print("  [3/5] Выделение связных областей (BFS)...")
    labels, components = find_components_bfs(mask_split)
    components = filter_small_components(components, MIN_AREA)
    print(f"    Найдено объектов: {len(components)}")

    # Пересобираем labels (только выжившие компоненты)
    labels_clean = np.zeros_like(labels)
    for new_id, pixels in enumerate(components, 1):
        for (y, x) in pixels:
            labels_clean[y, x] = new_id

    save("04_objects_labeled.png",
         draw_labeled_objects(image_bgr, labels_clean, components))

    # Если объектов меньше классов — кластеризация невозможна
    if len(components) < N_CLASSES:
        print(f"  [!] Объектов меньше, чем классов ({len(components)} < {N_CLASSES}). "
              f"Пропускаю кластеризацию.")
        return

    # ---- ЭТАП 3. Вычисление признаков ----
    print("  [4/5] Вычисление признаков...")
    features_list = [compute_features_for_component(c) for c in components]

    # Сохранение в CSV
    csv_path = os.path.join(output_subdir, "06_features.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "area", "perimeter", "cy", "cx", "compactness", "elongation"])
        writer.writeheader()
        for i, feats in enumerate(features_list, 1):
            row = {"id": i}
            row.update(feats)
            writer.writerow(row)

    # ---- ЭТАП 4. Кластеризация ----
    print("  [5/5] Кластеризация k-means...")
    X = features_to_matrix(features_list, keys=("area", "compactness", "elongation"))
    X_norm = normalize_features(X)

    cluster_labels, centers = kmeans_manual(
        X_norm, k=N_CLASSES, max_iter=KMEANS_MAX_ITER, seed=KMEANS_SEED
    )
    print(f"    Классы: {cluster_labels.tolist()}")

    # ---- ЭТАП 5. Раскраска ----
    result_black_bg = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_filtered)
    result_clustered = draw_clustered_objects(
        result_black_bg, labels_clean, components, cluster_labels
    )
    save("05_objects_clustered.png", result_clustered)

    print(f"  [OK] Результаты в: {output_subdir}")

    # ---- Визуализация ----
    if SHOW_PLOTS:
        show_images(
            [image_bgr, mask_filtered, result_clustered],
            [f"{fname}: оригинал",
             "Маска объектов",
             "Кластеризация (k-means, k=3)"],
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
        print(f"[!] В папке {INPUT_DIR} нет изображений.")
        print(f"    Скопируйте датасет из Lab01.")
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
            print(f"  [!] Ошибка при обработке {path}: {e}")

    print(f"\n=== Готово! Все результаты в: {os.path.abspath(OUTPUT_DIR)} ===")


if __name__ == "__main__":
    main()