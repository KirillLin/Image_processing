import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt #pip install opencv-contrib-python

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")          # папка с исходниками
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")        # папка с результатами

# Расширения, которые будем обрабатывать
SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Режим отображения графиков: True — показывать окна, False — только сохранять
SHOW_PLOTS = True

def rgb_to_hsv_manual(image_rgb: np.ndarray) -> np.ndarray:

    img = image_rgb.astype(np.float32) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]

    max_c = np.max(img, axis=2)
    min_c = np.min(img, axis=2)
    delta = max_c - min_c

    # Hue
    h = np.zeros_like(max_c)
    mask = delta != 0

    # R — максимум
    m_r = mask & (max_c == r)
    h[m_r] = ((60 * ((g[m_r] - b[m_r]) / delta[m_r]) + 360) % 360) / 2.0
    # G — максимум
    m_g = mask & (max_c == g)
    h[m_g] = (60 * ((b[m_g] - r[m_g]) / delta[m_g]) + 120) / 2.0
    # B — максимум
    m_b = mask & (max_c == b)
    h[m_b] = (60 * ((r[m_b] - g[m_b]) / delta[m_b]) + 240) / 2.0

    # Saturation
    s = np.zeros_like(max_c)
    s[max_c != 0] = delta[max_c != 0] / max_c[max_c != 0]

    # Value
    v = max_c

    hsv = np.stack([h, s, v], axis=2)
    return hsv


def color_segmentation_manual(image_rgb: np.ndarray,
                              blue_h_range=(100, 140),
                              green_h_range=(40, 90),
                              s_min=0.15,
                              v_min=0.15) -> np.ndarray:

    hsv = rgb_to_hsv_manual(image_rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    mask_blue = (h >= blue_h_range[0]) & (h <= blue_h_range[1]) & (s >= s_min) & (v >= v_min)
    mask_green = (h >= green_h_range[0]) & (h <= green_h_range[1]) & (s >= s_min) & (v >= v_min)

    mask = (mask_blue | mask_green).astype(np.uint8) * 255
    return mask


def median_filter_manual(image_gray: np.ndarray, ksize: int = 3) -> np.ndarray:

    pad = ksize // 2
    padded = np.pad(image_gray, pad, mode='edge')
    h, w = image_gray.shape
    out = np.zeros_like(image_gray)

    for i in range(h):
        for j in range(w):
            window = padded[i:i + ksize, j:j + ksize]
            out[i, j] = np.median(window)

    return out


def sobel_manual(image_gray: np.ndarray) -> np.ndarray:

    Kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype=np.float32)

    img = image_gray.astype(np.float32)

    # Свёртка вручную (без scipy/cv2)
    def convolve2d(image, kernel):
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        h, w = image.shape
        out = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                out[i, j] = np.sum(padded[i:i + kh, j:j + kw] * kernel)
        return out

    Gx = convolve2d(img, Kx)
    Gy = convolve2d(img, Ky)

    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    # Нормировка в 0..255
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max() * 255
    return magnitude.astype(np.uint8)


def morphology_manual(mask: np.ndarray, kernel_size: int = 3,
                      operation: str = 'open') -> np.ndarray:

    k = np.ones((kernel_size, kernel_size), np.uint8)
    pad = kernel_size // 2
    padded = np.pad(mask, pad, mode='constant', constant_values=0)
    h, w = mask.shape

    def erode(m):
        out = np.zeros_like(m)
        for i in range(h):
            for j in range(w):
                window = padded[i:i + kernel_size, j:j + kernel_size]
                out[i, j] = 255 if np.all(window == 255) else 0
        return out

    def dilate(m):
        out = np.zeros_like(m)
        for i in range(h):
            for j in range(w):
                window = padded[i:i + kernel_size, j:j + kernel_size]
                out[i, j] = 255 if np.any(window == 255) else 0
        return out

    if operation == 'erode':
        return erode(mask)
    elif operation == 'dilate':
        return dilate(mask)
    elif operation == 'open':
        return dilate(erode(mask))
    elif operation == 'close':
        return erode(dilate(mask))
    else:
        raise ValueError("operation must be 'erode' | 'dilate' | 'open' | 'close'")

def color_segmentation_cv2(image_bgr: np.ndarray) -> np.ndarray:
    """Цветовая сегментация через OpenCV (HSV + inRange)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 40, 40])
    upper_blue = np.array([140, 255, 255])
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_or(mask_blue, mask_green)
    return mask


def median_filter_cv2(image_gray: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Медианный фильтр через OpenCV."""
    return cv2.medianBlur(image_gray, ksize)


def sobel_cv2(image_gray: np.ndarray) -> np.ndarray:
    """Оператор Собеля через OpenCV."""
    grad_x = cv2.Sobel(image_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)


def morphology_cv2(mask: np.ndarray, kernel_size: int = 5,
                   operation: str = 'open') -> np.ndarray:
    """Морфология через OpenCV."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if operation == 'open':
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    elif operation == 'erode':
        return cv2.erode(mask, kernel)
    elif operation == 'dilate':
        return cv2.dilate(mask, kernel)
    else:
        raise ValueError("operation must be 'open' | 'close' | 'erode' | 'dilate'")

def remove_background(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    return result


def extract_contour_and_content(image_bgr: np.ndarray,
                                 mask: np.ndarray,
                                 sobel_func,
                                 morph_func) -> tuple:
    # Шаг 1: морфология — ТОЛЬКО open, k=3
    clean_mask = morph_func(mask, 3, 'open')
    # close убираем — он залепит цифры внутри карточек

    # Шаг 2: удаление фона
    object_only = remove_background(image_bgr, clean_mask)

    # Шаг 3: контур (Собель по границе)
    edges = sobel_func(clean_mask)
    _, edges_bin = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

    # Шаг 4: накладываем контур
    result = object_only.copy()
    result[edges_bin > 0] = (0, 0, 255)

    return result, clean_mask, edges_bin


def show_images(images: list, titles: list, save_path: str = None):
    n = len(images)
    cols = min(n, 3)
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

def process_one_image(image_path: str, output_subdir: str):
    os.makedirs(output_subdir, exist_ok=True)
    fname = os.path.basename(image_path)
    print(f"\n=== Обработка: {fname} ===")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"  [!] Не удалось прочитать {image_path}, пропускаю.")
        return
    MAX_SIDE = 500
    h, w = image_bgr.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        print(f"    Ресайз до {image_bgr.shape[1]}x{image_bgr.shape[0]}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print("  >>> Вариант А: авторская реализация...")
    mask_manual = color_segmentation_manual(image_rgb)
    mask_manual_med = median_filter_manual(mask_manual, 3)
    result_manual, clean_mask_manual, edges_manual = extract_contour_and_content(
        image_bgr,
        mask_manual_med,
        sobel_func=sobel_manual,
        morph_func=morphology_manual
    )

    print("  >>> Вариант В: реализация через OpenCV...")
    mask_cv2 = color_segmentation_cv2(image_bgr)
    mask_cv2_med = median_filter_cv2(mask_cv2, 5)
    result_cv2, clean_mask_cv2, edges_cv2 = extract_contour_and_content(
        image_bgr,
        mask_cv2_med,
        sobel_func=sobel_cv2,
        morph_func=morphology_cv2
    )

    def save(name, img):
        cv2.imwrite(os.path.join(output_subdir, name), img)

    save("01_original.png", image_bgr)
    save("02_mask_manual.png", mask_manual)
    save("03_mask_manual_median.png", mask_manual_med)
    save("04_result_manual.png", result_manual)
    save("05_mask_cv2.png", mask_cv2)
    save("06_mask_cv2_median.png", mask_cv2_med)
    save("07_result_cv2.png", result_cv2)

    if SHOW_PLOTS:
        show_images(
            [image_bgr, mask_manual, mask_manual_med, result_manual],
            [f"{fname}: Оригинал",
             "Маска (автор.)",
             "Маска + медиана (автор.)",
             "ИТОГ вариант А"],
            save_path=os.path.join(output_subdir, "steps_manual.png")
        )
        show_images(
            [image_bgr, mask_cv2, mask_cv2_med, result_cv2],
            [f"{fname}: Оригинал",
             "Маска (OpenCV)",
             "Маска + медиана (OpenCV)",
             "ИТОГ вариант В"],
            save_path=os.path.join(output_subdir, "steps_cv2.png")
        )
        show_images(
            [result_manual, result_cv2],
            [f"ИТОГ А: {fname}", f"ИТОГ В: {fname}"],
            save_path=os.path.join(output_subdir, "comparison.png")
        )

    print(f"  [OK] Результаты в: {output_subdir}")

def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = []
    for ext in SUPPORTED_EXT:
        files.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext}")))
        files.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext.upper()}")))
    files = sorted(set(files))  # убираем дубликаты и сортируем

    if not files:
        print(f"[!] В папке {INPUT_DIR} нет изображений.")
        print(f"    Положите туда файлы с расширениями: {SUPPORTED_EXT}")
        return

    print(f"Найдено изображений: {len(files)}")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    # Обрабатываем каждое
    for idx, path in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        output_subdir = os.path.join(OUTPUT_DIR, stem)
        print(f"\n[{idx}/{len(files)}] {os.path.basename(path)}")
        try:
            process_one_image(path, output_subdir)
        except Exception as e:
            print(f"  [!] Ошибка при обработке {path}: {e}")
            continue

    print(f"\n=== Готово! Все результаты в: {os.path.abspath(OUTPUT_DIR)} ===")


if __name__ == "__main__":
    main()