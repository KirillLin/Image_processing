import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MAX_SIDE = 500
SHOW_PLOTS = True

BLUE_RANGES = [
    (np.array([100, 140,  75]), np.array([130, 255, 255])),
    (np.array([ 95,  125, 120]), np.array([135, 255, 255])),
]

GREEN_RANGES = [
    (np.array([ 40,  70,  60]), np.array([ 85, 255, 255])),
]


MIN_AREA = 400
MAX_ASPECT = 2.5
MIN_FILL = 0.35


LPF_KERNEL = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]], dtype=np.float32) / 9.0

HPF_KERNEL = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]], dtype=np.float32)

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

#свой соболь
def sobel_manual(image_gray: np.ndarray) -> np.ndarray:
    Kx = np.array([[ 1, 0, -1],
                   [ 2, 0, -2],
                   [ 1, 0, -1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    img = image_gray.astype(np.float32)

    def convolve2d(image, kernel):
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        padded = np.pad(image, ((ph, ph), (pw, pw)), mode='edge')
        h, w = image.shape
        out = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                out[i, j] = np.sum(padded[i:i + kh, j:j + kw] * kernel)
        return out

    Gx = convolve2d(img, Kx)
    Gy = convolve2d(img, Ky)
    mag = np.sqrt(Gx ** 2 + Gy ** 2)
    if mag.max() > 0:
        mag = mag / mag.max() * 255
    return mag.astype(np.uint8)


#своя медиана
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

#не свой соболь
def sobel_cv2(image_gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(image_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)

#не своя медиана
def median_filter_cv2(image_gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.medianBlur(image_gray, ksize)

#не своё сглаживание
def lowpass_cv2(image_gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(image_gray, ddepth=-1, kernel=kernel)

#не своя резкость
def highpass_cv2(image_gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(image_gray, ddepth=-1, kernel=kernel)



def build_color_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for low, high in BLUE_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))
    for low, high in GREEN_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))

    return mask


def filter_components_by_shape(mask: np.ndarray,
                               min_area: int = 400,
                               max_aspect: float = 2.5,
                               min_fill: float = 0.35) -> np.ndarray:

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = np.zeros_like(mask)

    for i in range(1, num):  # 0 — фон
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > max_aspect:
            continue
        fill = area / float(max(1, w * h))
        if fill < min_fill:
            continue
        out[labels == i] = 255

    return out


def remove_background(image_bgr: np.ndarray):
    mask = build_color_mask(image_bgr)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    mask = filter_components_by_shape(mask, MIN_AREA, MAX_ASPECT, MIN_FILL)

    result = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    return result, mask

#обратная задача - для себя
def remove_foreground(image_bgr: np.ndarray):
    mask = build_color_mask(image_bgr)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)

    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask = cv2.erode(mask, k2, iterations=1)

    mask = filter_components_by_shape(
        mask,
        min_area=MIN_AREA,
        max_aspect=MAX_ASPECT,
        min_fill=MIN_FILL,
    )

    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k9)

    mask = cv2.dilate(mask, k3, iterations=1)

    inverted = cv2.bitwise_not(mask)

    result = cv2.bitwise_and(image_bgr, image_bgr, mask=inverted)
    return result, inverted

def process_one_image(image_path: str, output_subdir: str):
    os.makedirs(output_subdir, exist_ok=True)
    fname = os.path.basename(image_path)
    print(f"\n=== Обработка: {fname} ===")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f" Не удалось прочитать {image_path}, пропуск.")
        return

    h, w = image_bgr.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        image_bgr = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)
        print(f"    Ресайз: {w}x{h} -> {new_size[0]}x{new_size[1]}")

    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    def save(name, img):
        cv2.imwrite(os.path.join(output_subdir, name), img)

    save("01_original.png", image_bgr)
    print("    [1/9] sobel_manual...")
    sobel_manual_img = sobel_manual(image_gray)
    save("02_sobel_manual.png", sobel_manual_img)
    print("    [2/9] cv2.Sobel...")
    sobel_cv2_img = sobel_cv2(image_gray)
    save("03_sobel_cv2.png", sobel_cv2_img)
    print("    [3/9] median_filter_manual...")
    median_manual_img = median_filter_manual(image_gray, ksize=3)
    save("04_median_manual.png", median_manual_img)
    print("    [4/9] cv2.medianBlur...")
    median_cv2_img = median_filter_cv2(image_gray, ksize=3)
    save("05_median_cv2.png", median_cv2_img)
    print("    [5/9] lowpass_cv2...")
    lowpass_img = lowpass_cv2(image_gray, LPF_KERNEL)
    save("06_lowpass_cv2.png", lowpass_img)
    print("    [6/9] highpass_cv2...")
    highpass_img = highpass_cv2(image_gray, HPF_KERNEL)
    save("07_highpass_cv2.png", highpass_img)
    print("    [7/9] remove_background...")
    result_no_bg, clean_mask = remove_background(image_bgr)
    print("    [9/10] remove_foreground...")
    result_no_fg, fg_mask = remove_foreground(image_bgr)
    save("11_mask_foreground.png", fg_mask)
    save("12_result_no_foreground.png", result_no_fg)

    raw_mask = build_color_mask(image_bgr)
    save("08_mask_raw.png", raw_mask)
    save("09_mask_clean.png", clean_mask)
    save("10_result_no_background.png", result_no_bg)
    print(f"  [OK] Результаты в: {output_subdir}")

    if SHOW_PLOTS:
        show_images(
            [image_bgr, sobel_manual_img, sobel_cv2_img,
             median_manual_img, median_cv2_img,
             lowpass_img, highpass_img,
             clean_mask, result_no_bg,
             fg_mask, result_no_fg],
            [f"{fname}: оригинал",
             "Собель — АВТОРСКИЙ",
             "Собель — OpenCV",
             "Медиана — АВТОРСКАЯ",
             "Медиана — OpenCV",
             "Низкочастотный — OpenCV",
             "Высокочастотный — OpenCV",
             "Маска объекта",
             "ИТОГ: Заданные цвета",
             "Маска фона (инверсия)",
             "ИТОГ: Без С/З"],
            save_path=os.path.join(output_subdir, "steps.png"),
            cols=4
        )

def show_images(images, titles, save_path=None, cols=4):
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


def main():
    files = []
    for ext in SUPPORTED_EXT:
        files.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext}")))
        files.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext.upper()}")))
    files = sorted(set(files))

    if not files:
        print(f"[!] В папке {INPUT_DIR} нет изображений.")
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
            print(f" Ошибка при обработке {path}: {e}")

    print(f"\n=== Готово! Все результаты в: {os.path.abspath(OUTPUT_DIR)} ===")


if __name__ == "__main__":
    main()