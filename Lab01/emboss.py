import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input1")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output1")

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MAX_SIDE = 800
SHOW_PLOTS = True

EMBOSS_KERNEL = np.array([[ 0,  1,  0],
                          [ 1,  0, -1],
                          [ 0, -1,  0]], dtype=np.float32)

EMBOSS_OFFSET = 128

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


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

def emboss_manual(image_gray: np.ndarray,
                  kernel: np.ndarray = EMBOSS_KERNEL,
                  offset: int = EMBOSS_OFFSET) -> np.ndarray:

    img = image_gray.astype(np.float32)
    result = convolve2d(img, kernel)
    result = result + offset
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)

def show_images(images, titles, save_path=None, cols=2):
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


def process_one_image(image_path: str, output_subdir: str):
    os.makedirs(output_subdir, exist_ok=True)
    fname = os.path.basename(image_path)
    print(f"\n Обработка: {fname} ")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"  Не удалось прочитать {image_path}, пропускаю.")
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
    save("02_original_gray.png", image_gray)

    # ---- Эффект тиснения (авторская реализация) ----
    print("    emboss_manual...")
    emboss_img = emboss_manual(image_gray)
    save("03_emboss.png", emboss_img)

    print(f"  [OK] Результаты в: {output_subdir}")

    # ---- Визуализация ----
    if SHOW_PLOTS:
        show_images(
            [image_bgr, image_gray, emboss_img],
            [f"{fname}: оригинал",
             "Оригинал (grayscale)",
             "Тиснение — АВТОРСКАЯ реализация"],
            save_path=os.path.join(output_subdir, "steps.png"),
            cols=3
        )

def main():
    files = []
    for ext in SUPPORTED_EXT:
        files.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext}")))
        files.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext.upper()}")))
    files = sorted(set(files))

    if not files:
        print(f" В папке {INPUT_DIR} нет изображений.")
        print(f"    Положите туда файл (jpg/png) и запустите снова.")
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
            print(f"  Ошибка при обработке {path}: {e}")

    print(f"\n Готово! Все результаты в: {os.path.abspath(OUTPUT_DIR)} ")


if __name__ == "__main__":
    main()