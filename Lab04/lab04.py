import os
import sys
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import (
    INPUT_DIR,
    OUTPUT_DIR,
    SUPPORTED_EXT,
    MAX_SIDE,
    SHOW_PLOTS,
    MIN_AREA,
    ROUND_ELONGATION_THRESHOLD,
)
from preprocessing import (
    build_color_mask,
    clean_mask,
    filter_by_area,
    find_components_recursive,
    filter_small_components,
)
from features import compute_features_for_component
from visualization import show_images, rotate_object_upright

sys.setrecursionlimit(50000)

MODEL_PATH = os.path.join(OUTPUT_DIR, "mnist_cnn.h5")
FINETUNED_PATH = os.path.join(OUTPUT_DIR, "mnist_cnn_finetuned.h5")

RETRAIN = False
EPOCHS = 10
BATCH_SIZE = 128
USE_AUGMENTATION = True
ORIENTATION_CHECK = True
FINE_TUNE = False

TRUE_LABELS = {
    "1695138157752": [0, 8, 1, 7, 3],
    "1695138157765": [0, 8, 1, 7, 3],
    "1695138157776": [0, 8, 1, 7, 3],
    "1695138157805": [0, 1, 7],
}


def extract_digit_rois(image_path: str, pad: int = 100):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"  [!] Не удалось прочитать {image_path}")
        return []

    h, w = image_bgr.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        image_bgr = cv2.resize(
            image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )

    image_shape = image_bgr.shape[:2]
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    mask = filter_by_area(clean_mask(build_color_mask(image_bgr)), MIN_AREA)
    _, components = find_components_recursive(mask)
    components = filter_small_components(components, MIN_AREA)
    if not components:
        print("  [!] Объекты не найдены")
        return []

    features_list = [
        compute_features_for_component(c, image_shape, image_gray) for c in components
    ]

    digits_only = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    padded = cv2.copyMakeBorder(
        digits_only, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    rois = []
    for i, (pixels, f) in enumerate(zip(components, features_list), 1):
        canvas = np.zeros_like(padded)
        shifted = [(y + pad, x + pad) for (y, x) in pixels]

        if f["elongation"] < ROUND_ELONGATION_THRESHOLD or len(pixels) < 10:
            for y, x in shifted:
                canvas[y, x] = padded[y, x]
        else:
            rotate_object_upright(padded, canvas, shifted, f["holes"])

        ys, xs = np.where(canvas.any(axis=2))
        if len(ys) == 0:
            print(f"  [!] Объект #{i}: пустой ROI после поворота - пропуск")
            continue

        roi = canvas[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].copy()
        rois.append(
            {
                "roi_bgr": roi,
                "center_xy": (int(round(f["cx"])), int(round(f["cy"]))),
                "features": f,
            }
        )
    return rois


def to_gray_u8(img):
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return None

    if img.ndim == 3:
        c = img.shape[2]
        if c == 3:
            g = cv2.cvtColor(
                img.astype(np.uint8) if img.dtype != np.uint8 else img,
                cv2.COLOR_BGR2GRAY,
            )
        elif c == 4:
            g = cv2.cvtColor(
                img.astype(np.uint8) if img.dtype != np.uint8 else img,
                cv2.COLOR_BGRA2GRAY,
            )
        elif c == 1:
            g = img[:, :, 0]
        else:
            g = img.mean(axis=2)
    elif img.ndim == 2:
        g = img
    else:
        return None

    if g.dtype != np.uint8:
        g = g.astype(np.float32)
        if g.max() <= 1.0:
            g = g * 255.0
        g = np.clip(g, 0, 255).astype(np.uint8)
    return g


def normalize_roi_to_mnist(roi):
    g = to_gray_u8(roi)
    if g is None or g.shape[0] < 2 or g.shape[1] < 2 or int(g.max()) == int(g.min()):
        return None

    k = max(1, min(g.shape) // 8)
    corners = np.concatenate(
        [g[:k, :k].ravel(), g[:k, -k:].ravel(), g[-k:, :k].ravel(), g[-k:, -k:].ravel()]
    )
    if np.median(corners) > 127:
        g = 255 - g

    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)

    thr, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.where(g > thr)
    if len(ys) == 0:
        return None
    crop = g[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    ch, cw = crop.shape
    scale = 20.0 / max(ch, cw)
    new_h = max(1, int(round(ch * scale)))
    new_w = max(1, int(round(cw * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized

    m = cv2.moments(canvas)
    if m["m00"] > 0:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        shift = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
        canvas = cv2.warpAffine(
            canvas,
            shift,
            (28, 28),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    return canvas.astype(np.float32) / 255.0


def collect_dataset(files):
    per_image = []
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        print(f"\n--- {os.path.basename(path)} ---")
        rois = extract_digit_rois(path)

        raw, x28, centers = [], [], []
        for r in rois:
            n = normalize_roi_to_mnist(r["roi_bgr"])
            if n is None:
                print("  [!] ROI не удалось нормализовать - пропуск")
                continue
            raw.append(r["roi_bgr"])
            x28.append(n)
            centers.append(r["center_xy"])

        labels = TRUE_LABELS.get(stem)
        if labels is not None and len(labels) != len(rois):
            print(
                f"  [!] В TRUE_LABELS['{stem}'] {len(labels)} меток, а найдено "
                f"{len(rois)} объектов - метки для этого фото игнорируются"
            )
            labels = None
        elif labels is not None and len(rois) != len(x28):
            print("  [!] Часть ROI была пропущена - метки для этого фото игнорируются")
            labels = None

        for i, c in enumerate(centers, 1):
            print(f"    #{i}: центр (x={c[0]}, y={c[1]})")

        per_image.append(
            {
                "stem": stem,
                "raw": raw,
                "x28": np.array(x28, dtype=np.float32),
                "labels": None if labels is None else np.array(labels, dtype=int),
            }
        )
    return per_image


def load_mnist():
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]
    return x_train, y_train, x_test, y_test


def build_cnn():
    from tensorflow import keras

    L = keras.layers
    return keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            L.Conv2D(32, (3, 3), activation="relu"),
            L.MaxPooling2D((2, 2)),
            L.Conv2D(64, (3, 3), activation="relu"),
            L.MaxPooling2D((2, 2)),
            L.Flatten(),
            L.Dense(128, activation="relu"),
            L.Dropout(0.3),
            L.Dense(10, activation="softmax"),
        ],
        name="digits_cnn",
    )


def _compile(model, lr=1e-3):
    from tensorflow import keras

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def train_mnist(x_train, y_train):
    from tensorflow import keras

    L = keras.layers

    cnn = build_cnn()
    if USE_AUGMENTATION:
        augment = keras.Sequential(
            [
                L.RandomRotation(15 / 360.0, fill_mode="constant", fill_value=0.0),
                L.RandomTranslation(
                    2 / 28.0, 2 / 28.0, fill_mode="constant", fill_value=0.0
                ),
            ],
            name="augment",
        )
        inp = keras.Input(shape=(28, 28, 1))
        train_model = keras.Model(inp, cnn(augment(inp)))
    else:
        train_model = cnn

    _compile(train_model)
    history = train_model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=2,
    )
    _compile(cnn)
    return cnn, history


def save_model(model, path):
    try:
        model.save(path)
        print(f"  Модель сохранена: {path}")
    except Exception as e:
        alt = os.path.splitext(path)[0] + ".keras"
        model.save(alt)
        print(f"  [!] .h5 не удалось ({e}); сохранено как {alt}")


def load_or_train_model():
    from tensorflow import keras

    alt = os.path.splitext(MODEL_PATH)[0] + ".keras"
    for p in (MODEL_PATH, alt):
        if os.path.exists(p) and not RETRAIN:
            print(f"Загружаю модель: {p}")
            model = keras.models.load_model(p)
            _compile(model)
            return model, None

    x_train, y_train, x_test, y_test = load_mnist()

    _save_grid(
        [(x_train[i, :, :, 0] * 255).astype(np.uint8) for i in range(25)],
        [f"MNIST: {y_train[i]}" for i in range(25)],
        os.path.join(OUTPUT_DIR, "mnist_samples_5x5.png"),
        cols=5,
    )

    print("\nОбучение CNN на MNIST...")
    model, history = train_mnist(x_train, y_train)
    save_model(model, MODEL_PATH)
    plot_history(history, os.path.join(OUTPUT_DIR, "training_curves.png"))

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n  Accuracy на тесте MNIST: {test_acc * 100:.2f}%  (loss={test_loss:.4f})")
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    plot_confusion_matrix(
        y_test, y_pred, os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    )
    return model, test_acc


def predict_digits(model, x28, check_flip=ORIENTATION_CHECK):
    if len(x28) == 0:
        return (
            np.array([], int),
            np.array([]),
            np.zeros((0, 28, 28), np.float32),
            np.array([], bool),
        )

    X = x28[..., None]
    p0 = model.predict(X, verbose=0)
    preds = p0.argmax(axis=1)
    probs = p0.max(axis=1)
    used = x28.copy()
    flipped = np.zeros(len(x28), dtype=bool)

    if check_flip:
        x180 = x28[:, ::-1, ::-1]
        p1 = model.predict(x180[..., None], verbose=0)
        better = p1.max(axis=1) > probs
        preds = np.where(better, p1.argmax(axis=1), preds)
        probs = np.where(better, p1.max(axis=1), probs)
        used[better] = x180[better]
        flipped = better
    return preds, probs, used, flipped


def _save_grid(images, titles, path, cols=5):
    if SHOW_PLOTS:
        show_images(images, titles, save_path=path, cols=cols)
    else:
        orig_show = plt.show
        plt.show = lambda *a, **k: None
        try:
            show_images(images, titles, save_path=path, cols=cols)
        finally:
            plt.show = orig_show
    plt.close("all")


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истина")
    ax.set_title("Confusion matrix (MNIST test)")
    thr = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if cm[i, j] > thr else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix: {save_path}")


def plot_history(history, save_path):
    if history is None:
        return
    h = history.history
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(h["loss"], label="train")
    axes[0].plot(h["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(h["accuracy"], label="train")
    axes[1].plot(h["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def show_predictions(
    rois_28,
    preds,
    probs,
    true_labels=None,
    save_path=None,
    rois_raw=None,
    flipped=None,
    cols=8,
):
    n = len(rois_28)
    if n == 0:
        print("  [!] show_predictions: нет данных для отображения")
        return

    cols = max(1, min(cols, n))
    blocks = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        2 * blocks, cols, figsize=(2.0 * cols, 3.6 * blocks), squeeze=False
    )

    for ax in axes.ravel():
        ax.axis("off")

    for i in range(n):
        b, c = divmod(i, cols)
        ax_in, ax_out = axes[2 * b][c], axes[2 * b + 1][c]

        if rois_raw is not None:
            ax_in.imshow(
                (
                    cv2.cvtColor(rois_raw[i], cv2.COLOR_BGR2RGB)
                    if rois_raw[i].ndim == 3
                    else rois_raw[i]
                ),
                cmap="gray",
            )
        else:
            ax_in.imshow(rois_28[i], cmap="gray", vmin=0, vmax=1)
        ax_in.set_title(f"Вход #{i + 1}", fontsize=8)

        shown = (
            rois_28[i][::-1, ::-1]
            if (flipped is not None and flipped[i])
            else rois_28[i]
        )
        ax_out.imshow(shown, cmap="gray", vmin=0, vmax=1)

        title = f"Предсказано: {int(preds[i])}\n{probs[i] * 100:.1f}%"
        color = "black"
        if true_labels is not None:
            ok = int(preds[i]) == int(true_labels[i])
            color = "green" if ok else "red"
            title += f"\n(истина: {int(true_labels[i])})"
        if flipped is not None and flipped[i]:
            title += "\n[повёрнуто 180°]"
        ax_out.set_title(title, fontsize=8, color=color)

    if true_labels is not None:
        acc = np.mean(np.asarray(preds) == np.asarray(true_labels))
        fig.suptitle(f"Accuracy на ROI: {acc * 100:.1f}% ({n} шт.)", fontsize=11)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Визуализация: {save_path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def finetune_last_layer(model, per_image, epochs=40):
    labeled = [d for d in per_image if d["labels"] is not None and len(d["x28"]) > 0]
    if len(labeled) < 2:
        print("  [!] Для fine-tuning нужны метки минимум для 2 снимков - пропуск")
        return model

    test_img, train_imgs = labeled[-1], labeled[:-1]
    x_tr = np.concatenate([d["x28"] for d in train_imgs])[..., None]
    y_tr = np.concatenate([d["labels"] for d in train_imgs])
    x_te = test_img["x28"][..., None]
    y_te = test_img["labels"]

    acc_before = np.mean(model.predict(x_te, verbose=0).argmax(1) == y_te)

    for layer in model.layers[:-1]:
        layer.trainable = False
    _compile(model, lr=1e-3)
    model.fit(x_tr, y_tr, epochs=epochs, batch_size=8, verbose=0)

    acc_after = np.mean(model.predict(x_te, verbose=0).argmax(1) == y_te)
    print(
        f"  Fine-tuning (тест = снимок {test_img['stem']}): "
        f"{acc_before * 100:.1f}% -> {acc_after * 100:.1f}%"
    )
    save_model(model, FINETUNED_PATH)
    return model


def main():
    files = []
    for ext in SUPPORTED_EXT:
        files.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext}")))
        files.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext.upper()}")))
    files = sorted(set(files))
    if not files:
        print(f"[!] В папке {INPUT_DIR} нет изображений")
        return

    model, _ = load_or_train_model()

    per_image = collect_dataset(files)
    all_x = [d["x28"] for d in per_image if len(d["x28"])]
    if not all_x:
        print("[!] Не получено ни одного ROI")
        return
    total = sum(len(x) for x in all_x)
    print(f"\nВсего ROI собственных цифр: {total}")
    if total < 50:
        print(
            "  [i] Меньше 50 образцов - оценка accuracy на них ориентировочная. "
            "Для честной оценки нужно больше данных; можно дообучить последний слой "
            "(FINE_TUNE=True), разделяя train/test по исходным снимкам."
        )

    flat_imgs, flat_titles = [], []
    for d in per_image:
        for i, img in enumerate(d["x28"]):
            flat_imgs.append((img * 255).astype(np.uint8))
            lab = f"{d['labels'][i]}" if d["labels"] is not None else f"#{i + 1}"
            flat_titles.append(f"{d['stem'][-4:]}: {lab}")
    _save_grid(
        flat_imgs[:25],
        flat_titles[:25],
        os.path.join(OUTPUT_DIR, "own_rois_5x5.png"),
        cols=5,
    )

    if FINE_TUNE:
        model = finetune_last_layer(model, per_image)

    all_pred, all_true = [], []
    for d in per_image:
        if len(d["x28"]) == 0:
            continue
        preds, probs, used, flipped = predict_digits(model, d["x28"])

        print(f"\n{d['stem']}:")
        for i, (p, pr) in enumerate(zip(preds, probs), 1):
            extra = ""
            if d["labels"] is not None:
                extra = (
                    "  +"
                    if p == d["labels"][i - 1]
                    else f"  - (истина {d['labels'][i - 1]})"
                )
            print(
                f"    #{i}: Предсказанная цифра: {p}  ({pr * 100:.1f}%)"
                f"{'  [повёрнуто 180°]' if flipped[i - 1] else ''}{extra}"
            )

        out_dir = os.path.join(OUTPUT_DIR, d["stem"])
        os.makedirs(out_dir, exist_ok=True)
        show_predictions(
            d["x28"],
            preds,
            probs,
            true_labels=d["labels"],
            save_path=os.path.join(out_dir, "lr4_predictions.png"),
            rois_raw=d["raw"],
            flipped=flipped,
        )

        if d["labels"] is not None:
            all_pred.extend(preds.tolist())
            all_true.extend(d["labels"].tolist())

    if all_true:
        acc = np.mean(np.array(all_pred) == np.array(all_true))
        print(
            f"\nAccuracy на собственных ROI: {acc * 100:.1f}% "
            f"({sum(p == t for p, t in zip(all_pred, all_true))}/{len(all_true)})"
        )
    else:
        print(
            "\nИстинные метки не заданы (TRUE_LABELS) - accuracy на своих ROI не считается."
        )

    print(f"\nГотово: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
