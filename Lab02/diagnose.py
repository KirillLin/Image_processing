"""
Диагностика: показывает распределение H, S, V для пикселей на фото.
Используется для подбора HSV-диапазонов.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = r"input\1695128011374.jpg"   # ← путь к фото

image = cv2.imread(IMAGE_PATH)
image = cv2.resize(image, None, fx=0.5, fy=0.5)  # уменьшить для скорости
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

# Отбираем только «цветные» пиксели — с насыщенностью выше порога
mask_colored = (s > 30) & (v > 30)
h_colored = h[mask_colored]
s_colored = s[mask_colored]
v_colored = v[mask_colored]

print(f"Всего пикселей: {h.size}")
print(f"Цветных пикселей: {h_colored.size}")
print(f"Процент цветных: {100 * h_colored.size / h.size:.1f}%")

print(f"\nГистограмма H (по цветным пикселям):")
hist, _ = np.histogram(h_colored, bins=36, range=(0, 180))
for i, count in enumerate(hist):
    if count > 0:
        h_lo = i * 5
        h_hi = h_lo + 5
        bar = "█" * int(count / hist.max() * 40)
        print(f"  H {h_lo:3d}-{h_hi:3d}: {count:6d} {bar}")

# Также можно визуально посмотреть: цветные пиксели
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Оригинал")
axes[0].axis("off")

axes[1].imshow(h, cmap="hsv", vmin=0, vmax=180)
axes[1].set_title("Канал H")
axes[1].axis("off")

axes[2].imshow(s, cmap="gray")
axes[2].set_title("Канал S")
axes[2].axis("off")

axes[3].imshow(v, cmap="gray")
axes[3].set_title("Канал V")
axes[3].axis("off")

plt.tight_layout()
plt.show()