import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

MAX_SIDE = 500
SHOW_PLOTS = True

BLUE_RANGES = []
GREEN_RANGES = [
    (np.array([50, 100, 40]), np.array([95, 255, 255])),
]

MIN_AREA = 150
MAX_AREA_RATIO = 0.05

N_CLASSES = 3
KMEANS_MAX_ITER = 100
KMEANS_SEED = 42

FEATURE_KEYS = ("area", "compactness", "elongation", "holes")

CLASS_COLORS = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
]

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROUND_ELONGATION_THRESHOLD = 1.5
TARGET_ANGLE = np.pi / 2

# Проверка поворота на 180°: переворот принимается, только если уверенность
# сети выросла больше чем на этот порог (иначе остаётся исходная ориентация).
FLIP_MARGIN = 0.3

# Отсев ложных объектов (блики, тонкие линии): доля площади объекта в его
# минимальном повёрнутом прямоугольнике. У цифр она заметно выше.
MIN_FILL = 0.06

# Отказ от классификации: если уверенность сети ниже порога, объект считается
# "не цифрой" (например, блик). Слабейшая настоящая цифра на ваших фото ~0.77.
MIN_CONFIDENCE = 0.5
