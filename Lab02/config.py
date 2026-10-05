"""
Настройки и константы для ЛР №2.
Все параметры собраны в одном месте — легко менять.
"""

import os
import numpy as np

# ======================== ПУТИ ========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ======================== ОБРАБОТКА ========================
MAX_SIDE = 500          # ресайз для скорости
SHOW_PLOTS = True       # показывать окна matplotlib

# ======================== HSV-ДИАПАЗОНЫ ========================
BLUE_RANGES = []
GREEN_RANGES = [
    (np.array([ 65, 150,  50]), np.array([ 90, 255, 255])),
]

# ======================== ФИЛЬТР ПО ПЛОЩАДИ ========================
MIN_AREA = 100       # минимальная площадь компоненты
MAX_AREA_RATIO = 0.05    # максимальная площадь — 5% кадра

# ======================== КЛАСТЕРИЗАЦИЯ ========================
N_CLASSES = 3
KMEANS_MAX_ITER = 100
KMEANS_SEED = 42

# ======================== ЦВЕТА КЛАССОВ (BGR) ========================
CLASS_COLORS = [
    (0, 0, 255),      # класс 0 — красный
    (255, 0, 0),      # класс 1 — синий
    (0, 255, 0),      # класс 2 — зелёный
]

# Создание папок
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)