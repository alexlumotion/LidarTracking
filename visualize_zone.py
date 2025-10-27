import json
import numpy as np
import matplotlib.pyplot as plt
from hokuyolx import HokuyoLX
from matplotlib.path import Path
import time

# --- Завантаження зони з файлу
with open("zone_config.json", "r", encoding="utf-8") as f:
    zone_points = json.load(f)["zone"]
zone_path = Path(zone_points)
print(f"✅ Завантажено зону з 4 точок: {zone_points}")

# --- Підключення до лідару
laser = HokuyoLX(addr=('192.168.0.10', 10940))

# --- Параметри виявлення
TOUCH_THRESHOLD = 0.15   # м — зміна відстані для "дотику"
MIN_POINTS = 5           # мінімальна кількість точок
SMOOTHING = 0.3          # оновлення фону

# --- Початковий фон
print("⏳ Калібрую фон...")
time.sleep(1)
_, base_dist = laser.get_dist()
base_dist = np.array(base_dist, dtype=float) / 1000.0
print("✅ Калібрування завершено")

# --- Підготовка графіка
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter([], [], s=5, c='cyan')

# межі графіка за даними полігона
zone_x, zone_y = zip(*zone_points)
ax.set_xlim(min(zone_x) - 0.5, max(zone_x) + 0.5)
ax.set_ylim(min(zone_y) - 0.5, max(zone_y) + 0.5)
ax.set_xlabel('X (м)')
ax.set_ylabel('Y (м)')
ax.set_title('Hokuyo — трекінг у вибраній зоні')

# малюємо полігон зони
verts = zone_points + [zone_points[0]]
ax.plot(*zip(*verts), c='red', lw=2)

# --- Основний цикл
while plt.fignum_exists(fig.number):
    timestamp, dist_mm = laser.get_dist()
    dist_m = np.array(dist_mm, dtype=float) / 1000.0
    dist_m = np.nan_to_num(dist_m, nan=base_dist)

    angles = np.linspace(-135, 135, len(dist_m)) * np.pi / 180.0
    x = dist_m * np.cos(angles)
    y = dist_m * np.sin(angles)

    # фільтруємо лише точки всередині полігона
    inside_mask = zone_path.contains_points(np.c_[x, y])
    x_in, y_in = x[inside_mask], y[inside_mask]

    # оновлення графіка
    sc.set_offsets(np.c_[x_in, y_in])
    fig.canvas.draw()
    fig.canvas.flush_events()

    # виявлення дотиків
    diff = base_dist - dist_m
    touch_mask = (diff > TOUCH_THRESHOLD)
    touch_points = np.sum(touch_mask & inside_mask)

    if touch_points > MIN_POINTS:
        idx = np.where(touch_mask & inside_mask)[0]
        x_touch = np.mean(x[idx])
        y_touch = np.mean(y[idx])
        print(f"👉 Touch detected at ({x_touch:.2f}, {y_touch:.2f}) м — {touch_points} точок")

    # оновлення фону
    base_dist = (1 - SMOOTHING) * base_dist + SMOOTHING * dist_m
    time.sleep(0.05)