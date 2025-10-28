import json
import numpy as np
import matplotlib.pyplot as plt
from hokuyolx import HokuyoLX
from matplotlib.path import Path
import time
from pathlib import Path as SysPath


def load_zone_points():
    while True:
        choice = input(
            "Вибери зону: 0 — з файлу zone_config.json, 1 — стандартна зона 1×1 м: "
        ).strip()
        if choice in {"0", "1"}:
            break
        print("Введи 0 або 1.")

    if choice == "0":
        config_path = SysPath("zone_config.json")
        if not config_path.exists():
            raise FileNotFoundError("zone_config.json не знайдено. Спочатку виконай setup_zone.py.")
        with config_path.open("r", encoding="utf-8") as f:
            points = json.load(f)["zone"]
        print(f"✅ Завантажено зону з 4 точок: {points}")
        return points

    custom_points = [(-0.5, 0.0), (0.5, 0.0), (0.5, 1.0), (-0.5, 1.0)]
    print(f"✅ Використовую стандартну зону 1×1 м: {custom_points}")
    return custom_points


zone_points = load_zone_points()
zone_path = Path(zone_points)

# --- Підключення до лідару
laser = HokuyoLX(addr=('192.168.0.10', 10940))

# --- Параметри виявлення
TOUCH_THRESHOLD = 0.15   # м — зміна відстані для "дотику"
MIN_POINTS = 5           # мінімальна кількість точок
SMOOTHING = 0.3          # оновлення фону
ANGLE_MIN = -90
ANGLE_MAX = 90

# --- Початковий фон
print("⏳ Калібрую фон...")
time.sleep(1)
_, base_dist = laser.get_dist()
base_dist = np.array(base_dist, dtype=float) / 1000.0
angle_deg_full = np.linspace(-135, 135, len(base_dist))
sector_mask = (angle_deg_full >= ANGLE_MIN) & (angle_deg_full <= ANGLE_MAX)
angles = np.deg2rad(angle_deg_full[sector_mask])
base_dist = base_dist[sector_mask]
if np.isnan(base_dist).any():
    valid = base_dist[~np.isnan(base_dist)]
    fallback = valid.mean() if valid.size else 1.0
    base_dist = np.where(np.isnan(base_dist), fallback, base_dist)
print("✅ Калібрування завершено")

# --- Підготовка графіка
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter([], [], s=5, c='cyan')
ax.scatter(0, 0, c='orange', marker='x', s=80)
ax.text(0, 0, " Лідар", color='orange', fontsize=9, va='bottom')

# межі графіка за даними полігона
zone_x, zone_y = zip(*zone_points)
ax.set_xlim(min(zone_x) - 0.5, max(zone_x) + 0.5)
ax.set_ylim(min(zone_y) - 0.5, max(zone_y) + 0.5)
ax.invert_yaxis()
ax.set_xlabel('X (м)')
ax.set_ylabel('Y (м)')
ax.set_title('Hokuyo — трекінг у вибраній зоні')
ax.text(ax.get_xlim()[1], ax.get_ylim()[0], " Право", color='gray', ha='right', va='top')
ax.text(ax.get_xlim()[0], ax.get_ylim()[0], " Ліво", color='gray', ha='left', va='top')
ax.text(ax.get_xlim()[1], ax.get_ylim()[1], " Низ", color='gray', ha='right', va='bottom')
ax.text(ax.get_xlim()[0], ax.get_ylim()[1], " Верх", color='gray', ha='left', va='bottom')

# малюємо полігон зони
verts = zone_points + [zone_points[0]]
ax.plot(*zip(*verts), c='red', lw=2)

# --- Основний цикл
while plt.fignum_exists(fig.number):
    timestamp, dist_mm = laser.get_dist()
    dist_full = np.array(dist_mm, dtype=float) / 1000.0
    dist_m = dist_full[sector_mask]
    dist_m = np.where(~np.isfinite(dist_m), base_dist, dist_m)

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
