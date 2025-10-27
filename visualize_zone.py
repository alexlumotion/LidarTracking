import numpy as np
import matplotlib.pyplot as plt
from hokuyolx import HokuyoLX
import time

# --- Підключення до лідару
laser = HokuyoLX(addr=('192.168.0.10', 10940))

# --- Параметри зони (у метрах)
ZONE_WIDTH = 4.0
ZONE_HEIGHT = 2.0
ZONE_X_MIN, ZONE_X_MAX = -ZONE_WIDTH / 2, ZONE_WIDTH / 2
ZONE_Y_MIN, ZONE_Y_MAX = 0.0, ZONE_HEIGHT

# --- Параметри виявлення
TOUCH_THRESHOLD = 0.15   # м — наскільки змінюється відстань, щоб вважати "дотиком"
MIN_POINTS = 5           # мінімальна кількість точок для підтвердження події
SMOOTHING = 0.3          # коефіцієнт згладжування фону

# --- Фонове сканування (початковий "базовий" стан)
print("⏳ Калібрую фон (прибери всі об’єкти з поля зору)...")
time.sleep(1)
_, base_dist = laser.get_dist()
base_dist = np.array(base_dist, dtype=float) / 1000.0
print("✅ Калібрування завершено — починаю моніторинг!")

# --- Підготовка графіка
plt.ion()
fig, ax = plt.subplots(figsize=(6, 3))
sc = ax.scatter([], [], s=6, c='cyan')
ax.set_xlim(ZONE_X_MIN, ZONE_X_MAX)
ax.set_ylim(ZONE_Y_MIN, ZONE_Y_MAX)
ax.set_xlabel('X (м)')
ax.set_ylabel('Y (м)')
ax.set_title('Hokuyo UST-10LX — зона 4×2 м (Touch Detection)')

# --- Основний цикл
while True:
    timestamp, dist_mm = laser.get_dist()
    dist_m = np.array(dist_mm, dtype=float) / 1000.0
    dist_m = np.nan_to_num(dist_m, nan=base_dist)  # заміна NaN на фон

    # кути для 270°
    angles = np.linspace(-135, 135, len(dist_m)) * np.pi / 180.0
    x = dist_m * np.cos(angles)
    y = dist_m * np.sin(angles)

    # Зона 4×2 м перед лідаром
    mask = (x > ZONE_X_MIN) & (x < ZONE_X_MAX) & (y > ZONE_Y_MIN) & (y < ZONE_Y_MAX)
    x_zone, y_zone = x[mask], y[mask]

    # Оновлення графіка
    sc.set_offsets(np.c_[x_zone, y_zone])
    fig.canvas.draw()
    fig.canvas.flush_events()

    # --- Виявлення дотиків
    diff = base_dist - dist_m
    touch_mask = (diff > TOUCH_THRESHOLD)
    touch_points = np.sum(touch_mask)

    if touch_points > MIN_POINTS:
        idx = np.where(touch_mask)[0]
        x_touch = np.mean(x[idx])
        y_touch = np.mean(y[idx])
        print(f"👉 Touch detected at ({x_touch:.2f}, {y_touch:.2f}) м — {touch_points} точок")

    # --- Поступово оновлюємо фон для стабільності
    base_dist = (1 - SMOOTHING) * base_dist + SMOOTHING * dist_m

    time.sleep(0.05)