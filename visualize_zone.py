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

# --- Підготовка графіка
plt.ion()
fig, ax = plt.subplots(figsize=(6, 3))
sc = ax.scatter([], [], s=5, c='cyan')
ax.set_xlim(ZONE_X_MIN, ZONE_X_MAX)
ax.set_ylim(ZONE_Y_MIN, ZONE_Y_MAX)
ax.set_xlabel('X (м)')
ax.set_ylabel('Y (м)')
ax.set_title('Hokuyo UST-10LX — зона 4×2 м')

# --- Зчитування у циклі
while True:
    timestamp, dist_mm = laser.get_dist()
    dist_m = np.array(dist_mm) / 1000.0  # мм → м

    # кути для 270°
    angles = np.linspace(-135, 135, len(dist_m)) * np.pi / 180.0
    x = dist_m * np.cos(angles)
    y = dist_m * np.sin(angles)

    # лише точки у зоні 4×2 м
    mask = (x > ZONE_X_MIN) & (x < ZONE_X_MAX) & (y > ZONE_Y_MIN) & (y < ZONE_Y_MAX)
    x_zone, y_zone = x[mask], y[mask]

    # оновлення графіка
    sc.set_offsets(np.c_[x_zone, y_zone])
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.05)