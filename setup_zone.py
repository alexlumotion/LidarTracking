import json
import numpy as np
import matplotlib.pyplot as plt
from hokuyolx import HokuyoLX
from matplotlib.path import Path
import time

# --- Підключення до лідару
laser = HokuyoLX(addr=('192.168.0.10', 10940))

print("⏳ Відображаю повний діапазон (270°)...")
time.sleep(1)

plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter([], [], s=4, c='lightblue')
ax.set_xlim(-5, 5)
ax.set_ylim(0, 10)
ax.set_xlabel('X (м)')
ax.set_ylabel('Y (м)')
ax.set_title('Налаштування зони трекінгу (клікни 4 точки)')

# --- Змінні для вибору точок
points = []
polygon_patch = None

def onclick(event):
    global points, polygon_patch

    if event.inaxes != ax:
        return

    # додаємо клік
    points.append((event.xdata, event.ydata))
    ax.scatter(event.xdata, event.ydata, c='red', s=30)

    if len(points) == 4:
        # малюємо полігон
        verts = points + [points[0]]
        xs, ys = zip(*verts)
        ax.plot(xs, ys, c='red', lw=2)
        fig.canvas.mpl_disconnect(cid)
        plt.title("✅ Зона збережена у zone_config.json")

        # зберігаємо у файл
        with open("zone_config.json", "w", encoding="utf-8") as f:
            json.dump({"zone": points}, f, indent=2)
        print("✅ Зона збережена:", points)

# --- підписуємо подію кліку
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# --- оновлення графіка в реальному часі
while plt.fignum_exists(fig.number):
    timestamp, dist_mm = laser.get_dist()
    dist_m = np.array(dist_mm) / 1000.0
    angles = np.linspace(-135, 135, len(dist_m)) * np.pi / 180.0
    x = dist_m * np.cos(angles)
    y = dist_m * np.sin(angles)
    sc.set_offsets(np.c_[x, y])
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.1)