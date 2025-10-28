import json
import socket
import threading
from queue import Queue
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
        return points, False

    custom_points = [(-0.5, 0.0), (0.5, 0.0), (0.5, 1.0), (-0.5, 1.0)]
    print(f"✅ Використовую стандартну зону 1×1 м: {custom_points}")
    return custom_points, True


class TouchEventServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.server_socket.settimeout(1.0)
        self.client = None
        self.client_lock = threading.Lock()
        self.queue: Queue = Queue()
        self.running = True
        self.accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self.accept_thread.start()
        self.sender_thread.start()
        print(f"📡 TouchEventServer listening on {self.host}:{self.port}")

    def _accept_loop(self):
        while self.running:
            try:
                conn, addr = self.server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with self.client_lock:
                if self.client:
                    try:
                        self.client.close()
                    except OSError:
                        pass
                self.client = conn
                self.client.settimeout(2.0)
            print(f"🔌 Client connected from {addr}")

    def _sender_loop(self):
        while self.running:
            event = self.queue.get()
            if event is None:
                break
            payload = json.dumps(event) + "\n"
            with self.client_lock:
                client = self.client
            if not client:
                continue
            try:
                client.sendall(payload.encode("utf-8"))
            except OSError:
                print("⚠️ Client disconnected")
                with self.client_lock:
                    try:
                        if self.client:
                            self.client.close()
                    except OSError:
                        pass
                    self.client = None

    def send_event(self, event: dict):
        if self.running:
            self.queue.put(event)

    def shutdown(self):
        self.running = False
        try:
            self.server_socket.close()
        except OSError:
            pass
        self.queue.put(None)
        with self.client_lock:
            if self.client:
                try:
                    self.client.close()
                except OSError:
                    pass
                self.client = None
        print("🛑 TouchEventServer stopped")


zone_points, is_custom_zone = load_zone_points()
zone_path = Path(zone_points)

# --- Параметри виявлення
TOUCH_THRESHOLD = 0.15   # м — зміна відстані для "дотику"
MIN_POINTS = 5           # мінімальна кількість точок
SMOOTHING = 0.3          # оновлення фону
ANGLE_MIN = -90
ANGLE_MAX = 90
ACTIVATION_FRAMES = 2    # кількість послідовних кадрів для підтвердження появи
DEACTIVATION_FRAMES = 3  # кількість порожніх кадрів для завершення події
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 9100

# --- Підключення до лідару
laser = HokuyoLX(addr=('192.168.0.10', 10940))
event_server = TouchEventServer(SERVER_HOST, SERVER_PORT)

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
zone_forward = [pt[0] for pt in zone_points]
zone_lateral = [pt[1] for pt in zone_points]
margin_forward = 0.0 if is_custom_zone else 0.5
margin_lateral = 0.0 if is_custom_zone else 0.5
x_min, x_max = min(zone_lateral) - margin_lateral, max(zone_lateral) + margin_lateral
y_min, y_max = min(zone_forward) - margin_forward, max(zone_forward) + margin_forward
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel('Y (м) — ліво/право')
ax.set_ylabel('X (м) — вперед')
ax.set_title('Hokuyo — трекінг у вибраній зоні')
x_left, x_right = ax.get_xlim()
y_bottom, y_top = ax.get_ylim()
ax.text(x_left, y_bottom, " Ліво", color='gray', ha='left', va='bottom')
ax.text(x_right, y_bottom, " Право", color='gray', ha='right', va='bottom')
ax.text((x_left + x_right) / 2.0, y_top, " Вперед", color='gray', ha='center', va='top')
ax.text((x_left + x_right) / 2.0, y_bottom, " Ближче", color='gray', ha='center', va='bottom')

# малюємо полігон зони
verts = zone_points + [zone_points[0]]
plot_x = [pt[1] for pt in verts]
plot_y = [pt[0] for pt in verts]
ax.plot(plot_x, plot_y, c='red', lw=2)

# --- Стан події
is_touch_active = False
last_touch_coords = None
touch_frames = 0
missing_frames = 0

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
    sc.set_offsets(np.c_[y_in, x_in])
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
        last_touch_coords = (x_touch, y_touch)
        touch_frames += 1
        missing_frames = 0
        if not is_touch_active and touch_frames >= ACTIVATION_FRAMES:
            is_touch_active = True
            print(f"👉 Touch started at ({x_touch:.2f}, {y_touch:.2f}) м — {touch_points} точок")
            event_server.send_event(
                {
                    "event": "touch_start",
                    "x": float(x_touch),
                    "y": float(y_touch),
                    "points": int(touch_points),
                    "timestamp": time.time(),
                }
            )
    else:
        touch_frames = 0
        missing_frames += 1
        if is_touch_active and missing_frames >= DEACTIVATION_FRAMES:
            is_touch_active = False
            if last_touch_coords:
                print(f"👋 Touch ended near ({last_touch_coords[0]:.2f}, {last_touch_coords[1]:.2f}) м")
            else:
                print("👋 Touch ended")
            event_server.send_event(
                {
                    "event": "touch_end",
                    "x": float(last_touch_coords[0]) if last_touch_coords else None,
                    "y": float(last_touch_coords[1]) if last_touch_coords else None,
                    "timestamp": time.time(),
                }
            )
            last_touch_coords = None
            missing_frames = 0

    # оновлення фону
    base_dist = (1 - SMOOTHING) * base_dist + SMOOTHING * dist_m
    time.sleep(0.05)

event_server.shutdown()
