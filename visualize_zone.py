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
            "Вибери зону: 0 — з файлу zone_config.json, 1 — стандартна зона 1×1 м, 2 — весь діапазон 270° (радіус 1 м): "
        ).strip()
        if choice in {"0", "1", "2"}:
            break
        print("Введи 0, 1 або 2.")

    if choice == "0":
        config_path = SysPath("zone_config.json")
        if not config_path.exists():
            raise FileNotFoundError("zone_config.json не знайдено. Спочатку виконай setup_zone.py.")
        with config_path.open("r", encoding="utf-8") as f:
            points = json.load(f)["zone"]
        print(f"✅ Завантажено зону з 4 точок: {points}")
        return {"points": points, "is_custom_zone": False, "mode": "polygon"}

    if choice == "1":
        custom_points = [
            (0.0, -0.5),
            (1.0, -0.5),
            (1.0, 0.5),
            (0.0, 0.5),
        ]
        print(f"✅ Використовую стандартну зону 1×1 м (X: 0–1 м, Y: -0.5–0.5 м): {custom_points}")
        return {"points": custom_points, "is_custom_zone": True, "mode": "polygon"}

    radius = 1.0
    arc_deg = np.linspace(-135, 135, 181)
    arc_points = [(radius * np.cos(np.deg2rad(deg)), radius * np.sin(np.deg2rad(deg))) for deg in arc_deg]
    sector_points = [(0.0, 0.0)] + arc_points + [(0.0, 0.0)]
    print(f"✅ Використовую повний діапазон лідару 270° з радіусом {radius} м")
    return {"points": sector_points, "is_custom_zone": True, "mode": "sector", "radius": radius}


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


FLIP_Y = True  # тестове віддзеркалення ліво/право
DETECTION_PROFILE = "ball"  # режими: "touch" | "ball"

DETECTION_PRESETS = {
    "touch": {
        "threshold": 0.15,
        "min_points": 5,
        "smoothing": 0.3,
        "activation_frames": 2,
        "deactivation_frames": 3,
        "debounce": 0.1,
    },
    "ball": {
        "threshold": 0.07,
        "min_points": 2,
        "smoothing": 0.05,
        "activation_frames": 1,
        "deactivation_frames": 1,
        "debounce": 0.4,
    },
}

LASER_MAX_RETRIES = 3
LASER_RECONNECT_DELAY = 0.5

zone_config = load_zone_points()
zone_points = zone_config["points"]
if FLIP_Y:
    zone_points = [(pt[0], -pt[1]) for pt in zone_points]
is_custom_zone = zone_config["is_custom_zone"]
mode = zone_config["mode"]
radius_limit = zone_config.get("radius")
zone_path = Path(zone_points)

# --- Параметри виявлення
if DETECTION_PROFILE not in DETECTION_PRESETS:
    raise ValueError(f"Невідомий DETECTION_PROFILE: {DETECTION_PROFILE}")

detector_cfg = DETECTION_PRESETS[DETECTION_PROFILE]
TOUCH_THRESHOLD = detector_cfg["threshold"]   # м — зміна відстані для спрацювання
MIN_POINTS = detector_cfg["min_points"]       # мінімальна кількість активних променів
SMOOTHING = detector_cfg["smoothing"]         # швидкість оновлення фону
if mode == "sector":
    ANGLE_MIN, ANGLE_MAX = -135, 135
else:
    ANGLE_MIN, ANGLE_MAX = (-80, 80) if is_custom_zone else (-90, 90)
ACTIVATION_FRAMES = detector_cfg["activation_frames"]
DEACTIVATION_FRAMES = detector_cfg["deactivation_frames"]
DEBOUNCE_SECONDS = detector_cfg["debounce"]
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 9100

# --- Підключення до лідару
def create_laser():
    return HokuyoLX(addr=('192.168.0.10', 10940))


laser = None


def reset_laser():
    global laser
    if laser is None:
        return
    closer = getattr(laser, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception as exc:
            print(f"⚠️ Помилка при закритті Hokuyo: {exc}")
    laser = None


def fetch_scan(max_retries: int = LASER_MAX_RETRIES):
    global laser
    last_exc = None
    for attempt in range(1, max_retries + 1):
        if laser is None:
            try:
                laser = create_laser()
                print("🔄 Hokuyo підключено")
            except Exception as exc:
                last_exc = exc
                print(f"⚠️ Не вдалося підключити Hokuyo ({attempt}/{max_retries}): {exc}")
                time.sleep(LASER_RECONNECT_DELAY * attempt)
                continue
        try:
            return laser.get_dist()
        except Exception as exc:
            last_exc = exc
            print(f"⚠️ Помилка читання Hokuyo ({attempt}/{max_retries}): {exc}")
            reset_laser()
            time.sleep(LASER_RECONNECT_DELAY * attempt)
    raise RuntimeError(f"Не вдалося отримати дані з Hokuyo: {last_exc}")


event_server = TouchEventServer(SERVER_HOST, SERVER_PORT)

# --- Початковий фон
print("⏳ Калібрую фон...")
time.sleep(1)
try:
    _, base_dist = fetch_scan()
except RuntimeError as exc:
    event_server.shutdown()
    raise SystemExit(f"❌ Критична помилка під час калібрування: {exc}")
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
if mode == "sector":
    margin_forward = margin_lateral = 0.2
else:
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
last_detection_time = 0.0

# --- Основний цикл
while plt.fignum_exists(fig.number):
    try:
        timestamp, dist_mm = fetch_scan()
    except RuntimeError as exc:
        print(f"❌ Неможливо отримати дані від Hokuyo: {exc}")
        time.sleep(LASER_RECONNECT_DELAY)
        continue

    dist_full = np.array(dist_mm, dtype=float) / 1000.0
    dist_m = dist_full[sector_mask]
    dist_m = np.where(~np.isfinite(dist_m), base_dist, dist_m)

    x = dist_m * np.cos(angles)
    y = dist_m * np.sin(angles)
    if FLIP_Y:
        y = -y

    # фільтруємо лише точки всередині активної зони
    if mode == "sector":
        limit = radius_limit if radius_limit is not None else 1.0
        inside_mask = dist_m <= limit
    else:
        inside_mask = zone_path.contains_points(np.c_[x, y])
    x_in, y_in = x[inside_mask], y[inside_mask]

    # оновлення графіка
    sc.set_offsets(np.c_[y_in, x_in])
    fig.canvas.draw()
    fig.canvas.flush_events()

    # виявлення дотиків / кидків
    diff = base_dist - dist_m
    signal_mask = diff >= TOUCH_THRESHOLD
    active_idx = np.where(signal_mask & inside_mask)[0]
    touch_points = active_idx.size
    now = time.time()
    cooldown_passed = (now - last_detection_time) >= DEBOUNCE_SECONDS

    if touch_points >= MIN_POINTS:
        x_touch = float(np.mean(x[active_idx]))
        y_touch = float(np.mean(y[active_idx]))
        last_touch_coords = (x_touch, y_touch)
        touch_frames += 1
        missing_frames = 0
        if not is_touch_active and touch_frames >= ACTIVATION_FRAMES and cooldown_passed:
            is_touch_active = True
            last_detection_time = now
            print(f"🎾 Ball detected at ({x_touch:.2f}, {y_touch:.2f}) м — {touch_points} променів")
            event_server.send_event(
                {
                    "event": "touch_start",
                    "x": x_touch,
                    "y": y_touch,
                    "points": int(touch_points),
                    "timestamp": now,
                }
            )
    else:
        touch_frames = 0
        missing_frames += 1
        if is_touch_active and missing_frames >= DEACTIVATION_FRAMES:
            is_touch_active = False
            if last_touch_coords:
                print(f"✅ Ball cleared near ({last_touch_coords[0]:.2f}, {last_touch_coords[1]:.2f}) м")
            event_server.send_event(
                {
                    "event": "touch_end",
                    "x": float(last_touch_coords[0]) if last_touch_coords else None,
                    "y": float(last_touch_coords[1]) if last_touch_coords else None,
                    "timestamp": now,
                }
            )
            last_touch_coords = None
            missing_frames = 0

    # оновлення фону — лише коли немає активних променів
    if touch_points == 0:
        base_dist = (1 - SMOOTHING) * base_dist + SMOOTHING * dist_m
    time.sleep(0.05)

event_server.shutdown()
