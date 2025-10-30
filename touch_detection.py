import time
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path

from event_server import TouchEventServer
from lidar_io import LASER_RECONNECT_DELAY, fetch_scan, reset_laser


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


def run_touch_detection(
    zone_points: Sequence[Tuple[float, float]],
    is_custom_zone: bool,
    mode: str,
    radius_limit: Optional[float],
    event_server: TouchEventServer,
):
    if DETECTION_PROFILE not in DETECTION_PRESETS:
        raise ValueError(f"Невідомий DETECTION_PROFILE: {DETECTION_PROFILE}")

    detector_cfg = DETECTION_PRESETS[DETECTION_PROFILE]
    touch_threshold = detector_cfg["threshold"]
    min_points = detector_cfg["min_points"]
    smoothing = detector_cfg["smoothing"]
    if mode == "sector":
        angle_min, angle_max = -135, 135
    else:
        angle_min, angle_max = (-80, 80) if is_custom_zone else (-90, 90)
    activation_frames = detector_cfg["activation_frames"]
    deactivation_frames = detector_cfg["deactivation_frames"]
    debounce_seconds = detector_cfg["debounce"]

    zone_path = Path(zone_points)

    try:
        print("⏳ Калібрую фон...")
        time.sleep(1)
        try:
            _, base_dist = fetch_scan()
        except RuntimeError as exc:
            raise SystemExit(f"❌ Критична помилка під час калібрування: {exc}")
        base_dist = np.array(base_dist, dtype=float) / 1000.0
        angle_deg_full = np.linspace(-135, 135, len(base_dist))
        sector_mask = (angle_deg_full >= angle_min) & (angle_deg_full <= angle_max)
        angles = np.deg2rad(angle_deg_full[sector_mask])
        base_dist = base_dist[sector_mask]
        if np.isnan(base_dist).any():
            valid = base_dist[~np.isnan(base_dist)]
            fallback = valid.mean() if valid.size else 1.0
            base_dist = np.where(np.isnan(base_dist), fallback, base_dist)
        print("✅ Калібрування завершено")

        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter([], [], s=5, c='cyan')
        ax.scatter(0, 0, c='orange', marker='x', s=80)
        ax.text(0, 0, " Лідар", color='orange', fontsize=9, va='bottom')

        zone_forward: List[float] = [pt[0] for pt in zone_points]
        zone_lateral: List[float] = [pt[1] for pt in zone_points]
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

        verts = list(zone_points) + [zone_points[0]]
        plot_x = [pt[1] for pt in verts]
        plot_y = [pt[0] for pt in verts]
        ax.plot(plot_x, plot_y, c='red', lw=2)

        is_touch_active = False
        last_touch_coords = None
        touch_frames = 0
        missing_frames = 0
        last_detection_time = 0.0

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

            if mode == "sector":
                limit = radius_limit if radius_limit is not None else 1.0
                inside_mask = dist_m <= limit
            else:
                inside_mask = zone_path.contains_points(np.c_[x, y])
            x_in, y_in = x[inside_mask], y[inside_mask]

            sc.set_offsets(np.c_[y_in, x_in])
            fig.canvas.draw()
            fig.canvas.flush_events()

            diff = base_dist - dist_m
            signal_mask = diff >= touch_threshold
            active_idx = np.where(signal_mask & inside_mask)[0]
            touch_points = active_idx.size
            now = time.time()
            cooldown_passed = (now - last_detection_time) >= debounce_seconds

            if touch_points >= min_points:
                x_touch = float(np.mean(x[active_idx]))
                y_touch = float(np.mean(y[active_idx]))
                last_touch_coords = (x_touch, y_touch)
                touch_frames += 1
                missing_frames = 0
                if not is_touch_active and touch_frames >= activation_frames and cooldown_passed:
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
                if is_touch_active and missing_frames >= deactivation_frames:
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

            if touch_points == 0:
                base_dist = (1 - smoothing) * base_dist + smoothing * dist_m
            time.sleep(0.05)
    finally:
        event_server.shutdown()
        reset_laser()
