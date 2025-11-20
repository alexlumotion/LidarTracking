import time
from dataclasses import dataclass
from pathlib import Path as FSPath
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path

from event_server import TouchEventServer
from lidar_io import LASER_RECONNECT_DELAY, fetch_scan, reset_laser
from ubh_reader import iter_ubh_frames

try:
    from sklearn.cluster import DBSCAN as SklearnDBSCAN
except ImportError:
    SklearnDBSCAN = None


@dataclass
class TrackedCluster:
    centroid: Tuple[float, float]
    points: int
    touch_frames: int = 0
    missing_frames: int = 0
    is_active: bool = False
    last_detection_time: float = 0.0
    last_touch_coords: Optional[Tuple[float, float]] = None
    updated: bool = False


FLIP_Y = True  # тестове віддзеркалення ліво/право
USE_RAW_POINTS = False  # для тесту використовуємо кластеризацію
ENABLE_ZONE_FILTER = True  # вимкни, щоб ігнорувати полігон зони
ENABLE_THRESHOLD_FILTER = True  # використати відхилення від бази
RAW_POINT_EVENT = "touch_end"  # який тип події відправляти у raw-режимі
DEBUG_LOGS = True  # встанови False, щоб вимкнути діагностику
DETECTION_PROFILE = "ball"  # режими: "touch" | "ball"

DETECTION_PRESETS = {
    "touch": {
        "threshold": 0.15,
        "min_points": 5,
        "smoothing": 0.3,
        "activation_frames": 2,
        "deactivation_frames": 3,
        "debounce": 0.1,
        "cluster_eps": 0.07,
        "cluster_match": 0.15,
    },
    "ball": {
        "threshold": 0.07,
        "min_points": 2,
        "smoothing": 0.05,
        "activation_frames": 1,
        "deactivation_frames": 1,
        "debounce": 0.4,
        "cluster_eps": 0.06,
        "cluster_match": 0.12,
    },
}

LOOP_SLEEP_SECONDS = 0.02
#REPLAY_UBH_FILE: Optional[str] = "2025_11_19_13_03_37_675.ubh"  # шлях до .ubh для офлайнового тесту
# Set to None to read live scans instead of playback file
REPLAY_UBH_FILE: Optional[str] = None  # шлях до .ubh для офлайнового тесту

REPLAY_LOOP = False  # якщо True, після кінця файлу починаємо спочатку
MIN_POINTS_FOR_COUNT = 5  # мінімум променів, щоб зарахувати дотик


def _fallback_dbscan(coords: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Проста реалізація DBSCAN, якщо sklearn недоступний."""
    n_points = coords.shape[0]
    labels = -np.ones(n_points, dtype=int)
    cluster_id = 0
    visited = np.zeros(n_points, dtype=bool)
    eps_sq = eps * eps
    neighbors_cache: Dict[int, np.ndarray] = {}

    def region_query(idx: int) -> np.ndarray:
        if idx in neighbors_cache:
            return neighbors_cache[idx]
        deltas = coords - coords[idx]
        dist_sq = np.einsum("ij,ij->i", deltas, deltas)
        neighbors = np.where(dist_sq <= eps_sq)[0]
        neighbors_cache[idx] = neighbors
        return neighbors

    for point_idx in range(n_points):
        if visited[point_idx]:
            continue
        visited[point_idx] = True
        neighbors = region_query(point_idx)
        if neighbors.size < min_samples:
            continue
        labels[point_idx] = cluster_id
        seeds = set(neighbors.tolist())
        while seeds:
            current = seeds.pop()
            if not visited[current]:
                visited[current] = True
                current_neighbors = region_query(current)
                if current_neighbors.size >= min_samples:
                    seeds.update(current_neighbors.tolist())
            if labels[current] == -1:
                labels[current] = cluster_id
        cluster_id += 1
    return labels


def cluster_active_points(x_vals: np.ndarray, y_vals: np.ndarray, eps: float, min_samples: int) -> List[np.ndarray]:
    if x_vals.size == 0:
        return []
    coords = np.column_stack((x_vals, y_vals))
    if SklearnDBSCAN is not None:
        model = SklearnDBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(coords)
    else:
        labels = _fallback_dbscan(coords, eps, min_samples)
    unique_labels = [label for label in set(labels) if label != -1]
    clusters = [np.where(labels == label)[0] for label in unique_labels]
    return clusters


def _make_replay_fetcher(path: FSPath) -> Callable[[], Tuple[int, List[float]]]:
    frames = iter_ubh_frames(path)

    def fetch() -> Tuple[int, List[float]]:
        nonlocal frames
        try:
            frame = next(frames)
        except StopIteration:
            if not REPLAY_LOOP:
                raise RuntimeError("UBH replay завершився")
            frames = iter_ubh_frames(path)
            frame = next(frames)
        return frame.timestamp, frame.ranges_mm.astype(float).tolist()

    return fetch


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
    cluster_eps = detector_cfg["cluster_eps"]
    cluster_match = detector_cfg["cluster_match"]

    zone_path = Path(zone_points)

    replay_fetch: Optional[Callable[[], Tuple[int, List[float]]]] = None
    detected_touch_count = 0
    last_event_time = 0.0
    last_event_coords: Optional[Tuple[float, float]] = None
    if REPLAY_UBH_FILE:
        replay_fetch = _make_replay_fetcher(FSPath(REPLAY_UBH_FILE))
        print(f"🔁 Використовую запис з файла {REPLAY_UBH_FILE}")

    def next_scan() -> Tuple[int, List[float]]:
        if replay_fetch is not None:
            return replay_fetch()
        return fetch_scan()

    try:
        print("⏳ Калібрую фон...")
        time.sleep(1)
        try:
            _, base_dist = next_scan()
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

        tracked_clusters: Dict[int, TrackedCluster] = {}
        next_cluster_id = 1

        while plt.fignum_exists(fig.number):
            try:
                _, dist_mm = next_scan()
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

            if ENABLE_ZONE_FILTER:
                if mode == "sector":
                    limit = radius_limit if radius_limit is not None else 1.0
                    inside_mask = dist_m <= limit
                else:
                    inside_mask = zone_path.contains_points(np.c_[x, y])
            else:
                inside_mask = np.ones_like(dist_m, dtype=bool)
            x_in, y_in = x[inside_mask], y[inside_mask]

            sc.set_offsets(np.c_[y_in, x_in])
            fig.canvas.draw()
            fig.canvas.flush_events()

            diff = base_dist - dist_m
            if ENABLE_THRESHOLD_FILTER:
                signal_mask = diff >= touch_threshold
            else:
                signal_mask = np.ones_like(diff, dtype=bool)
            active_idx = np.where(signal_mask & inside_mask)[0]
            touch_points = int(active_idx.size)
            total_active_points = touch_points
            now = time.time()

            if DEBUG_LOGS and touch_points > 0:
                diff_min = float(np.min(diff)) if diff.size else 0.0
                diff_max = float(np.max(diff)) if diff.size else 0.0
                print(
                    f"[debug] touch_points={touch_points} diff_min={diff_min:.3f} diff_max={diff_max:.3f}"
                )
                coords_sample = list(zip(x[active_idx], y[active_idx]))
                if len(coords_sample) > 5:
                    coords_sample = coords_sample[:5]
                print(f"[debug] active_coords_sample={coords_sample}")

            for cluster in tracked_clusters.values():
                cluster.updated = False

            if USE_RAW_POINTS:
                if touch_points > 0:
                    for x_touch, y_touch in zip(x[active_idx], y[active_idx]):
                        event_server.send_event(
                            {
                                "event": RAW_POINT_EVENT,
                                "x": float(x_touch),
                                "y": float(y_touch),
                                "points": 1,
                                "timestamp": now,
                            }
                        )
                if total_active_points == 0:
                    base_dist = (1 - smoothing) * base_dist + smoothing * dist_m
                time.sleep(LOOP_SLEEP_SECONDS)
                continue

            detected_clusters = []
            if touch_points >= min_points:
                cluster_indices = cluster_active_points(x[active_idx], y[active_idx], cluster_eps, min_points)
                for local_indices in cluster_indices:
                    actual_indices = active_idx[local_indices]
                    x_cluster = x[actual_indices]
                    y_cluster = y[actual_indices]
                    centroid = (float(np.mean(x_cluster)), float(np.mean(y_cluster)))
                    detected_clusters.append(
                        {
                            "indices": actual_indices,
                            "centroid": centroid,
                            "points": int(actual_indices.size),
                        }
                    )

            for detection in detected_clusters:
                centroid = detection["centroid"]
                assigned_id = None
                best_distance = cluster_match
                for cluster_id, cluster_state in tracked_clusters.items():
                    if cluster_state.updated:
                        continue
                    dist = np.hypot(
                        centroid[0] - cluster_state.centroid[0],
                        centroid[1] - cluster_state.centroid[1],
                    )
                    if dist <= best_distance:
                        best_distance = dist
                        assigned_id = cluster_id

                if assigned_id is None:
                    assigned_id = next_cluster_id
                    next_cluster_id += 1
                    tracked_clusters[assigned_id] = TrackedCluster(
                        centroid=centroid,
                        points=detection["points"],
                        last_touch_coords=centroid,
                    )
                cluster_state = tracked_clusters[assigned_id]
                cluster_state.centroid = centroid
                cluster_state.points = detection["points"]
                cluster_state.last_touch_coords = centroid
                cluster_state.touch_frames += 1
                cluster_state.missing_frames = 0
                cluster_state.updated = True
                if DEBUG_LOGS:
                    print(
                        f"[debug] cluster_id={assigned_id} frames={cluster_state.touch_frames} centroid={cluster_state.centroid}"
                    )

                if not cluster_state.is_active:
                    cooldown_passed = (now - cluster_state.last_detection_time) >= debounce_seconds
                    if cluster_state.touch_frames >= activation_frames and cooldown_passed:
                        same_event = False
                        if last_event_coords is not None:
                            dt = now - last_event_time
                            dx = cluster_state.centroid[0] - last_event_coords[0]
                            dy = cluster_state.centroid[1] - last_event_coords[1]
                            same_event = dt <= 0.1 and (dx * dx + dy * dy) ** 0.5 <= 0.15
                        cluster_state.is_active = True
                        countable = cluster_state.points >= MIN_POINTS_FOR_COUNT
                        if not same_event and countable:
                            detected_touch_count += 1
                            last_event_time = now
                            last_event_coords = cluster_state.centroid
                        cluster_state.last_detection_time = now
                        x_touch, y_touch = cluster_state.centroid
                        if same_event:
                            print(
                                f"[debug] merged into #{detected_touch_count} at ({x_touch:.2f}, {y_touch:.2f}) м — {cluster_state.points} променів"
                            )
                        elif countable:
                            print(
                                f"🎾 Ball detected #{detected_touch_count} at ({x_touch:.2f}, {y_touch:.2f}) м — {cluster_state.points} променів"
                            )
                        else:
                            print(
                                f"[debug] ignored cluster with {cluster_state.points} points (<{MIN_POINTS_FOR_COUNT}) at ({x_touch:.2f}, {y_touch:.2f})"
                            )
                        event_server.send_event(
                            {
                                "event": "touch_start",
                                "x": x_touch,
                                "y": y_touch,
                                "points": cluster_state.points,
                                "timestamp": now,
                            }
                        )

            clusters_to_remove = []
            for cluster_id, cluster_state in tracked_clusters.items():
                if cluster_state.updated:
                    continue
                cluster_state.touch_frames = 0
                cluster_state.missing_frames += 1
                if cluster_state.is_active and cluster_state.missing_frames >= deactivation_frames:
                    cluster_state.is_active = False
                    coords = cluster_state.last_touch_coords
                    if coords:
                        print(f"✅ Ball cleared near ({coords[0]:.2f}, {coords[1]:.2f}) м")
                    event_server.send_event(
                        {
                            "event": "touch_end",
                            "x": float(coords[0]) if coords else None,
                            "y": float(coords[1]) if coords else None,
                            "timestamp": now,
                        }
                    )
                    clusters_to_remove.append(cluster_id)
                elif cluster_state.missing_frames >= deactivation_frames:
                    clusters_to_remove.append(cluster_id)

            for cluster_id in clusters_to_remove:
                tracked_clusters.pop(cluster_id, None)

            if total_active_points == 0:
                base_dist = (1 - smoothing) * base_dist + smoothing * dist_m
            time.sleep(LOOP_SLEEP_SECONDS)
    finally:
        event_server.shutdown()
        reset_laser()
        if detected_touch_count:
            print(f"ℹ️ Total detections: {detected_touch_count}")
