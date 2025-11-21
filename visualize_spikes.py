import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ubh_reader import compute_angles, iter_ubh_frames, read_ubh_header


def detect_spikes(
    path: Path,
    base_frames: int = 60,
    threshold_m: float = 0.07,
    min_points: int = 15,
) -> Tuple[List[dict], np.ndarray]:
    header = read_ubh_header(path)
    angles = compute_angles(header)
    frames = list(iter_ubh_frames(path))
    if not frames:
        raise RuntimeError("Немає кадрів у файлі")
    ranges = np.stack([frame.ranges_mm for frame in frames]) / 1000.0
    base = np.mean(ranges[:base_frames], axis=0)
    events: List[dict] = []
    collecting = False
    buffer_coords: List[Tuple[float, float]] = []
    start_idx = 0
    for idx, frame in enumerate(frames):
        distances = ranges[idx]
        diff = base - distances
        mask = diff >= threshold_m
        active = np.where(mask)[0]
        coords = np.column_stack(
            (distances[active] * np.cos(angles[active]), distances[active] * np.sin(angles[active]))
        )
        if active.size >= min_points:
            if not collecting:
                collecting = True
                buffer_coords = []
                start_idx = idx
            buffer_coords.extend(coords.tolist())
        else:
            if collecting:
                collecting = False
                events.append(
                    {
                        "start_idx": start_idx,
                        "end_idx": idx - 1,
                        "start_time": frames[start_idx].logtime,
                        "end_time": frames[idx - 1].logtime,
                        "coords": np.array(buffer_coords),
                    }
                )
    if collecting and buffer_coords:
        events.append(
            {
                "start_idx": start_idx,
                "end_idx": len(frames) - 1,
                "start_time": frames[start_idx].logtime,
                "end_time": frames[-1].logtime,
                "coords": np.array(buffer_coords),
            }
        )
    return events, angles


def save_event_plots(events: List[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, event in enumerate(events, 1):
        coords = event["coords"]
        if coords.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(coords[:, 1], coords[:, 0], s=10, c="tab:blue", alpha=0.7)
        ax.set_xlabel("Y (м)")
        ax.set_ylabel("X (м)")
        ax.set_title(
            f"Сплеск #{idx}\n{event['start_time']} — {event['end_time']}\nточок: {coords.shape[0]}"
        )
        ax.set_aspect("equal", "box")
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"spike_{idx:02d}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Візуалізація сплесків з UBH")
    parser.add_argument("file", type=Path)
    parser.add_argument("--out", type=Path, default=Path("spike_plots"))
    args = parser.parse_args()
    events, _ = detect_spikes(args.file)
    print(f"Знайдено {len(events)} сплесків")
    save_event_plots(events, args.out)


if __name__ == "__main__":
    main()
