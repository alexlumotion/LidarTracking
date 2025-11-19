#python play_ubh.py 2025_11_19_13_03_37_675.ubh --speed 1.0

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ubh_reader import compute_angles, iter_ubh_frames, read_ubh_header


def run_player(path: Path, speed: float) -> None:
    header = read_ubh_header(path)
    angles = compute_angles(header)
    scan_interval = float(header.get("scanMsec", "25")) / 1000.0
    frame_delay = scan_interval / max(speed, 1e-3)

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter([], [])
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"UBH playback: {path.name}")
    limit_x = (0.2, 3.2)
    limit_y = (-1.8, 1.8)
    ax.set_xlim(limit_x)
    ax.set_ylim(limit_y)
    ax.set_xticks(np.arange(limit_x[0], limit_x[1] + 0.2, 0.2))
    ax.set_yticks(np.arange(limit_y[0], limit_y[1] + 0.2, 0.2))
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    timestamp_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    for frame in iter_ubh_frames(path):
        distances_m = frame.ranges_mm / 1000.0
        valid = distances_m > 0
        x = distances_m[valid] * np.cos(angles[valid])
        y = distances_m[valid] * np.sin(angles[valid])
        within = (
            (x >= limit_x[0])
            & (x <= limit_x[1])
            & (y >= limit_y[0])
            & (y <= limit_y[1])
        )
        scatter.set_offsets(np.column_stack((x[within], y[within])))
        timestamp_text.set_text(f"{frame.logtime} / {frame.timestamp}")
        fig.canvas.draw_idle()
        plt.pause(0.001)
        time.sleep(frame_delay)

    plt.ioff()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Play back UBH lidar recordings.")
    parser.add_argument("file", type=Path, help="Path to .ubh capture")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0)",
    )
    args = parser.parse_args()
    run_player(args.file, args.speed)


if __name__ == "__main__":
    main()
