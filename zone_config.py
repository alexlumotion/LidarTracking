import json
from pathlib import Path as SysPath

import numpy as np


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

