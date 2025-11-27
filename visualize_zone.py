from event_server import TouchEventServer
import touch_detection as td
from zone_config import load_zone_points


SERVER_HOST = "0.0.0.0"
SERVER_PORT = 9100


def main():
    # 1) Джерело даних: лайв чи запис
    while True:
        source_choice = input("Вибери джерело: 0 — лайв Hokuyo, 1 — запис (.ubh): ").strip()
        if source_choice in {"0", "1"}:
            break
        print("Введи 0 або 1.")

    if source_choice == "0":
        td.REPLAY_UBH_FILE = None
    else:
        default_file = td.REPLAY_UBH_FILE or ""
        path = input(f"Шлях до .ubh (enter, щоб взяти {default_file}): ").strip()
        td.REPLAY_UBH_FILE = path or default_file
        if not td.REPLAY_UBH_FILE:
            raise SystemExit("Не вказано файл .ubh для відтворення.")

    # 2) Зона трекінгу
    zone_config = load_zone_points()
    zone_points = zone_config["points"]
    if td.FLIP_Y:
        zone_points = [(pt[0], -pt[1]) for pt in zone_points]
    is_custom_zone = zone_config["is_custom_zone"]
    mode = zone_config["mode"]
    radius_limit = zone_config.get("radius")

    # 3) Алгоритм
    while True:
        algo_choice = input(
            "Вибери алгоритм: 0 — сплески (SPIKE), 1 — усі точки (raw), 2 — debug-spike (поріг 0.04 м, мін. 5 променів, вікно 0.75 с): "
        ).strip()
        if algo_choice in {"0", "1", "2"}:
            break
        print("Введи 0, 1 або 2.")

    # Configure detection mode
    if algo_choice == "0":
        td.SPIKE_DETECTION_MODE = True
        td.USE_RAW_POINTS = False
        td.ENABLE_THRESHOLD_FILTER = True
        td.DEBUG_SPIKE_MODE = False
    else:
        td.DEBUG_SPIKE_MODE = False
        if algo_choice == "1":
            # raw: шлемо всі точки всередині зони (полігон/сектор), без порога по базі
            td.SPIKE_DETECTION_MODE = False
            td.USE_RAW_POINTS = True
            td.ENABLE_THRESHOLD_FILTER = False
            td.ENABLE_ZONE_FILTER = True
        else:
            # debug-spike: без порога, мінімум 5 променів, групування до 0.75 с
            td.SPIKE_DETECTION_MODE = False
            td.USE_RAW_POINTS = False
            td.ENABLE_THRESHOLD_FILTER = False
            td.DEBUG_SPIKE_MODE = True
            td.ENABLE_ZONE_FILTER = True

    event_server = TouchEventServer(SERVER_HOST, SERVER_PORT)
    td.run_touch_detection(zone_points, is_custom_zone, mode, radius_limit, event_server)


if __name__ == "__main__":
    main()
