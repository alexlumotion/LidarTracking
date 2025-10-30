from event_server import TouchEventServer
from touch_detection import FLIP_Y, run_touch_detection
from zone_config import load_zone_points


SERVER_HOST = "0.0.0.0"
SERVER_PORT = 9100


def main():
    zone_config = load_zone_points()
    zone_points = zone_config["points"]
    if FLIP_Y:
        zone_points = [(pt[0], -pt[1]) for pt in zone_points]
    is_custom_zone = zone_config["is_custom_zone"]
    mode = zone_config["mode"]
    radius_limit = zone_config.get("radius")

    event_server = TouchEventServer(SERVER_HOST, SERVER_PORT)
    run_touch_detection(zone_points, is_custom_zone, mode, radius_limit, event_server)


if __name__ == "__main__":
    main()

