import time

from hokuyolx import HokuyoLX


LASER_MAX_RETRIES = 3
LASER_RECONNECT_DELAY = 0.5

laser = None


def create_laser():
    return HokuyoLX(addr=('192.168.0.10', 10940))


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

