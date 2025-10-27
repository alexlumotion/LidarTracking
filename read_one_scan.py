import socket
import time

LIDAR_IP = "192.168.0.10"
LIDAR_PORT = 10940


def decode_scip_block(data: str):
    """Декодує ASCII-дані SCIP у масив відстаней (мм)."""
    vals = []
    for i in range(0, len(data) - 2, 3):
        a = ord(data[i]) - 48
        b = ord(data[i + 1]) - 48
        c = ord(data[i + 2]) - 48
        vals.append(a + (b << 6) + (c << 12))
    return vals


def read_one_scan():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"🔌 Connecting to {LIDAR_IP}:{LIDAR_PORT} ...")
        s.connect((LIDAR_IP, LIDAR_PORT))
        s.settimeout(3.0)

        # 1️⃣ Увімкнути лазер
        print("💡 Enabling laser (BM)...")
        s.sendall(b"BM\n")
        time.sleep(0.2)
        bm_resp = s.recv(128).decode(errors="ignore")
        print("BM response:", bm_resp.strip())

        if "00P" not in bm_resp:
            print("⚠️ Laser did not confirm activation (BM).")
            return

        # 2️⃣ Активувати протокол SCIP 2.0
        s.sendall(b"SCIP2.0\n")
        time.sleep(0.1)

        # 3️⃣ Запитати один повний скан
        cmd = "MD0000076800\n"  # безперервне вимірювання одного циклу
        print("📡 Sending scan request:", cmd.strip())
        s.sendall(cmd.encode())
        time.sleep(0.1)

        buffer = ""
        start_time = time.time()

        ack_lines_seen = 0
        while True:
            try:
                chunk = s.recv(4096)
                if not chunk:
                    break
                text = chunk.decode("ascii", errors="ignore")
                buffer += text

                # Hokuyo завершує передачу через фінальне "00P" або "99b"
                lines = buffer.split("\n")
                ack_lines_seen = sum(1 for ln in lines if ln.strip() == "00P")
                if any(ln.strip() == "99b" for ln in lines):
                    break
                if ack_lines_seen >= 2:
                    break
                if time.time() - start_time > 5:
                    print("⏱ Timeout waiting for data.")
                    break
            except socket.timeout:
                break

        # 4️⃣ Вимкнути передачу
        s.sendall(b"QT\n")
        time.sleep(0.05)

    # 5️⃣ Обробити дані
    raw_lines = [ln.rstrip("\r") for ln in buffer.split("\n") if ln]
    print(f"📄 Total raw lines: {len(raw_lines)}")
    for idx, raw in enumerate(raw_lines[:12]):
        tail_bytes = " ".join(f"{ord(ch):02x}" for ch in raw[-4:])
        head = raw[:8] + ("..." if len(raw) > 8 else "")
        print(f"🧾 RAW[{idx}]: len={len(raw)}, head={repr(head)}, last4={repr(raw[-4:])}, bytes={tail_bytes}")

    measurement_chunks = []
    for raw in raw_lines:
        line = raw.strip()
        if not line or line.startswith(("MD", "BM", "QT", "00P", "99b", "OK")):
            continue
        # Останній символ у кожному рядку — контрольна сума, її потрібно відкинути.
        if len(line) > 1:
            measurement_chunks.append(line[:-1])
        else:
            measurement_chunks.append(line)
    lines = measurement_chunks
    print(f"📏 Measurement chunks: {len(lines)}, sample lengths: {[len(chunk) for chunk in lines[:5]]}")
    if not lines:
        print("⚠️ No valid data lines received.")
        return

    joined = "".join(lines)
    distances = decode_scip_block(joined)
    valid = [d for d in distances if d > 0]

    print(f"✅ Received {len(distances)} points")
    print("First 20 distances (mm):", valid[:20])
    if valid:
        print(f"min={min(valid)} mm, max={max(valid)} mm, mean={sum(valid)/len(valid):.1f} mm")
    else:
        print("⚠️ All distances are zero.")


if __name__ == "__main__":
    read_one_scan()
