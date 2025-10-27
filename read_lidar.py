import socket

# IP та порт лідара
LIDAR_IP = "192.168.0.10"
LIDAR_PORT = 10940

# Команда SCIP: 'VV' — отримати версію
COMMAND = "VV\n"

def read_lidar_version():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"🔌 Connecting to {LIDAR_IP}:{LIDAR_PORT} ...")
        s.connect((LIDAR_IP, LIDAR_PORT))
        s.sendall(COMMAND.encode())
        response = s.recv(1024)
        print("📡 Response:\n", response.decode(errors='ignore'))

if __name__ == "__main__":
    read_lidar_version()
