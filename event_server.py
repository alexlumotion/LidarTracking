import json
import socket
import threading
from queue import Queue


class TouchEventServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.server_socket.settimeout(1.0)
        self.client = None
        self.client_lock = threading.Lock()
        self.queue: Queue = Queue()
        self.running = True
        self.accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self.accept_thread.start()
        self.sender_thread.start()
        print(f"📡 TouchEventServer listening on {self.host}:{self.port}")

    def _accept_loop(self):
        while self.running:
            try:
                conn, addr = self.server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with self.client_lock:
                if self.client:
                    try:
                        self.client.close()
                    except OSError:
                        pass
                self.client = conn
                self.client.settimeout(2.0)
            print(f"🔌 Client connected from {addr}")

    def _sender_loop(self):
        while self.running:
            event = self.queue.get()
            if event is None:
                break
            payload = json.dumps(event) + "\n"
            with self.client_lock:
                client = self.client
            if not client:
                continue
            try:
                client.sendall(payload.encode("utf-8"))
            except OSError:
                print("⚠️ Client disconnected")
                with self.client_lock:
                    try:
                        if self.client:
                            self.client.close()
                    except OSError:
                        pass
                    self.client = None

    def send_event(self, event: dict):
        if self.running:
            self.queue.put(event)

    def shutdown(self):
        self.running = False
        try:
            self.server_socket.close()
        except OSError:
            pass
        self.queue.put(None)
        with self.client_lock:
            if self.client:
                try:
                    self.client.close()
                except OSError:
                    pass
                self.client = None
        print("🛑 TouchEventServer stopped")

