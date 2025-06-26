import socket
import time

# Simple UDP test server
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 8888))
print("Listening on 0.0.0.0:8888")

while True:
    try:
        data, addr = sock.recvfrom(1024)
        print(f"Received from {addr}: {data.decode()}")
    except KeyboardInterrupt:
        break

sock.close()