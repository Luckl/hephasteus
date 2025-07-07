import socket

# Listen for UDP packets from WSL2 (on Windows host)
LISTEN_PORT = 8888
RELAY_PORT = 8888  # ESP32s listen here

# Get the WSL2 interface IP (usually 172.x.x.x)
# You can find this by running 'ipconfig' on Windows and looking for the WSL2 interface
WSL2_INTERFACE_IP = "0.0.0.0"  # This should match the nameserver in WSL2's /etc/resolv.conf

print(f"Starting UDP relay...")
print(f"Listening on {WSL2_INTERFACE_IP}:{LISTEN_PORT}")
print(f"Rebroadcasting to 255.255.255.255:{RELAY_PORT}")

# Listen on the WSL2 interface specifically
listen_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
listen_sock.bind((WSL2_INTERFACE_IP, LISTEN_PORT))

# Set up broadcast socket
broadcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
broadcast_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

print(f"UDP relay running. Press Ctrl+C to stop.")
try:
    while True:
        data, addr = listen_sock.recvfrom(4096)
        print(f"Received from {addr}: {data.decode('utf-8', errors='ignore')}")
        broadcast_sock.sendto(data, ('255.255.255.255', RELAY_PORT))
        print(f"Rebroadcasted to 255.255.255.255:{RELAY_PORT}")
except KeyboardInterrupt:
    print("Relay stopped.")
except Exception as e:
    print(f"Error: {e}")
finally:
    listen_sock.close()
    broadcast_sock.close() 