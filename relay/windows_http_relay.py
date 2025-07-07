from flask import Flask, request, jsonify
import socket
import threading
import time

app = Flask(__name__)

# UDP broadcast socket for sending to ESP32s
broadcast_sock = None

def setup_broadcast_socket():
    """Setup UDP broadcast socket"""
    global broadcast_sock
    broadcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    broadcast_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    print("UDP broadcast socket ready")

@app.route('/relay/discover', methods=['POST'])
def relay_discovery():
    """Relay discovery request from WSL2 to LAN via UDP broadcast"""
    try:
        data = request.get_json()
        message = data.get('message', 'DISCOVER_CAMERAS')
        port = data.get('port', 8888)
        
        print(f"Relaying discovery message: {message} to port {port}")
        
        # Send UDP broadcast
        broadcast_sock.sendto(message.encode('utf-8'), ('255.255.255.255', port))
        print(f"Broadcasted to 255.255.255.255:{port}")
        
        return jsonify({'status': 'success', 'message': f'Relayed: {message}'})
        
    except Exception as e:
        print(f"Error relaying discovery: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'UDP Relay'})

if __name__ == '__main__':
    setup_broadcast_socket()
    print("HTTP Relay Server starting...")
    print("Endpoints:")
    print("  POST /relay/discover - Relay discovery message")
    print("  GET  /health         - Health check")
    print("Server will run on http://localhost:5001")
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=False)
    except KeyboardInterrupt:
        print("Relay server stopped.")
    finally:
        if broadcast_sock:
            broadcast_sock.close() 