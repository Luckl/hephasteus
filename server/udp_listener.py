import logging
import socket
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

# Global UDP listener for ESP32 responses
udp_listener_socket = None
udp_listener_thread = None
udp_listener_running = False

def udp_listener_thread_func(camera_streams, socketio):
    """Background thread function to continuously listen for ESP32 responses"""
    global udp_listener_socket, udp_listener_running
    
    try:
        # Create UDP socket for listening
        udp_listener_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_listener_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp_listener_socket.bind(('0.0.0.0', 8888))
        udp_listener_socket.settimeout(1.0)  # 1 second timeout for responsiveness
        
        logger.info("UDP listener started on port 8888")
        
        while udp_listener_running:
            try:
                data, addr = udp_listener_socket.recvfrom(1024)
                message = data.decode('utf-8').strip()
                
                if message.startswith('ESP32_CAMERA:'):
                    parts = message.split(':')
                    if len(parts) >= 3:
                        ip = parts[1]
                        port = int(parts[2])
                        
                        # Generate camera name from IP
                        camera_name = f"ESP32 Camera ({ip})"
                        
                        # Check if camera already exists
                        if camera_name not in camera_streams:
                            from camera_stream import CameraStream
                            camera_streams[camera_name] = CameraStream(camera_name, ip, port)
                            logger.info(f"Discovered new camera: {camera_name} at {ip}:{port}")
                            
                            # Emit discovery event to connected clients
                            socketio.emit('camera_discovered', {
                                'name': camera_name,
                                'ip': ip,
                                'port': port,
                                'timestamp': datetime.now().isoformat()
                            })
                        else:
                            # Update last seen time
                            camera_streams[camera_name].discovered_time = datetime.now()
                            
            except socket.timeout:
                continue  # Normal timeout, continue listening
            except Exception as e:
                logger.error(f"UDP listener error: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Failed to start UDP listener: {e}")
    finally:
        if udp_listener_socket:
            udp_listener_socket.close()
            logger.info("UDP listener socket closed")

def start_udp_listener(camera_streams, socketio):
    """Start the continuous UDP listener thread"""
    global udp_listener_thread, udp_listener_running
    
    if udp_listener_thread and udp_listener_thread.is_alive():
        logger.info("UDP listener already running")
        return
    
    udp_listener_running = True
    udp_listener_thread = threading.Thread(
        target=udp_listener_thread_func, 
        args=(camera_streams, socketio), 
        daemon=True
    )
    udp_listener_thread.start()
    logger.info("UDP listener thread started")

def stop_udp_listener():
    """Stop the continuous UDP listener thread"""
    global udp_listener_running, udp_listener_socket
    
    udp_listener_running = False
    if udp_listener_socket:
        udp_listener_socket.close()
    logger.info("UDP listener stopped") 