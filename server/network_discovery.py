import logging
import subprocess
import platform
import re
import socket
from datetime import datetime

logger = logging.getLogger(__name__)

def get_network_interfaces():
    """Get all available network interfaces with their broadcast addresses"""
    interfaces = []
    
    try:
        if platform.system() == "Windows":
            # Windows: use ipconfig
            result = subprocess.run(['ipconfig'], capture_output=True, text=True)
            if result.returncode == 0:
                interfaces = parse_windows_ipconfig(result.stdout)
        else:
            # Linux/Mac: use ifconfig or ip
            try:
                result = subprocess.run(['ip', 'addr'], capture_output=True, text=True)
                if result.returncode == 0:
                    # logger.debug("Using 'ip addr' command to get network interfaces")
                    interfaces = parse_linux_ip_addr(result.stdout)
                else:
                    logger.debug(f"'ip addr' failed with return code {result.returncode}")
                    # Fallback to ifconfig
                    result = subprocess.run(['ifconfig'], capture_output=True, text=True)
                    if result.returncode == 0:
                        logger.debug("Using 'ifconfig' command to get network interfaces")
                        interfaces = parse_ifconfig(result.stdout)
                    else:
                        logger.debug(f"'ifconfig' failed with return code {result.returncode}")
            except FileNotFoundError:
                logger.debug("'ip' command not found, trying 'ifconfig'")
                # Try ifconfig as fallback
                result = subprocess.run(['ifconfig'], capture_output=True, text=True)
                if result.returncode == 0:
                    interfaces = parse_ifconfig(result.stdout)
                else:
                    logger.debug(f"'ifconfig' failed with return code {result.returncode}")
                    
    except Exception as e:
        logger.error(f"Error getting network interfaces: {e}")
    
    return interfaces

def parse_windows_ipconfig(output):
    """Parse Windows ipconfig output"""
    interfaces = []
    current_interface = {}
    
    for line in output.split('\n'):
        line = line.strip()
        
        # Interface name
        if line and not line.startswith(' ') and ':' in line and not line.startswith('Windows'):
            if current_interface:
                interfaces.append(current_interface)
            current_interface = {'name': line.split(':')[0]}
        
        # IP address
        elif 'IPv4 Address' in line:
            match = re.search(r'(\d+\.\d+\.\d+\.\d+)', line)
            if match:
                current_interface['ip'] = match.group(1)
        
        # Subnet mask
        elif 'Subnet Mask' in line:
            match = re.search(r'(\d+\.\d+\.\d+\.\d+)', line)
            if match:
                current_interface['subnet'] = match.group(1)
    
    if current_interface:
        interfaces.append(current_interface)
    
    return interfaces

def parse_linux_ip_addr(output):
    """Parse Linux 'ip addr' output"""
    interfaces = []
    current_interface = {}
    
    for line in output.split('\n'):
        line = line.strip()
        
        # Interface number and name
        if re.match(r'^\d+:', line):
            if current_interface:
                interfaces.append(current_interface)
            parts = line.split(':')
            current_interface = {'name': parts[1].strip()}
        
        # IP address - look for both IP and broadcast on the same line
        elif 'inet ' in line and 'brd ' in line:
            # Extract IP address
            ip_match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', line)
            if ip_match:
                current_interface['ip'] = ip_match.group(1)
            
            # Extract broadcast address
            brd_match = re.search(r'brd (\d+\.\d+\.\d+\.\d+)', line)
            if brd_match:
                current_interface['broadcast'] = brd_match.group(1)
        
        # IP address without broadcast (fallback)
        elif 'inet ' in line and 'brd ' not in line:
            match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', line)
            if match:
                current_interface['ip'] = match.group(1)
    
    if current_interface:
        interfaces.append(current_interface)
    
    return interfaces

def parse_ifconfig(output):
    """Parse ifconfig output (Mac/Linux)"""
    interfaces = []
    current_interface = {}
    
    for line in output.split('\n'):
        line = line.strip()
        
        # Interface name
        if line and not line.startswith(' ') and ':' in line:
            if current_interface:
                interfaces.append(current_interface)
            current_interface = {'name': line.split(':')[0]}
        
        # IP address
        elif 'inet ' in line:
            match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', line)
            if match:
                current_interface['ip'] = match.group(1)
        
        # Broadcast address
        elif 'broadcast ' in line:
            match = re.search(r'broadcast (\d+\.\d+\.\d+\.\d+)', line)
            if match:
                current_interface['broadcast'] = match.group(1)
    
    if current_interface:
        interfaces.append(current_interface)
    
    return interfaces

def calculate_broadcast_address(ip, subnet_mask):
    """Calculate broadcast address from IP and subnet mask"""
    try:
        ip_parts = [int(x) for x in ip.split('.')]
        mask_parts = [int(x) for x in subnet_mask.split('.')]
        
        # Calculate network address
        network = [ip_parts[i] & mask_parts[i] for i in range(4)]
        
        # Calculate broadcast address
        broadcast = [network[i] | (255 - mask_parts[i]) for i in range(4)]
        
        return '.'.join(map(str, broadcast))
    except Exception as e:
        logger.error(f"Error calculating broadcast address: {e}")
        return None

def select_best_interface(interfaces):
    """Select the best interface for broadcasting to ESP32 devices"""
    valid_interfaces = []
    
    for iface in interfaces:
        # Skip interfaces without IP
        if 'ip' not in iface:
            continue
            
        ip = iface['ip']
        
        # Skip loopback and link-local addresses
        if (ip.startswith('127.') or 
            ip.startswith('169.254.') or
            ip.startswith('10.255.255.')):  # Skip WSL2 loopback addresses
            continue
        
        # Calculate broadcast address if not provided
        if 'broadcast' not in iface and 'subnet' in iface:
            broadcast = calculate_broadcast_address(ip, iface['subnet'])
            if broadcast:
                iface['broadcast'] = broadcast
        elif 'broadcast' not in iface:
            continue
        
        # Only include interfaces with broadcast address
        if 'broadcast' in iface:
            valid_interfaces.append(iface)
    
    # Sort by preference: prefer actual network interfaces over loopback
    # Give higher priority to interfaces that look like real network interfaces
    def interface_priority(iface):
        name = iface.get('name', '').lower()
        ip = iface.get('ip', '')
        
        # Highest priority: eth0, wlan0, etc. (real network interfaces)
        if name in ['eth0', 'wlan0', 'en0', 'en1', 'wlan1']:
            return 0
        # Medium priority: other interfaces with private IP ranges
        elif ip.startswith('192.168.') or ip.startswith('10.') or ip.startswith('172.'):
            return 1
        # Lower priority: everything else
        else:
            return 2
    
    valid_interfaces.sort(key=interface_priority)
    
    if valid_interfaces:
        selected = valid_interfaces[0]
        return selected
    
    logger.warning("No suitable network interfaces found")
    return None

def run_discovery():
    """Broadcast discovery requests to ESP32 cameras"""
    # logger.debug("Broadcasting ESP32 camera discovery request...")
    
    # Get all network interfaces
    interfaces = get_network_interfaces()
    
    # Select the best interface
    selected_interface = select_best_interface(interfaces)
    
    if not selected_interface:
        logger.error("No suitable network interface found for broadcasting")
        return
    
    network_ip = selected_interface['ip']
    broadcast_ip = selected_interface['broadcast']
    
    try:
        # Create socket for broadcasting
        broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        # Bind to specific interface to ensure broadcasts go to the right network
        broadcast_socket.bind((network_ip, 0))
        
        try:
            # Send discovery request to broadcast address
            discovery_request = "DISCOVER_CAMERAS"
            broadcast_addresses = [
                (broadcast_ip, 8888),  # Interface-specific broadcast
                ('255.255.255.255', 8888),  # Global broadcast
            ]
            
            for broadcast_addr in broadcast_addresses:
                # try:
                    bytes_sent = broadcast_socket.sendto(discovery_request.encode('utf-8'), broadcast_addr)
                    # logger.debug(f"Sent discovery to {broadcast_addr}")
                # except Exception as e:
                    # logger.debug(f"Failed to send to {broadcast_addr}: {e}")
                    # pass
        finally:
            broadcast_socket.close()
            
    except Exception as e:
        logger.error(f"Discovery broadcast error: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}") 