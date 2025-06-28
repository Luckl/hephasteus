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
                    interfaces = parse_linux_ip_addr(result.stdout)
                else:
                    # Fallback to ifconfig
                    result = subprocess.run(['ifconfig'], capture_output=True, text=True)
                    if result.returncode == 0:
                        interfaces = parse_ifconfig(result.stdout)
            except FileNotFoundError:
                # Try ifconfig as fallback
                result = subprocess.run(['ifconfig'], capture_output=True, text=True)
                if result.returncode == 0:
                    interfaces = parse_ifconfig(result.stdout)
                    
    except Exception as e:
        logger.error(f"Error getting network interfaces: {e}")
    
    return interfaces

def parse_windows_ipconfig(output):
    """Parse Windows ipconfig output"""
    interfaces = []
    current_interface = {}
    
    for line in output.split('\n'):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # New interface section - starts with adapter name and ends with colon
        if line.endswith(':') and not line.startswith(' '):
            # Save previous interface if it has data
            if current_interface and 'name' in current_interface:
                interfaces.append(current_interface)
            
            # Start new interface
            current_interface = {'name': line[:-1].strip()}  # Remove the colon
        
        # IPv4 Address
        elif 'IPv4 Address' in line and ':' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                ip = parts[1].strip()
                # Remove "(Preferred)" suffix if present
                if '(Preferred)' in ip:
                    ip = ip.replace('(Preferred)', '').strip()
                if ip and ip != '(Preferred)':
                    current_interface['ip'] = ip
        
        # Subnet Mask
        elif 'Subnet Mask' in line and ':' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                mask = parts[1].strip()
                if mask:
                    current_interface['subnet'] = mask
        
        # Default Gateway
        elif 'Default Gateway' in line and ':' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                gateway = parts[1].strip()
                if gateway:
                    current_interface['gateway'] = gateway
    
    # Don't forget the last interface
    if current_interface and 'name' in current_interface:
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
        
        # IP address
        elif 'inet ' in line:
            match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', line)
            if match:
                current_interface['ip'] = match.group(1)
        
        # Broadcast address
        elif 'brd ' in line:
            match = re.search(r'brd (\d+\.\d+\.\d+\.\d+)', line)
            if match:
                current_interface['broadcast'] = match.group(1)
    
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
    logger.debug(f"Found {len(interfaces)} network interfaces to evaluate")
    valid_interfaces = []
    
    for iface in interfaces:
        # Skip interfaces without IP
        if 'ip' not in iface:
            logger.debug(f"Skipping interface {iface.get('name', 'Unknown')} - no IP address")
            continue
            
        ip = iface['ip']
        
        # Skip loopback, link-local, and WSL interfaces
        if (ip.startswith('127.') or 
            ip.startswith('169.254.') or 
            ip.startswith('172.29.') or  # WSL2 default
            ip.startswith('172.30.') or  # WSL2 range
            ip.startswith('172.31.') or  # WSL2 range
            'wsl' in iface['name'].lower() or
            'docker' in iface['name'].lower() or
            'veth' in iface['name'].lower()):
            logger.debug(f"Skipping interface {iface['name']} ({ip}) - unsuitable for broadcasting")
            continue
        
        # Calculate broadcast address if not provided
        if 'broadcast' not in iface and 'subnet' in iface:
            broadcast = calculate_broadcast_address(ip, iface['subnet'])
            if broadcast:
                iface['broadcast'] = broadcast
                logger.debug(f"Calculated broadcast {broadcast} for {iface['name']} ({ip})")
        
        # Only include interfaces with broadcast address
        if 'broadcast' in iface:
            logger.debug(f"Valid interface: {iface['name']} - IP: {ip}, Broadcast: {iface['broadcast']}")
            valid_interfaces.append(iface)
    
    # Sort by preference: prefer interfaces with gateways (connected to router)
    valid_interfaces.sort(key=lambda x: 'gateway' not in x)
    
    if valid_interfaces:
        selected = valid_interfaces[0]
        logger.debug(f"Selected interface: {selected['name']} ({selected['ip']})")
        return selected
    
    logger.warning("No suitable network interfaces found")
    return None

def run_discovery():
    """Broadcast discovery requests to ESP32 cameras"""
    logger.debug("Broadcasting ESP32 camera discovery request...")
    
    # Get all network interfaces
    interfaces = get_network_interfaces()
    logger.debug(f"Found {len(interfaces)} network interfaces")
    
    # Select the best interface
    selected_interface = select_best_interface(interfaces)
    
    if not selected_interface:
        logger.error("No suitable network interface found for broadcasting")
        return
    
    network_ip = selected_interface['ip']
    broadcast_ip = selected_interface['broadcast']
    
    logger.debug(f"Selected interface: {selected_interface['name']}")
    logger.debug(f"Local IP: {network_ip}")
    logger.debug(f"Broadcast IP: {broadcast_ip}")
    
    try:
        # Create socket for broadcasting
        broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        # Bind to specific interface to ensure broadcasts go to the right network
        broadcast_socket.bind((network_ip, 0))
        logger.debug(f"Bound to interface {network_ip}")
        
        try:
            # Send discovery request to broadcast
            discovery_request = "DISCOVER_CAMERAS"
            broadcast_addr = (broadcast_ip, 8888)  # ESP32s listen on 8888
            broadcast_socket.sendto(discovery_request.encode('utf-8'), broadcast_addr)
            logger.debug(f"Sent discovery request: {discovery_request} to {broadcast_addr}")
            
        finally:
            broadcast_socket.close()
            
    except Exception as e:
        logger.error(f"Discovery broadcast error: {e}") 