#!/usr/bin/env python3
"""
Find Windows host IP that WSL2 can reach
"""

import subprocess
import re
import socket

def find_windows_host_ip():
    """Try different methods to find Windows host IP"""
    print("Searching for Windows host IP...")
    
    # Method 1: Try common WSL2 gateway IPs
    common_gateways = [
        "172.29.160.1",  # Your current gateway
        "172.30.0.1",
        "172.31.0.1",
        "192.168.1.1",   # Common router IP
        "10.0.0.1",      # Common router IP
    ]
    
    for gateway in common_gateways:
        print(f"Testing gateway: {gateway}")
        try:
            result = subprocess.run(['ping', '-c', '1', '-W', '1', gateway], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✅ Found reachable gateway: {gateway}")
                return gateway
        except:
            pass
    
    # Method 2: Try to get Windows hostname and resolve it
    try:
        print("Trying to resolve Windows hostname...")
        result = subprocess.run(['hostname'], capture_output=True, text=True)
        if result.returncode == 0:
            hostname = result.stdout.strip()
            print(f"WSL2 hostname: {hostname}")
            
            # Try different hostname variations
            hostname_variations = [
                f"{hostname}.local",
                f"{hostname}",
                "DESKTOP-16H97LM.local",  # From your terminal prompt
                "DESKTOP-16H97LM"
            ]
            
            for host in hostname_variations:
                print(f"Trying to resolve: {host}")
                try:
                    ip = socket.gethostbyname(host)
                    print(f"✅ Resolved {host} to {ip}")
                    return ip
                except socket.gaierror:
                    pass
    except Exception as e:
        print(f"Error resolving hostname: {e}")
    
    # Method 3: Try to get Windows IP from WSL2 environment
    try:
        print("Checking WSL2 environment variables...")
        import os
        if 'WSL_DISTRO_NAME' in os.environ:
            print(f"WSL distro: {os.environ['WSL_DISTRO_NAME']}")
    except:
        pass
    
    print("❌ Could not find reachable Windows host IP")
    return None

if __name__ == "__main__":
    ip = find_windows_host_ip()
    if ip:
        print(f"\nRecommended Windows host IP: {ip}")
        print(f"You can test it with: ping {ip}")
    else:
        print("\nNo reachable Windows host found.")
        print("You may need to:")
        print("1. Check Windows firewall settings")
        print("2. Restart WSL2: wsl --shutdown")
        print("3. Use a different approach for device discovery") 