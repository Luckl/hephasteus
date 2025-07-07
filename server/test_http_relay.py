#!/usr/bin/env python3
"""
Test HTTP relay connection
"""

import requests
import json

def test_http_relay():
    print("Testing HTTP relay connection...")
    
    # Windows host IP from /etc/resolv.conf
    windows_host_ip = "172.29.160.1"
    relay_url = f"http://{windows_host_ip}:5001"
    
    try:
        # Test health endpoint
        print(f"Testing health endpoint: {relay_url}/health")
        response = requests.get(f"{relay_url}/health", timeout=5)
        print(f"Health check status: {response.status_code}")
        if response.status_code == 200:
            print(f"Health response: {response.json()}")
        
        # Test discovery relay
        print(f"Testing discovery relay: {relay_url}/relay/discover")
        payload = {
            'message': 'TEST_DISCOVERY',
            'port': 8888
        }
        response = requests.post(f"{relay_url}/relay/discover", json=payload, timeout=5)
        print(f"Discovery relay status: {response.status_code}")
        if response.status_code == 200:
            print(f"Discovery response: {response.json()}")
        
        print("HTTP relay test completed successfully")
        
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to HTTP relay at {relay_url}")
        print("Make sure the relay server is running on Windows")
    except Exception as e:
        print(f"HTTP relay test failed: {e}")

if __name__ == "__main__":
    test_http_relay() 