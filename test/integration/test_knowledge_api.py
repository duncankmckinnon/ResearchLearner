#!/usr/bin/env python3
"""
Simple test script for the knowledge store API endpoints
"""

import requests
import json
import time

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_endpoint(method, endpoint, data=None, params=None):
    """Test an API endpoint"""
    url = f"{BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, params=params)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url, params=params)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        print(f"{method} {endpoint}")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Success")
            result = response.json()
            print(f"Response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        else:
            print("❌ Failed")
            print(f"Error: {response.text}")
        print("-" * 50)
        return response
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        print("-" * 50)
        return None

def main():
    """Test the knowledge store API endpoints"""
    print("Testing Knowledge Store API Endpoints")
    print("=" * 50)
    
    # Test 1: Search knowledge store
    print("Test 1: Search knowledge store")
    search_data = {
        "query": "machine learning",
        "limit": 5,
        "user_id": "test_user"
    }
    test_endpoint("POST", "/knowledge/search", data=search_data)
    
    # Test 2: Get related papers
    print("Test 2: Get related papers")
    test_endpoint("GET", "/knowledge/papers/neural networks", params={"limit": 3, "user_id": "test_user"})
    
    # Test 3: Get research insights
    print("Test 3: Get research insights")
    test_endpoint("GET", "/knowledge/insights/deep learning", params={"limit": 5, "user_id": "test_user"})
    
    # Test 4: Get knowledge summary
    print("Test 4: Get knowledge summary")
    test_endpoint("GET", "/knowledge/summary/artificial intelligence", params={"user_id": "test_user"})
    
    # Test 5: Get all memories
    print("Test 5: Get all memories")
    test_endpoint("GET", "/knowledge/memories", params={"limit": 10, "user_id": "test_user"})
    
    print("Testing complete!")

if __name__ == "__main__":
    print("Note: Make sure the server is running on localhost:8000")
    print("You can start it with: python agent/main.py")
    print("Waiting 3 seconds before starting tests...")
    time.sleep(3)
    main()