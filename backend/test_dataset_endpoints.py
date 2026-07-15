#!/usr/bin/env python3
"""
Test script for custom dataset management endpoints.
This script tests all four endpoints with sample data.
"""

import requests
import json
import pandas as pd
import io
import tempfile
import os

import pytest

def _server_available():
    try:
        import requests as _rq
        return _rq.get("http://localhost:8080/api/health", timeout=1).ok
    except Exception:
        return False

# These are live-server integration checks, not unit tests.
pytestmark = pytest.mark.skipif(
    not _server_available(), reason="requires a running NeuroBlock server"
)



# Configuration
BASE_URL = "http://localhost:8080/api/datasets"

def create_sample_csv():
    """Create a sample CSV file for testing."""
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'feature3': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    
    # Create temporary CSV file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    return temp_file.name

def test_preview_endpoint():
    """Test the /preview endpoint."""
    print("Testing /preview endpoint...")
    
    csv_file = create_sample_csv()
    
    try:
        with open(csv_file, 'rb') as f:
            files = {'file': ('test_data.csv', f, 'text/csv')}
            response = requests.post(f"{BASE_URL}/preview", files=files)
            
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Preview endpoint successful!")
            print(f"Dataset shape: {data['data']['shape']}")
            print(f"Columns: {data['data']['columns']}")
            print(f"Data types: {data['data']['data_types']}")
            return data['data']
        else:
            print(f"❌ Preview endpoint failed: {response.text}")
            return None
            
    finally:
        os.unlink(csv_file)

def check_validate_endpoint(file_info):
    """Test the /validate endpoint."""
    print("\nTesting /validate endpoint...")
    
    config = {
        'preprocessing_config': {
            'target_column': 'target',
            'feature_columns': ['feature1', 'feature2', 'feature3'],
            'task_type': 'classification'
        },
        'file_info': file_info
    }
    
    response = requests.post(
        f"{BASE_URL}/validate",
        json=config,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Validation endpoint successful!")
        validation = data['validation_results']
        print(f"Valid: {validation['valid']}")
        print(f"Warnings: {validation['warnings']}")
        print(f"Errors: {validation['errors']}")
        print(f"Recommendations: {validation['recommendations']}")
        return True
    else:
        print(f"❌ Validation endpoint failed: {response.text}")
        return False

def test_create_endpoint():
    """Test the /create endpoint."""
    print("\nTesting /create endpoint...")
    
    csv_file = create_sample_csv()
    
    try:
        config = {
            'dataset_name': 'test_dataset',
            'target_column': 'target',
            'feature_columns': ['feature1', 'feature2', 'feature3'],
            'task_type': 'classification',
            'missing_value_strategy': 'drop'
        }
        
        with open(csv_file, 'rb') as f:
            files = {'file': ('test_data.csv', f, 'text/csv')}
            data = {'config': json.dumps(config)}
            response = requests.post(f"{BASE_URL}/create", files=files, data=data)
            
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Create endpoint successful!")
            print(f"Dataset name: {data['dataset']['name']}")
            print(f"Shape: {data['dataset']['shape']}")
            print(f"Task type: {data['dataset']['task_type']}")
            return True
        else:
            print(f"❌ Create endpoint failed: {response.text}")
            return False
            
    finally:
        os.unlink(csv_file)

def test_list_endpoint():
    """Test the /custom endpoint."""
    print("\nTesting /custom endpoint...")
    
    response = requests.get(f"{BASE_URL}/custom")
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ List endpoint successful!")
        print(f"Found {len(data['datasets'])} custom datasets")
        for dataset in data['datasets']:
            print(f"  - {dataset['name']} ({dataset['task_type']}, shape: {dataset['shape']})")
        return True
    else:
        print(f"❌ List endpoint failed: {response.text}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Custom Dataset Management Endpoints")
    print("=" * 50)
    
    # Test preview endpoint
    file_info = test_preview_endpoint()
    
    if file_info:
        # Test validate endpoint
        check_validate_endpoint(file_info)
        
        # Test create endpoint
        test_create_endpoint()
        
        # Test list endpoint
        test_list_endpoint()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")

if __name__ == "__main__":
    main() 