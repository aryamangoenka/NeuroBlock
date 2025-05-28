#!/usr/bin/env python3
"""
Test script for extended DatasetRegistry functionality.
This script tests custom dataset registration, loading, and integration.
"""

import sys
import os
import numpy as np
import tempfile
import json
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.dataset_loader import (
    DatasetRegistry, 
    register_custom_dataset, 
    get_custom_datasets, 
    load_custom_dataset,
    load_dataset
)

def create_sample_custom_dataset():
    """Create a sample custom dataset for testing."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 4
    
    # Create features
    X = np.random.randn(n_samples, n_features)
    
    # Create binary classification target
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Create temporary files
    temp_dir = tempfile.mkdtemp()
    dataset_name = 'test_custom_dataset'
    
    # Save data file
    data_file = os.path.join(temp_dir, f'{dataset_name}.npz')
    np.savez(data_file, X=X, y=y)
    
    # Create metadata
    metadata = {
        'name': dataset_name,
        'task_type': 'classification',
        'target_column': 'target',
        'feature_columns': ['feature1', 'feature2', 'feature3', 'feature4'],
        'original_filename': 'test_data.csv',
        'shape': [n_samples, n_features + 1],
        'processed_shape': [n_samples, n_features],
        'created_at': datetime.now().isoformat(),
        'file_path': data_file,
        'class_labels': ['class_0', 'class_1']
    }
    
    # Save metadata file
    metadata_file = os.path.join(temp_dir, f'{dataset_name}_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return {
        'name': dataset_name,
        'file_path': data_file,
        'metadata_path': metadata_file,
        'task_type': 'classification',
        'feature_count': n_features,
        'class_labels': ['class_0', 'class_1'],
        'temp_dir': temp_dir
    }

def test_dataset_registry_basic():
    """Test basic DatasetRegistry functionality."""
    print("Testing basic DatasetRegistry functionality...")
    
    registry = DatasetRegistry()
    
    # Test getting available datasets
    available = registry.get_available_datasets()
    print(f"✅ Available built-in datasets: {available}")
    
    # Test loading a built-in dataset
    try:
        (x_train, y_train), (x_test, y_test) = registry.get_loader("Iris")()
        print(f"✅ Loaded Iris dataset: train {x_train.shape}, test {x_test.shape}")
    except Exception as e:
        print(f"❌ Failed to load Iris dataset: {e}")
    
    return True

def test_custom_dataset_registration():
    """Test custom dataset registration."""
    print("\nTesting custom dataset registration...")
    
    # Create sample custom dataset
    dataset_config = create_sample_custom_dataset()
    
    try:
        # Test registration
        dataset_id = register_custom_dataset(dataset_config)
        print(f"✅ Registered custom dataset: {dataset_id}")
        
        # Test listing custom datasets
        custom_datasets = get_custom_datasets()
        print(f"✅ Found {len(custom_datasets)} custom datasets")
        
        for dataset in custom_datasets:
            print(f"  - {dataset['name']} ({dataset['task_type']}, shape: {dataset['shape']})")
        
        return dataset_config
        
    except Exception as e:
        print(f"❌ Failed to register custom dataset: {e}")
        return None

def test_custom_dataset_loading(dataset_config):
    """Test custom dataset loading."""
    print("\nTesting custom dataset loading...")
    
    try:
        # Test loading through registry
        (x_train, y_train), (x_test, y_test) = load_custom_dataset(dataset_config['name'])
        
        print(f"✅ Loaded custom dataset '{dataset_config['name']}':")
        print(f"  - x_train shape: {x_train.shape}")
        print(f"  - y_train shape: {y_train.shape}")
        print(f"  - x_test shape: {x_test.shape}")
        print(f"  - y_test shape: {y_test.shape}")
        print(f"  - Data types: x_train={x_train.dtype}, y_train={y_train.dtype}")
        
        # Test loading through main load_dataset function
        (x_train2, y_train2), (x_test2, y_test2) = load_dataset(dataset_config['name'])
        print(f"✅ Loaded via main load_dataset function - shapes match: {x_train2.shape == x_train.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load custom dataset: {e}")
        return False

def test_integration_with_existing_system():
    """Test integration with existing dataset loading system."""
    print("\nTesting integration with existing system...")
    
    try:
        # Test that custom datasets appear in available datasets list
        from backend.dataset_loader import dataset_registry
        
        all_datasets = dataset_registry.get_available_datasets()
        custom_datasets = dataset_registry.get_custom_datasets()
        
        print(f"✅ Total available datasets: {len(all_datasets)}")
        print(f"✅ Custom datasets: {len(custom_datasets)}")
        
        # Verify custom datasets are in the main list
        custom_names = {ds['name'] for ds in custom_datasets}
        custom_in_main = [name for name in all_datasets if name in custom_names]
        
        print(f"✅ Custom datasets in main list: {len(custom_in_main)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def cleanup_temp_files(dataset_config):
    """Clean up temporary test files."""
    if dataset_config and 'temp_dir' in dataset_config:
        import shutil
        try:
            shutil.rmtree(dataset_config['temp_dir'])
            print(f"✅ Cleaned up temporary files: {dataset_config['temp_dir']}")
        except Exception as e:
            print(f"⚠️  Could not clean up temp files: {e}")

def main():
    """Run all tests."""
    print("🧪 Testing Extended DatasetRegistry Functionality")
    print("=" * 60)
    
    dataset_config = None
    
    try:
        # Test basic functionality
        test_dataset_registry_basic()
        
        # Test custom dataset registration
        dataset_config = test_custom_dataset_registration()
        
        if dataset_config:
            # Test custom dataset loading
            test_custom_dataset_loading(dataset_config)
            
            # Test integration
            test_integration_with_existing_system()
    
    finally:
        # Clean up
        if dataset_config:
            cleanup_temp_files(dataset_config)
    
    print("\n" + "=" * 60)
    print("✅ All DatasetRegistry tests completed!")

if __name__ == "__main__":
    main() 