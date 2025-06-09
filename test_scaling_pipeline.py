#!/usr/bin/env python3
"""Test script to trace scaling issues through the pipeline"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import json
import os
from flask import Flask
from datetime import datetime

# Import our modules
from backend.api.dataset_routes import create_custom_dataset
from backend.dataset_loader import dataset_registry, load_dataset
from backend.utils.session_manager import get_session_datasets_dir, get_session_id

def test_scaling_pipeline():
    """Test the complete scaling pipeline"""
    
    print("🧪 TESTING SCALING PIPELINE")
    print("=" * 50)
    
    # Create Flask app context
    app = Flask(__name__)
    app.config['PROJECT_ROOT'] = '.'
    
    with app.app_context():
        # 1. Load the test dataset
        print("\n📊 Step 1: Loading test dataset")
        df = pd.read_csv('test_scaling.csv')
        print("Original data ranges:")
        print(df.describe())
        
        # 2. Simulate dataset creation process (what happens in dataset_routes.py)
        print("\n🔧 Step 2: Simulating dataset creation")
        
        dataset_name = 'test_scaling'
        target_column = 'target'
        feature_columns = ['feature1', 'feature2', 'feature3']
        task_type = 'regression'
        
        # Extract features and target
        X = df[feature_columns]
        y = df[target_column]
        
        print(f"Features before processing:")
        print(f"X ranges: {[(col, X[col].min(), X[col].max()) for col in X.columns]}")
        print(f"y range: {y.min()} - {y.max()}")
        
        # Basic preprocessing (what happens in create_custom_dataset)
        categorical_columns = X.select_dtypes(include=['object']).columns
        print(f"Categorical columns: {list(categorical_columns)}")
        
        # Create processed dataset
        processed_data = {
            'name': dataset_name,
            'task_type': task_type,
            'target_column': target_column,
            'feature_columns': feature_columns,
            'created_at': datetime.now().isoformat(),
            'categorical_mappings': {},
            'class_labels': None,
            'feature_types': ['int', 'int', 'int']
        }
        
        # Save as .npz (simulating the actual save process)
        datasets_dir = get_session_datasets_dir()
        os.makedirs(datasets_dir, exist_ok=True)
        
        dataset_file = os.path.join(datasets_dir, f'{dataset_name}.npz')
        np.savez(dataset_file, X=X.values, y=y.values)
        
        metadata_file = os.path.join(datasets_dir, f'{dataset_name}_metadata.json')
        processed_data['file_path'] = dataset_file
        processed_data['processed_shape'] = [X.shape[0], X.shape[1]]
        
        with open(metadata_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
            
        print(f"✅ Created dataset files:")
        print(f"  Data: {dataset_file}")
        print(f"  Metadata: {metadata_file}")
        
        # 3. Test loading through dataset_loader.py
        print("\n🔄 Step 3: Testing dataset loading (with scaling)")
        
        try:
            # Clear any existing registration
            if dataset_name in dataset_registry.loaders:
                del dataset_registry.loaders[dataset_name]
                
            # Load dataset (this should apply scaling)
            (x_train, y_train), (x_test, y_test) = load_dataset(dataset_name)
            
            print(f"✅ Dataset loaded successfully!")
            print(f"Training data shapes: X={x_train.shape}, y={y_train.shape}")
            print(f"Test data shapes: X={x_test.shape}, y={y_test.shape}")
            
            # Check if scaling was applied
            print(f"\n📈 Checking if scaling was applied:")
            print(f"X_train mean per feature: {np.mean(x_train.numpy(), axis=0)}")
            print(f"X_train std per feature: {np.std(x_train.numpy(), axis=0)}")
            print(f"X_train min per feature: {np.min(x_train.numpy(), axis=0)}")
            print(f"X_train max per feature: {np.max(x_train.numpy(), axis=0)}")
            
            # Check target values
            print(f"\ny_train range: {np.min(y_train.numpy())} - {np.max(y_train.numpy())}")
            
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            import traceback
            traceback.print_exc()
            
        # 4. Check if scaler was saved
        print("\n💾 Step 4: Checking if scaler was saved")
        scaler_path = os.path.join(datasets_dir, f'{dataset_name}_scaler.pkl')
        
        if os.path.exists(scaler_path):
            print(f"✅ Scaler saved at: {scaler_path}")
            
            import pickle
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            print(f"Scaler mean: {scaler.mean_}")
            print(f"Scaler scale: {scaler.scale_}")
            
        else:
            print(f"❌ Scaler NOT saved at: {scaler_path}")
            
        # 5. Test a prediction to see scaling in validation
        print("\n🔮 Step 5: Testing validation prediction")
        
        try:
            from backend.utils.feature_metadata import FeatureMetadataManager
            
            # Load feature metadata
            feature_metadata = FeatureMetadataManager.load_feature_metadata(
                dataset_name=dataset_name, 
                model_path="./exports/trained_model.keras"
            )
            
            if feature_metadata:
                print("✅ Feature metadata loaded")
                print(f"Features: {feature_metadata.get('feature_names')}")
                print(f"Types: {feature_metadata.get('feature_types')}")
                
                # Test input processing
                test_input = {'feature1': 3, 'feature2': 300, 'feature3': 5}
                print(f"Test input: {test_input}")
                
                input_array = FeatureMetadataManager.process_validation_input(test_input, feature_metadata)
                print(f"Processed input (before scaler): {input_array}")
                
                # Apply scaler if exists
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    scaled_input = scaler.transform(input_array)
                    print(f"Processed input (after scaler): {scaled_input}")
                else:
                    print("❌ No scaler found for validation")
                    
            else:
                print("❌ No feature metadata found")
                
        except Exception as e:
            print(f"❌ Error in validation test: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_scaling_pipeline() 