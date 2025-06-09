#!/usr/bin/env python3
"""Test the prediction endpoint directly to see if scaling is working"""

import sys
sys.path.append('.')

import json
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask
from backend.api.routes import api_blueprint
from backend.utils.session_manager import get_session_datasets_dir, get_session_id
import os

def test_prediction_endpoint():
    """Test the prediction endpoint directly"""
    
    print("🧪 TESTING PREDICTION ENDPOINT")
    print("=" * 45)
    
    # Create Flask app
    app = Flask(__name__)
    app.config['PROJECT_ROOT'] = '.'
    app.config['TRAINED_MODEL_PATH'] = './exports/trained_model.keras'
    app.register_blueprint(api_blueprint)
    
    # Check if model exists
    model_path = './exports/trained_model.keras'
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Need to train a model first!")
        return
    
    print(f"✅ Model found at: {model_path}")
    
    with app.app_context():
        # Simulate a session context
        with app.test_request_context():
            # Try to get session datasets dir
            try:
                datasets_dir = get_session_datasets_dir()
                print(f"✅ Session datasets dir: {datasets_dir}")
                
                # Look for scalers
                scaler_files = []
                if os.path.exists(datasets_dir):
                    for file in os.listdir(datasets_dir):
                        if file.endswith('_scaler.pkl'):
                            scaler_files.append(file)
                
                print(f"📊 Found {len(scaler_files)} scaler files: {scaler_files}")
                
                if scaler_files:
                    # Test with the first scaler we find
                    scaler_file = scaler_files[0]
                    dataset_name = scaler_file.replace('_scaler.pkl', '')
                    scaler_path = os.path.join(datasets_dir, scaler_file)
                    
                    print(f"🧪 Testing with dataset: {dataset_name}")
                    print(f"🔧 Scaler path: {scaler_path}")
                    
                    # Load the scaler to see what it contains
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    
                    print(f"📈 Scaler mean: {scaler.mean_}")
                    print(f"📈 Scaler scale: {scaler.scale_}")
                    
                    # Get the dataset metadata
                    metadata_file = os.path.join(datasets_dir, f'{dataset_name}_metadata.json')
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        feature_columns = metadata.get('feature_columns', [])
                        print(f"📋 Features: {feature_columns}")
                        
                        # Create test inputs
                        if dataset_name == 'cereal':
                            # Test with different scaling scenarios
                            test_cases = [
                                {
                                    'protein': 1,      # Small value
                                    'fat': 5,         # Medium value  
                                    'sodium': 340,    # LARGE value (this should be scaled down)
                                    'fiber': 10,      # Medium value
                                    'carbo': 10,      # Medium value
                                    'sugars': 3       # Small value
                                },
                                {
                                    'protein': 6,      # Max value
                                    'fat': 0,         # Min value
                                    'sodium': 0,      # Min value (should be scaled)
                                    'fiber': 0,       # Min value
                                    'carbo': -1,      # Negative value
                                    'sugars': 15      # High value
                                }
                            ]
                            
                            for i, test_input in enumerate(test_cases):
                                print(f"\\n🔮 Test case {i+1}: {test_input}")
                                
                                # Manually test the scaling logic from the endpoint
                                from backend.utils.feature_metadata import FeatureMetadataManager
                                
                                # Load feature metadata
                                feature_metadata = FeatureMetadataManager.load_feature_metadata(
                                    dataset_name=dataset_name, 
                                    model_path=model_path
                                )
                                
                                if feature_metadata:
                                    # Process input
                                    input_array = FeatureMetadataManager.process_validation_input(test_input, feature_metadata)
                                    print(f"  Raw input array: {input_array}")
                                    
                                    # Apply scaler
                                    scaled_input = scaler.transform(input_array)
                                    print(f"  Scaled input array: {scaled_input}")
                                    
                                    # Load model and predict
                                    try:
                                        model = tf.keras.models.load_model(model_path)
                                        prediction = model.predict(scaled_input, verbose=0)
                                        print(f"  Raw prediction: {prediction}")
                                        
                                        # Format result
                                        result = FeatureMetadataManager.format_prediction_result(prediction, feature_metadata)
                                        print(f"  ✅ Final result: {result}")
                                        
                                    except Exception as e:
                                        print(f"  ❌ Prediction error: {str(e)}")
                                else:
                                    print(f"  ❌ No feature metadata found")
                        else:
                            print(f"⚠️  Unknown dataset: {dataset_name}")
                    else:
                        print(f"❌ No metadata file found: {metadata_file}")
                else:
                    print("❌ No scaler files found")
                    
            except Exception as e:
                print(f"❌ Session error: {str(e)}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    test_prediction_endpoint() 