#!/usr/bin/env python3
"""Direct test of prediction logic with scaling fix"""

import sys
sys.path.append('.')

import numpy as np
import tensorflow as tf
import os
import pickle

def test_direct_prediction():
    """Test the prediction logic directly"""
    
    print("🧪 DIRECT PREDICTION TEST (WITH SCALING FIX)")
    print("=" * 55)
    
    # Test input
    test_input = {
        'protein': 4,
        'fat': 1, 
        'sodium': 340,    # This was the scaling problem!
        'sugars': 6,
        'potass': 300,
        'vitamins': 100
    }
    
    print(f"🔍 Test input: {test_input}")
    
    # Manually apply the prediction logic from routes.py
    dataset_name = 'cereal'
    model_path = './exports/trained_model.keras'
    
    # Check for scaler (but don't apply it per our fix)
    builtin_datasets = ['Iris', 'MNIST', 'CIFAR-10', 'California Housing', 'Breast Cancer']
    session_dir = "./sessions/57355be7-f419-41a3-8e95-dda3a6407cd5/datasets"
    custom_scaler_path = os.path.join(session_dir, f'{dataset_name}_scaler.pkl')
    
    print(f"\\n🔧 Scaling logic (NEW FIX):")
    print(f"   Dataset: {dataset_name}")
    print(f"   Is built-in: {dataset_name in builtin_datasets}")
    print(f"   Scaler exists: {os.path.exists(custom_scaler_path)}")
    
    # Load feature metadata
    from backend.utils.feature_metadata import FeatureMetadataManager
    feature_metadata = FeatureMetadataManager.load_feature_metadata(
        dataset_name=dataset_name, 
        model_path=model_path
    )
    
    if not feature_metadata:
        print("❌ No feature metadata found")
        return
        
    print(f"✅ Feature metadata loaded")
    print(f"   Features: {feature_metadata.get('feature_names')}")
    
    # Process input data
    try:
        input_array = FeatureMetadataManager.process_validation_input(test_input, feature_metadata)
        print(f"✅ Input processed: {input_array}")
    except ValueError as ve:
        print(f"❌ Input processing error: {str(ve)}")
        return
    
    # Apply scaling logic with our fix
    if os.path.exists(custom_scaler_path) and dataset_name in builtin_datasets:
        print(f"   Applying scaling (built-in dataset)")
        with open(custom_scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        input_array = scaler.transform(input_array)
        print(f"   Scaled input: {input_array}")
    else:
        print(f"   🎯 SKIPPING scaling (custom dataset - model expects raw inputs)")
        print(f"   Raw input: {input_array}")
    
    # Load model and predict
    print(f"\\n🤖 Making prediction...")
    tf.keras.config.enable_unsafe_deserialization()
    model = tf.keras.models.load_model(model_path)
    prediction = model.predict(input_array, verbose=0)
    
    print(f"✅ Raw prediction: {prediction}")
    
    # Format result
    result = FeatureMetadataManager.format_prediction_result(prediction, feature_metadata)
    print(f"✅ Formatted result: {result}")
    
    # Analysis
    predicted_calories = result.get('prediction', 0)
    print(f"\\n📊 RESULT ANALYSIS:")
    print(f"   Predicted calories: {predicted_calories}")
    
    if 50 <= predicted_calories <= 200:
        print(f"   ✅ REALISTIC! (50-200 calories is normal for cereal)")
        print(f"   🎉 SCALING FIX SUCCESSFUL!")
    else:
        print(f"   ❌ Still unrealistic range")
    
    # Compare with old scaling approach
    print(f"\\n🔄 Comparison with old scaling approach:")
    if os.path.exists(custom_scaler_path):
        with open(custom_scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        scaled_input = scaler.transform(FeatureMetadataManager.process_validation_input(test_input, feature_metadata))
        old_prediction = model.predict(scaled_input, verbose=0)
        old_result = FeatureMetadataManager.format_prediction_result(old_prediction, feature_metadata)
        old_calories = old_result.get('prediction', 0)
        
        print(f"   Old approach (with scaling): {old_calories:.2f} calories")
        print(f"   New approach (no scaling): {predicted_calories:.2f} calories")
        print(f"   Improvement: {abs(old_calories - predicted_calories):.2f} calorie difference")

if __name__ == "__main__":
    test_direct_prediction() 