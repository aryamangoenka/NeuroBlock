#!/usr/bin/env python3
"""Final scaling test using known session directory"""

import sys
sys.path.append('.')

import json
import pickle
import numpy as np
import tensorflow as tf
import os

def test_scaling_final():
    """Test scaling using the known session directory"""
    
    print("🧪 FINAL SCALING TEST")
    print("=" * 40)
    
    # Use the known session directory
    session_dir = "./sessions/57355be7-f419-41a3-8e95-dda3a6407cd5/datasets"
    model_path = "./exports/trained_model.keras"
    
    print(f"📁 Session dir: {session_dir}")
    print(f"🤖 Model path: {model_path}")
    
    # Check if files exist
    if not os.path.exists(session_dir):
        print(f"❌ Session directory not found: {session_dir}")
        return
        
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Find scaler and metadata files
    scaler_file = None
    metadata_file = None
    
    for file in os.listdir(session_dir):
        if file.endswith('_scaler.pkl'):
            scaler_file = os.path.join(session_dir, file)
            dataset_name = file.replace('_scaler.pkl', '')
            metadata_file = os.path.join(session_dir, f'{dataset_name}_metadata.json')
            break
    
    if not scaler_file:
        print("❌ No scaler file found")
        return
        
    if not os.path.exists(metadata_file):
        print(f"❌ Metadata file not found: {metadata_file}")
        return
    
    print(f"✅ Found scaler: {scaler_file}")
    print(f"✅ Found metadata: {metadata_file}")
    
    # Load scaler
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"📈 Scaler mean: {scaler.mean_}")
    print(f"📈 Scaler scale: {scaler.scale_}")
    print(f"📈 Number of features: {len(scaler.scale_)}")
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    feature_columns = metadata.get('feature_columns', [])
    task_type = metadata.get('task_type', 'unknown')
    class_labels = metadata.get('class_labels', None)
    
    print(f"📋 Dataset: {dataset_name}")
    print(f"📋 Task type: {task_type}")
    print(f"📋 Features: {feature_columns}")
    print(f"📋 Class labels: {class_labels}")
    
    # Test scaling manually
    print(f"\\n🧪 Testing scaling manually...")
    
    if dataset_name == 'cereal':
        # Create test input based on actual features found
        print(f"📋 Actual features: {feature_columns}")
        
        # Create test input for the actual features found
        test_input = {}
        for feature in feature_columns:
            if feature == 'protein':
                test_input[feature] = 4
            elif feature == 'fat':
                test_input[feature] = 1
            elif feature == 'sodium':
                test_input[feature] = 340  # HIGH sodium (this was the scaling problem!)
            elif feature == 'sugars':
                test_input[feature] = 6
            elif feature == 'potass':
                test_input[feature] = 300  # High potassium
            elif feature == 'vitamins':
                test_input[feature] = 100  # Vitamins
            else:
                test_input[feature] = 5  # Default value
        
        print(f"🔍 Test input: {test_input}")
        
        # Convert to array in the correct order
        input_array = np.array([[test_input[col] for col in feature_columns]])
        print(f"🔍 Raw input array: {input_array}")
        
        # Apply scaling
        scaled_input = scaler.transform(input_array)
        print(f"🔍 Scaled input: {scaled_input}")
        
        # Show the scaling effect
        print(f"\\n📊 Scaling analysis:")
        for i, col in enumerate(feature_columns):
            original = input_array[0][i]
            scaled = scaled_input[0][i]
            scale_factor = scaler.scale_[i]
            mean = scaler.mean_[i]
            
            print(f"  {col:8s}: {original:6.1f} -> {scaled:6.3f} (scale={scale_factor:.2f}, mean={mean:.1f})")
        
        # Now test prediction
        print(f"\\n🔮 Testing prediction...")
        
        try:
            # Enable unsafe deserialization for custom layers
            tf.keras.config.enable_unsafe_deserialization()
            
            # Load model
            model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded successfully")
            
            # Predict with scaled input
            prediction = model.predict(scaled_input, verbose=0)
            print(f"🎯 Raw prediction: {prediction}")
            
            if task_type == 'regression':
                predicted_value = prediction[0][0] if prediction.shape[1] == 1 else prediction[0]
                print(f"🎯 Predicted {metadata.get('target_column', 'target')}: {predicted_value:.2f}")
            else:
                predicted_class = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class]
                class_name = class_labels[predicted_class] if class_labels else f"class_{predicted_class}"
                print(f"🎯 Predicted class: {class_name} (confidence: {confidence:.3f})")
            
            # Now test what happens WITHOUT scaling (the problem!)
            print(f"\\n⚠️  Testing WITHOUT scaling (the original problem)...")
            prediction_no_scaling = model.predict(input_array, verbose=0)
            print(f"💥 Raw prediction (no scaling): {prediction_no_scaling}")
            
            if task_type == 'regression':
                predicted_no_scaling = prediction_no_scaling[0][0] if prediction_no_scaling.shape[1] == 1 else prediction_no_scaling[0]
                print(f"💥 Predicted {metadata.get('target_column', 'target')} (no scaling): {predicted_no_scaling:.2f}")
                
                print(f"\\n📊 SCALING IMPACT:")
                print(f"  With scaling:    {predicted_value:.2f}")
                print(f"  Without scaling: {predicted_no_scaling:.2f}")
                print(f"  Difference:      {abs(predicted_value - predicted_no_scaling):.2f}")
            
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            
    print(f"\\n✅ Test completed!")

if __name__ == "__main__":
    test_scaling_final() 