#!/usr/bin/env python3
"""Debug script to understand model input/output scaling"""

import tensorflow as tf
import numpy as np
import pickle

def debug_model_scaling():
    """Debug the model scaling issue"""
    
    print("🔍 DEBUGGING MODEL SCALING")
    print("=" * 45)
    
    # Enable unsafe deserialization
    tf.keras.config.enable_unsafe_deserialization()
    
    # Load model
    model = tf.keras.models.load_model('./exports/trained_model.keras')
    print(f"✅ Model loaded")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    
    # Load scaler
    scaler_path = './sessions/57355be7-f419-41a3-8e95-dda3a6407cd5/datasets/cereal_scaler.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"✅ Scaler loaded")
    print(f"   Scaler mean: {scaler.mean_}")
    print(f"   Scaler scale: {scaler.scale_}")
    
    # Load actual training data point to test
    data = np.load('./sessions/57355be7-f419-41a3-8e95-dda3a6407cd5/datasets/cereal.npz')
    X_original = data['X']  # This should be raw unscaled data as saved
    y_original = data['y']  # This should be raw targets (50-160 calories)
    
    print(f"\\n📊 Original dataset:")
    print(f"   X shape: {X_original.shape}")
    print(f"   y shape: {y_original.shape}")
    print(f"   X sample: {X_original[0]}")
    print(f"   y sample: {y_original[0]} (target calories)")
    
    # Test 1: Use first data point
    test_x_raw = X_original[0:1].astype(np.float32)  # First row, keep as 2D
    test_y_actual = y_original[0]  # Actual target
    
    print(f"\\n🧪 Test 1: First training data point")
    print(f"   Raw input: {test_x_raw}")
    print(f"   Actual target: {test_y_actual} calories")
    
    # Predict with raw input
    pred_raw = model.predict(test_x_raw, verbose=0)
    print(f"   Prediction (raw input): {pred_raw[0][0]:.2f}")
    
    # Apply scaling and predict
    test_x_scaled = scaler.transform(test_x_raw)
    pred_scaled = model.predict(test_x_scaled, verbose=0)
    print(f"   Scaled input: {test_x_scaled}")
    print(f"   Prediction (scaled input): {pred_scaled[0][0]:.2f}")
    
    print(f"\\n📈 Analysis:")
    print(f"   If model expects SCALED input: prediction should be ~{test_y_actual}")
    print(f"   If model expects RAW input: prediction should be ~{test_y_actual}")
    print(f"   ")
    print(f"   Raw input prediction: {pred_raw[0][0]:.2f}")
    print(f"   Scaled input prediction: {pred_scaled[0][0]:.2f}")
    print(f"   ")
    if abs(pred_scaled[0][0] - test_y_actual) < abs(pred_raw[0][0] - test_y_actual):
        print(f"   ✅ Model expects SCALED input (closer to target)")
    else:
        print(f"   ✅ Model expects RAW input (closer to target)")
    
    # Test 2: Try a few more data points
    print(f"\\n🧪 Test 2: Multiple data points")
    for i in range(min(3, len(X_original))):
        x_test = X_original[i:i+1].astype(np.float32)
        y_test = y_original[i]
        
        pred_raw = model.predict(x_test, verbose=0)[0][0]
        pred_scaled = model.predict(scaler.transform(x_test), verbose=0)[0][0]
        
        print(f"   Point {i+1}: target={y_test}, raw_pred={pred_raw:.2f}, scaled_pred={pred_scaled:.2f}")
    
    # Test 3: Check what range the model outputs
    print(f"\\n🧪 Test 3: Model output range analysis")
    # Use multiple test points
    test_indices = range(min(10, len(X_original)))
    raw_predictions = []
    scaled_predictions = []
    
    for i in test_indices:
        x_test = X_original[i:i+1].astype(np.float32)
        raw_pred = model.predict(x_test, verbose=0)[0][0]
        scaled_pred = model.predict(scaler.transform(x_test), verbose=0)[0][0]
        raw_predictions.append(raw_pred)
        scaled_predictions.append(scaled_pred)
    
    print(f"   Raw input predictions: min={min(raw_predictions):.2f}, max={max(raw_predictions):.2f}, avg={np.mean(raw_predictions):.2f}")
    print(f"   Scaled input predictions: min={min(scaled_predictions):.2f}, max={max(scaled_predictions):.2f}, avg={np.mean(scaled_predictions):.2f}")
    print(f"   Actual targets: min={y_original.min()}, max={y_original.max()}, avg={y_original.mean():.2f}")
    
    print(f"\\n🎯 CONCLUSION:")
    raw_avg_error = np.mean([abs(raw_predictions[i] - y_original[i]) for i in test_indices])
    scaled_avg_error = np.mean([abs(scaled_predictions[i] - y_original[i]) for i in test_indices])
    
    print(f"   Average error with raw inputs: {raw_avg_error:.2f}")
    print(f"   Average error with scaled inputs: {scaled_avg_error:.2f}")
    
    if scaled_avg_error < raw_avg_error:
        print(f"   ✅ VALIDATION SHOULD USE SCALED INPUTS")
        print(f"   ❌ Current issue: Validation might not be applying scaling correctly")
    else:
        print(f"   ✅ VALIDATION SHOULD USE RAW INPUTS") 
        print(f"   ❌ Current issue: Validation is incorrectly applying scaling")

if __name__ == "__main__":
    debug_model_scaling() 