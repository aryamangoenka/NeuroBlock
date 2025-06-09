#!/usr/bin/env python3
"""Direct test of scaling logic without Flask context"""

import pandas as pd
import numpy as np
import os
import tempfile
import pickle
from sklearn.preprocessing import StandardScaler

def test_scaling_directly():
    """Test scaling directly without Flask context"""
    
    print("🧪 DIRECT SCALING TEST")
    print("=" * 40)
    
    # 1. Load test data
    print("\n📊 Step 1: Loading test data")
    df = pd.read_csv('test_scaling.csv')
    
    feature_columns = ['feature1', 'feature2', 'feature3']
    target_column = 'target'
    
    X = df[feature_columns].values
    y = df[target_column].values
    
    print("Original data ranges:")
    for i, col in enumerate(feature_columns):
        print(f"  {col}: {X[:, i].min():.2f} - {X[:, i].max():.2f} (range: {X[:, i].max() - X[:, i].min():.2f})")
    print(f"  {target_column}: {y.min():.2f} - {y.max():.2f}")
    
    print(f"\nFeature scaling differences:")
    ranges = [X[:, i].max() - X[:, i].min() for i in range(X.shape[1])]
    max_range = max(ranges)
    for i, col in enumerate(feature_columns):
        ratio = ranges[i] / max_range
        print(f"  {col}: {ratio:.3f} (range {ranges[i]:.0f} vs max {max_range:.0f})")
    
    # 2. Apply scaling like in dataset_loader.py
    print("\n🔧 Step 2: Applying StandardScaler")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("After scaling:")
    print(f"  Means: {np.mean(X_scaled, axis=0)}")
    print(f"  Stds: {np.std(X_scaled, axis=0)}")
    print(f"  Min values: {np.min(X_scaled, axis=0)}")
    print(f"  Max values: {np.max(X_scaled, axis=0)}")
    
    # 3. Test validation input processing
    print("\n🔮 Step 3: Testing validation input processing")
    
    # Sample input values (original scale)
    test_input = {'feature1': 3, 'feature2': 300, 'feature3': 5}
    print(f"Test input (original scale): {test_input}")
    
    # Convert to array
    input_array = np.array([[test_input[col] for col in feature_columns]])
    print(f"Input array (original scale): {input_array}")
    
    # Apply scaler
    scaled_input = scaler.transform(input_array)
    print(f"Input array (after scaling): {scaled_input}")
    
    # 4. Show the problem - what happens without scaling?
    print("\n⚠️  Step 4: Showing the problem without scaling")
    
    # Simulate a simple linear model without scaling
    # Model weights would be very different for each feature due to range differences
    print("Without scaling, gradients would be:")
    
    # Simulate gradient magnitudes proportional to feature ranges  
    gradients_no_scaling = ranges
    print(f"  Gradient magnitudes: {gradients_no_scaling}")
    print(f"  Feature2 gradient is {gradients_no_scaling[1]/gradients_no_scaling[0]:.0f}x larger than feature1!")
    
    # With scaling, gradients would be more balanced
    scaled_ranges = [np.max(X_scaled[:, i]) - np.min(X_scaled[:, i]) for i in range(X_scaled.shape[1])]
    print(f"  After scaling, ranges are: {scaled_ranges}")
    
    # 5. Test the actual validation pipeline
    print("\n🔄 Step 5: Testing actual validation logic")
    
    # Save scaler temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
        pickle.dump(scaler, tmp_file)
        scaler_path = tmp_file.name
    
    print(f"Saved scaler to: {scaler_path}")
    
    # Load scaler and test
    with open(scaler_path, 'rb') as f:
        loaded_scaler = pickle.load(f)
    
    # Test different inputs to see scaling effect
    test_cases = [
        {'feature1': 1, 'feature2': 121, 'feature3': 1},    # Min values
        {'feature1': 4, 'feature2': 485, 'feature3': 9},    # Max values  
        {'feature1': 3, 'feature2': 300, 'feature3': 5},    # Middle values
    ]
    
    for i, test_case in enumerate(test_cases):
        input_arr = np.array([[test_case[col] for col in feature_columns]])
        scaled_arr = loaded_scaler.transform(input_arr)
        print(f"  Test case {i+1}: {test_case}")
        print(f"    Original: {input_arr[0]}")
        print(f"    Scaled:   {scaled_arr[0]}")
    
    # Clean up
    os.unlink(scaler_path)
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_scaling_directly() 