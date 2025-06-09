"""
Generic feature metadata manager for custom datasets.
This handles feature information extraction, encoding, and validation for any custom dataset.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flask import current_app
from backend.utils.logging import get_logger

logger = get_logger(__name__)

class FeatureMetadataManager:
    """Manages feature metadata for custom datasets during training and validation."""
    
    @staticmethod
    def extract_and_save_feature_metadata(dataset_name, X, y, metadata_dict, model_path):
        """
        Extract and save feature metadata from dataset during training.
        
        Args:
            dataset_name (str): Name of the dataset
            X (np.ndarray): Feature data
            y (np.ndarray): Target data  
            metadata_dict (dict): Original dataset metadata
            model_path (str): Path where the trained model is saved
            
        Returns:
            dict: Complete feature metadata for validation
        """
        try:
            logger.info(f"Extracting feature metadata for dataset: {dataset_name}")
            
            # Get basic info from original metadata
            feature_names = metadata_dict.get('feature_columns', [])
            feature_types = metadata_dict.get('feature_types', [])
            task_type = metadata_dict.get('task_type', 'classification')
            target_column = metadata_dict.get('target_column', 'target')
            
            # Auto-detect feature types if not provided
            if not feature_types or len(feature_types) != len(feature_names):
                feature_types = FeatureMetadataManager._auto_detect_feature_types(X, feature_names)
                logger.info(f"Auto-detected feature types: {dict(zip(feature_names, feature_types))}")
            
            # Handle categorical features and create encodings
            categorical_encodings = {}
            
            # First, try to get original categorical mappings from dataset metadata
            original_categorical_mappings = metadata_dict.get('categorical_mappings', {})
            
            for i, (feature_name, feature_type) in enumerate(zip(feature_names, feature_types)):
                if feature_type == 'categorical':
                    # Try to use original categorical mapping if available
                    if feature_name in original_categorical_mappings:
                        original_mapping = original_categorical_mappings[feature_name]
                        categorical_encodings[feature_name] = {
                            'values': original_mapping['original_values'],
                            'encoding': original_mapping['value_to_code']
                        }
                        logger.info(f"Using preserved categorical mapping for '{feature_name}': {categorical_encodings[feature_name]}")
                    else:
                        # Fallback: try to infer from data
                        unique_encoded_values = np.unique(X[:, i])
                        
                        # Double-check: if categorical feature has floating point values that look continuous,
                        # treat it as numeric instead
                        if not np.all(unique_encoded_values == unique_encoded_values.astype(int)):
                            # Check if values look like continuous data
                            value_range = np.max(unique_encoded_values) - np.min(unique_encoded_values)
                            # If range is large OR values are clearly continuous floats, treat as numeric
                            if value_range > 1.0 or len(unique_encoded_values) > 5:
                                logger.warning(f"Feature '{feature_name}' was marked categorical but looks continuous (range: {value_range:.6f}, unique values: {len(unique_encoded_values)}). Treating as float.")
                                feature_types[i] = 'float'
                                continue
                            # For binary features with floating point values, keep as categorical but note the issue
                            elif len(unique_encoded_values) == 2:
                                logger.info(f"Feature '{feature_name}' has 2 floating-point values {unique_encoded_values} - keeping as categorical for binary classification.")
                        
                        # Create reasonable categorical values since original mapping is lost
                        if len(unique_encoded_values) == 2:
                            # Binary categorical (common case)
                            if feature_name.lower() in ['type', 'category', 'class']:
                                original_values = ["Hot", "Cold"]  # Common cereal types
                            elif feature_name.lower() in ['gender', 'sex']:
                                original_values = ["Male", "Female"]
                            else:
                                original_values = [f"{feature_name.title()}_{int(val)}" for val in unique_encoded_values]
                        else:
                            # Multi-categorical
                            original_values = [f"{feature_name.title()}_{int(val)}" for val in unique_encoded_values]
                        
                        # Create mapping from original value to encoded value
                        categorical_encodings[feature_name] = {
                            'values': original_values,
                            'encoding': {str(original_values[i]): int(unique_encoded_values[i]) 
                                       for i in range(len(original_values))}
                        }
                        logger.warning(f"No original mapping found for '{feature_name}', created fallback mapping: {categorical_encodings[feature_name]}")
            
            # Handle target encoding for classification
            target_encoding = None
            class_labels = None
            if task_type == 'classification':
                # Handle both one-hot encoded and regular encoded targets
                if len(y.shape) > 1 and y.shape[1] > 1:
                    # One-hot encoded target - get number of classes from shape
                    num_classes = y.shape[1]
                    unique_targets = list(range(num_classes))
                    logger.info(f"Detected one-hot encoded target with {num_classes} classes")
                else:
                    # Regular encoded target
                    unique_targets = np.unique(y)
                    num_classes = len(unique_targets)
                
                # Try to use original class labels from dataset metadata first
                original_class_labels = metadata_dict.get('class_labels', None)
                
                if original_class_labels and len(original_class_labels) == num_classes:
                    # Use the original meaningful class labels
                    class_labels = original_class_labels
                    target_encoding = {int(i): label for i, label in enumerate(class_labels)}
                    logger.info(f"Using original class labels: {class_labels} for {num_classes} classes")
                else:
                    # Fallback to generic labels if original labels not available or don't match
                    if num_classes > 2:
                        # Multi-class classification
                        class_labels = [f"class_{i}" for i in range(num_classes)]
                        target_encoding = {int(i): label for i, label in enumerate(class_labels)}
                    else:
                        # Binary classification
                        class_labels = ["class_0", "class_1"]
                        target_encoding = {0: class_labels[0], 1: class_labels[1]}
                    logger.warning(f"Using generic class labels: {class_labels}. Original labels: {original_class_labels}, Detected classes: {num_classes}")
            
            # Create complete feature metadata
            feature_metadata = {
                'dataset_name': dataset_name,
                'feature_names': feature_names,
                'feature_types': feature_types,
                'categorical_encodings': categorical_encodings,
                'task_type': task_type,
                'target_column': target_column,
                'class_labels': class_labels,
                'target_encoding': target_encoding,
                'input_shape': list(X.shape[1:]),
                'num_features': len(feature_names),
                'num_classes': len(class_labels) if class_labels else None,
                'created_during_training': True
            }
            
            # Save feature metadata alongside model
            features_path = model_path.replace('.keras', '_features.json').replace('.h5', '_features.json')
            with open(features_path, 'w') as f:
                json.dump(feature_metadata, f, indent=2)
            
            logger.info(f"Feature metadata saved to: {features_path}")
            return feature_metadata
            
        except Exception as e:
            logger.error(f"Error extracting feature metadata: {str(e)}", exc_info=True)
            return None
    
    @staticmethod
    def _auto_detect_feature_types(X, feature_names):
        """Auto-detect feature types from data."""
        feature_types = []
        for i in range(X.shape[1]):
            column_data = X[:, i]
            unique_values = np.unique(column_data)
            
            # Check if all values are integers (no decimal parts)
            is_integer = np.all(column_data == column_data.astype(int))
            
            # For categorical detection, be more strict:
            # Only consider as categorical if:
            # 1. All values are integers AND
            # 2. There are very few unique values (≤ 5) AND  
            # 3. Values are small integers (absolute value ≤ 100)
            if (is_integer and 
                len(unique_values) <= 5 and 
                np.all(np.abs(unique_values) <= 100)):
                feature_types.append('categorical')
            # If all values are integers but too many unique values or large numbers, treat as int
            elif is_integer:
                feature_types.append('int')
            # Otherwise, consider as float (this includes continuous numeric data)
            else:
                feature_types.append('float')
        
        return feature_types
    
    @staticmethod
    def load_feature_metadata(dataset_name=None, model_path=None):
        """
        Load feature metadata for a dataset.
        
        Args:
            dataset_name (str, optional): Name of the dataset
            model_path (str, optional): Path to the model file
            
        Returns:
            dict: Feature metadata or None if not found
        """
        try:
            # Try to load from model path first
            if model_path:
                features_path = model_path.replace('.keras', '_features.json').replace('.h5', '_features.json')
                if os.path.exists(features_path):
                    with open(features_path, 'r') as f:
                        return json.load(f)
            
            # Fall back to dataset-specific metadata if dataset_name provided
            if dataset_name:
                try:
                    from backend.utils.session_manager import get_session_datasets_dir, get_session_id
                    session_id = get_session_id()
                    datasets_dir = get_session_datasets_dir(session_id)
                    metadata_path = os.path.join(datasets_dir, f'{dataset_name}_metadata.json')
                    
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            original_metadata = json.load(f)
                        
                        # Convert to feature metadata format
                        return FeatureMetadataManager._convert_original_metadata(original_metadata)
                except Exception as e:
                    logger.warning(f"Could not load session-specific metadata: {str(e)}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading feature metadata: {str(e)}", exc_info=True)
            return None
    
    @staticmethod
    def _convert_original_metadata(original_metadata):
        """Convert original dataset metadata to feature metadata format."""
        feature_names = original_metadata.get('feature_columns', [])
        feature_types = original_metadata.get('feature_types', ['float'] * len(feature_names))
        task_type = original_metadata.get('task_type', 'classification')
        class_labels = original_metadata.get('class_labels', None)
        
        # Convert categorical mappings to feature metadata format
        categorical_encodings = {}
        original_categorical_mappings = original_metadata.get('categorical_mappings', {})
        
        for feature_name, mapping in original_categorical_mappings.items():
            if 'original_values' in mapping and 'value_to_code' in mapping:
                categorical_encodings[feature_name] = {
                    'values': mapping['original_values'],
                    'encoding': mapping['value_to_code']
                }
        
        return {
            'dataset_name': original_metadata.get('name', 'unknown'),
            'feature_names': feature_names,
            'feature_types': feature_types,
            'categorical_encodings': categorical_encodings,  # Now properly populated
            'task_type': task_type,
            'target_column': original_metadata.get('target_column', 'target'),
            'class_labels': class_labels,
            'target_encoding': None,
            'input_shape': original_metadata.get('processed_shape', [0])[-1:],
            'num_features': len(feature_names),
            'num_classes': len(class_labels) if class_labels else None,
            'created_during_training': False
        }
    
    @staticmethod
    def process_validation_input(input_data, feature_metadata):
        """
        Process validation input data according to feature metadata.
        
        Args:
            input_data (dict): Raw input data from frontend
            feature_metadata (dict): Feature metadata from training
            
        Returns:
            np.ndarray: Processed input array ready for prediction
        """
        try:
            feature_names = feature_metadata['feature_names']
            feature_types = feature_metadata['feature_types']
            categorical_encodings = feature_metadata.get('categorical_encodings', {})
            
            processed_values = []
            
            for i, feature_name in enumerate(feature_names):
                if feature_name not in input_data:
                    raise ValueError(f"Missing feature: {feature_name}")
                
                raw_value = input_data[feature_name]
                feature_type = feature_types[i] if i < len(feature_types) else 'float'
                
                # Process based on feature type
                if feature_type == 'categorical':
                    if feature_name in categorical_encodings:
                        encoding_map = categorical_encodings[feature_name]['encoding']
                        available_values = categorical_encodings[feature_name]['values']
                        
                        # Try exact string match first
                        if str(raw_value) in encoding_map:
                            processed_values.append(float(encoding_map[str(raw_value)]))
                        else:
                            # Try fuzzy matching for floating point values
                            try:
                                input_float = float(raw_value)
                                best_match = None
                                min_diff = float('inf')
                                
                                for available_val in available_values:
                                    try:
                                        available_float = float(available_val)
                                        diff = abs(input_float - available_float)
                                        if diff < min_diff:
                                            min_diff = diff
                                            best_match = available_val
                                    except ValueError:
                                        continue
                                
                                # If we found a close match (within 1% or 0.01 absolute difference)
                                if best_match is not None and (min_diff < 0.01 or min_diff / abs(input_float) < 0.01):
                                    processed_values.append(float(encoding_map[str(best_match)]))
                                else:
                                    raise ValueError(
                                        f"Invalid value '{raw_value}' for categorical feature '{feature_name}'. "
                                        f"Valid values: {list(encoding_map.keys())}. "
                                        f"Closest match: {best_match} (difference: {min_diff:.6f})"
                                    )
                            except ValueError as ve:
                                if "Invalid value" in str(ve):
                                    raise ve
                                else:
                                    raise ValueError(
                                        f"Invalid value '{raw_value}' for categorical feature '{feature_name}'. "
                                        f"Valid values: {list(encoding_map.keys())}"
                                    )
                    else:
                        # Try to convert to number directly
                        try:
                            processed_values.append(float(raw_value))
                        except ValueError:
                            raise ValueError(f"Could not process categorical value '{raw_value}' for feature '{feature_name}'")
                
                elif feature_type in ['int', 'float']:
                    try:
                        processed_values.append(float(raw_value))
                    except ValueError:
                        raise ValueError(f"Could not convert '{raw_value}' to number for feature '{feature_name}'")
                
                else:
                    # Default: try to convert to float
                    try:
                        processed_values.append(float(raw_value))
                    except ValueError:
                        raise ValueError(f"Could not process value '{raw_value}' for feature '{feature_name}'")
            
            # Convert to numpy array and reshape for model input
            input_array = np.array(processed_values).reshape(1, -1)
            return input_array
            
        except Exception as e:
            logger.error(f"Error processing validation input: {str(e)}")
            raise
    
    @staticmethod
    def format_prediction_result(prediction, feature_metadata):
        """
        Format prediction result based on task type and metadata.
        
        Args:
            prediction (np.ndarray): Raw model prediction
            feature_metadata (dict): Feature metadata
            
        Returns:
            dict: Formatted prediction result
        """
        try:
            task_type = feature_metadata.get('task_type', 'classification')
            class_labels = feature_metadata.get('class_labels', None)
            target_encoding = feature_metadata.get('target_encoding', None)
            
            if task_type == 'classification':
                if prediction.shape[1] == 1:
                    # Binary classification
                    pred_value = float(prediction[0][0])
                    pred_class = int(pred_value > 0.5)
                    confidence = pred_value if pred_class else 1 - pred_value
                    
                    # Get class label if available
                    if class_labels and pred_class < len(class_labels):
                        label = class_labels[pred_class]
                    elif target_encoding and pred_class in target_encoding:
                        label = target_encoding[pred_class]
                    else:
                        label = f"class_{pred_class}"
                    
                    return {
                        'prediction': pred_class,
                        'prediction_label': label,
                        'confidence': confidence,
                        'raw_prediction': pred_value
                    }
                else:
                    # Multi-class classification
                    pred_class = int(np.argmax(prediction[0]))
                    confidence = float(prediction[0][pred_class])
                    
                    # Get class label if available
                    if class_labels and pred_class < len(class_labels):
                        label = class_labels[pred_class]
                    elif target_encoding and pred_class in target_encoding:
                        label = target_encoding[pred_class]
                    else:
                        label = f"class_{pred_class}"
                    
                    return {
                        'prediction': pred_class,
                        'prediction_label': label,
                        'confidence': confidence,
                        'class_probabilities': prediction[0].tolist()
                    }
            
            else:  # Regression
                pred_value = float(prediction[0][0] if prediction.shape[1] == 1 else prediction[0])
                return {
                    'prediction': pred_value,
                    'confidence': None  # No confidence for regression
                }
                
        except Exception as e:
            logger.error(f"Error formatting prediction result: {str(e)}")
            # Return raw prediction as fallback
            return {
                'prediction': float(prediction[0][0]) if prediction.shape[1] == 1 else prediction[0].tolist(),
                'confidence': None,
                'error': f"Could not format prediction: {str(e)}"
            }
    
    @staticmethod
    def check_feature_types_quality(feature_metadata):
        """
        Check if the current feature types are optimal and suggest improvements.
        
        Args:
            feature_metadata (dict): Feature metadata to check
            
        Returns:
            dict: Quality assessment and recommendations
        """
        issues = []
        recommendations = []
        
        feature_names = feature_metadata.get('feature_names', [])
        feature_types = feature_metadata.get('feature_types', [])
        categorical_encodings = feature_metadata.get('categorical_encodings', {})
        
        for i, (name, ftype) in enumerate(zip(feature_names, feature_types)):
            if ftype == 'categorical' and name in categorical_encodings:
                values = categorical_encodings[name]['values']
                
                # Check if categorical feature has floating point values
                try:
                    float_values = [float(v) for v in values]
                    if not all(v == int(v) for v in float_values):
                        # Check if it looks like continuous data
                        value_range = max(float_values) - min(float_values)
                        if value_range > 1.0 or len(values) > 5:
                            issues.append(f"Feature '{name}' is marked as categorical but contains continuous floating-point values")
                            recommendations.append(f"Consider retraining with '{name}' as a numeric (float) feature")
                except ValueError:
                    pass  # Not numeric, truly categorical
        
        return {
            'has_issues': len(issues) > 0,
            'issues': issues,
            'recommendations': recommendations,
            'overall_quality': 'poor' if len(issues) > 2 else 'fair' if len(issues) > 0 else 'good'
        } 