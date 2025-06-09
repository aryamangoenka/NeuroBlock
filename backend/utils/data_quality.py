"""
Enhanced data quality validation and feature type detection utilities.
This module provides improved ML processing for custom datasets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from backend.utils.logging import get_logger

logger = get_logger(__name__)

class DataQualityValidator:
    """Enhanced data quality validation and feature analysis."""
    
    @staticmethod
    def detect_feature_type(column_data: pd.Series, column_name: str) -> str:
        """
        Enhanced feature type detection using ML-based heuristics.
        
        Args:
            column_data (pd.Series): Column data to analyze
            column_name (str): Name of the column for logging
            
        Returns:
            str: Detected feature type ('categorical', 'numeric', 'boolean', 'datetime', 'text')
        """
        try:
            # Handle missing values
            non_null_data = column_data.dropna()
            if len(non_null_data) == 0:
                logger.warning(f"Column '{column_name}' has no non-null values")
                return 'other'
            
            unique_values = non_null_data.unique()
            unique_count = len(unique_values)
            total_count = len(non_null_data)
            unique_ratio = unique_count / total_count
            
            # Boolean detection
            if column_data.dtype == 'bool' or unique_count == 2:
                # Check if binary values look boolean
                if set(str(v).lower() for v in unique_values if pd.notna(v)) <= {
                    'true', 'false', '1', '0', 'yes', 'no', 't', 'f', 'y', 'n'
                }:
                    logger.debug(f"Column '{column_name}' detected as boolean (unique values: {unique_values})")
                    return 'boolean'
            
            # Datetime detection
            if column_data.dtype.kind in 'Mm':  # datetime types
                logger.debug(f"Column '{column_name}' detected as datetime")
                return 'datetime'
            
            # String/object type analysis
            if column_data.dtype == 'object':
                # Check if it's actually numeric strings
                try:
                    numeric_conversion = pd.to_numeric(non_null_data, errors='coerce')
                    if not numeric_conversion.isna().any():
                        # All values can be converted to numeric
                        if all(float(val).is_integer() for val in numeric_conversion):
                            logger.debug(f"Column '{column_name}' is numeric strings (integers)")
                            return 'int'
                        else:
                            logger.debug(f"Column '{column_name}' is numeric strings (floats)")
                            return 'float'
                except:
                    pass
                
                # Categorical vs text classification
                if unique_ratio < 0.05 or unique_count <= 20:
                    # Low unique ratio or small number of categories
                    avg_length = np.mean([len(str(val)) for val in unique_values])
                    if avg_length <= 50:  # Short strings are likely categorical
                        logger.debug(f"Column '{column_name}' detected as categorical (unique ratio: {unique_ratio:.3f}, avg length: {avg_length:.1f})")
                        return 'categorical'
                
                # High cardinality text
                avg_length = np.mean([len(str(val)) for val in non_null_data])
                if avg_length > 100 or unique_ratio > 0.8:
                    logger.debug(f"Column '{column_name}' detected as text (avg length: {avg_length:.1f}, unique ratio: {unique_ratio:.3f})")
                    return 'text'
                else:
                    logger.debug(f"Column '{column_name}' detected as categorical (object type)")
                    return 'categorical'
            
            # Numeric type analysis
            if column_data.dtype.kind in 'biufc':  # numeric types
                # Check if it's actually categorical despite being numeric
                if unique_ratio < 0.05 and unique_count <= 20:
                    # Check if values look like discrete categories
                    if all(isinstance(val, (int, np.integer)) for val in unique_values):
                        # Integer values that might be categories
                        value_range = max(unique_values) - min(unique_values)
                        if value_range <= 100:  # Small range of integers
                            logger.debug(f"Column '{column_name}' detected as categorical (numeric codes: {unique_values})")
                            return 'categorical'
                
                # Regular numeric detection
                if column_data.dtype.kind in 'iu':  # integers
                    # Check if all values are whole numbers
                    if all(float(val).is_integer() for val in non_null_data):
                        logger.debug(f"Column '{column_name}' detected as integer")
                        return 'int'
                    else:
                        logger.debug(f"Column '{column_name}' detected as float (integer type with decimals)")
                        return 'float'
                else:  # floats
                    logger.debug(f"Column '{column_name}' detected as float")
                    return 'float'
            
            # Fallback
            logger.warning(f"Column '{column_name}' type detection unclear, defaulting to 'other'")
            return 'other'
            
        except Exception as e:
            logger.error(f"Error detecting type for column '{column_name}': {str(e)}")
            return 'other'
    
    @staticmethod
    def validate_data_quality(X: np.ndarray, y: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive data quality validation.
        
        Args:
            X (np.ndarray): Feature data
            y (np.ndarray): Target data
            metadata (dict): Dataset metadata
            
        Returns:
            dict: Quality assessment with warnings and recommendations
        """
        warnings = []
        recommendations = []
        quality_score = 100  # Start with perfect score and deduct points
        
        try:
            feature_names = metadata.get('feature_columns', [])
            task_type = metadata.get('task_type', 'classification')
            categorical_mappings = metadata.get('categorical_mappings', {})
            
            # 1. Feature scaling analysis
            if X.shape[1] > 1:
                feature_ranges = [float(X[:, i].max() - X[:, i].min()) for i in range(X.shape[1])]
                feature_means = [float(np.mean(X[:, i])) for i in range(X.shape[1])]
                feature_stds = [float(np.std(X[:, i])) for i in range(X.shape[1])]
                
                # Check for extreme scaling differences
                max_range = float(max(feature_ranges)) if feature_ranges else 0.0
                min_range = float(min([r for r in feature_ranges if r > 0])) if any(r > 0 for r in feature_ranges) else 1.0
                
                if max_range / min_range > 1000:
                    warnings.append(f"Extreme feature scaling differences (max range: {max_range:.2f}, min range: {min_range:.2f})")
                    recommendations.append("Apply feature scaling (StandardScaler or MinMaxScaler) for better model performance")
                    quality_score -= 15
                elif max_range / min_range > 100:
                    warnings.append(f"Significant feature scaling differences detected")
                    recommendations.append("Consider feature scaling for optimal model performance")
                    quality_score -= 8
                
                # Check for features with zero variance
                zero_variance_features = [i for i, r in enumerate(feature_ranges) if r == 0]
                if zero_variance_features:
                    feature_names_zero = [feature_names[i] if i < len(feature_names) else f"feature_{i}" 
                                        for i in zero_variance_features]
                    warnings.append(f"Features with zero variance: {feature_names_zero}")
                    recommendations.append("Remove zero-variance features as they don't contribute to learning")
                    quality_score -= 20
            
            # 2. Categorical feature analysis
            high_cardinality_features = []
            for feature_name, mapping in categorical_mappings.items():
                cardinality = len(mapping.get('original_values', []))
                if cardinality > 50:
                    high_cardinality_features.append(f"{feature_name} ({cardinality} categories)")
                    quality_score -= 5
                elif cardinality > 20:
                    high_cardinality_features.append(f"{feature_name} ({cardinality} categories)")
                    quality_score -= 2
            
            if high_cardinality_features:
                warnings.append(f"High cardinality categorical features: {', '.join(high_cardinality_features)}")
                recommendations.append("Consider feature engineering (grouping rare categories, target encoding) for high cardinality features")
            
            # 3. Dataset size analysis
            n_samples, n_features = X.shape
            
            if n_samples < 100:
                warnings.append(f"Very small dataset ({n_samples} samples)")
                recommendations.append("Consider data augmentation or collecting more data")
                quality_score -= 25
            elif n_samples < 1000:
                warnings.append(f"Small dataset ({n_samples} samples)")
                recommendations.append("Monitor for overfitting, consider cross-validation")
                quality_score -= 10
            
            # 4. Feature-to-sample ratio
            if n_features / n_samples > 0.1:
                warnings.append(f"High feature-to-sample ratio ({n_features} features, {n_samples} samples)")
                recommendations.append("Consider feature selection or dimensionality reduction")
                quality_score -= 15
            
            # 5. Target analysis for classification
            if task_type == 'classification':
                if len(y.shape) > 1 and y.shape[1] > 1:
                    # One-hot encoded
                    class_counts = np.sum(y, axis=0)
                    num_classes = len(class_counts)
                else:
                    # Regular encoded
                    unique_targets, class_counts = np.unique(y, return_counts=True)
                    num_classes = len(unique_targets)
                
                # Class imbalance analysis
                min_class_count = int(min(class_counts))
                max_class_count = int(max(class_counts))
                imbalance_ratio = float(max_class_count / min_class_count) if min_class_count > 0 else float('inf')
                
                if imbalance_ratio > 10:
                    warnings.append(f"Severe class imbalance (ratio: {imbalance_ratio:.1f})")
                    recommendations.append("Consider class balancing techniques (SMOTE, class weights, resampling)")
                    quality_score -= 20
                elif imbalance_ratio > 3:
                    warnings.append(f"Moderate class imbalance (ratio: {imbalance_ratio:.1f})")
                    recommendations.append("Monitor model performance across all classes")
                    quality_score -= 8
                
                # Too many classes for small dataset
                if num_classes > n_samples / 50:
                    warnings.append(f"Many classes ({num_classes}) relative to dataset size")
                    recommendations.append("Consider grouping similar classes or collecting more data")
                    quality_score -= 10
            
            # 6. Target analysis for regression
            elif task_type == 'regression':
                target_std = float(np.std(y))
                target_mean = float(np.mean(y))
                target_range = float(np.max(y) - np.min(y))
                
                if target_std == 0:
                    warnings.append("Target variable has zero variance")
                    recommendations.append("Check if this should be a classification problem")
                    quality_score -= 30
                elif target_range / (abs(target_mean) + 1e-8) > 100:
                    warnings.append("Target variable has very wide range relative to mean")
                    recommendations.append("Consider target transformation (log, sqrt) for better model performance")
                    quality_score -= 10
            
            # 7. Missing value analysis (if metadata includes this info)
            missing_info = metadata.get('missing_percentages', {})
            high_missing_features = []
            for feature, missing_pct in missing_info.items():
                if missing_pct > 50:
                    high_missing_features.append(f"{feature} ({missing_pct:.1f}%)")
                    quality_score -= 15
                elif missing_pct > 20:
                    high_missing_features.append(f"{feature} ({missing_pct:.1f}%)")
                    quality_score -= 5
            
            if high_missing_features:
                warnings.append(f"Features with high missing values: {', '.join(high_missing_features)}")
                recommendations.append("Consider imputation strategies or feature removal for high-missing features")
            
            # Determine overall quality level
            if quality_score >= 90:
                quality_level = 'excellent'
            elif quality_score >= 75:
                quality_level = 'good'
            elif quality_score >= 60:
                quality_level = 'fair'
            elif quality_score >= 40:
                quality_level = 'poor'
            else:
                quality_level = 'critical'
            
            return {
                'quality_score': max(0, quality_score),
                'quality_level': quality_level,
                'warnings': warnings,
                'recommendations': recommendations,
                'analysis': {
                    'n_samples': int(n_samples),
                    'n_features': int(n_features),
                    'task_type': task_type,
                    'feature_ranges': feature_ranges if X.shape[1] > 1 else [],
                    'categorical_features': int(len(categorical_mappings)),
                    'high_cardinality_features': int(len(high_cardinality_features))
                }
            }
            
        except Exception as e:
            logger.error(f"Error in data quality validation: {str(e)}")
            return {
                'quality_score': 0,
                'quality_level': 'error',
                'warnings': [f"Quality analysis failed: {str(e)}"],
                'recommendations': ["Review data format and try again"],
                'analysis': {}
            }
    
    @staticmethod
    def validate_model_architecture(dataset_metadata: Dict[str, Any], model_architecture: Dict[str, Any]) -> Optional[str]:
        """
        Validate that model architecture suits the dataset.
        
        Args:
            dataset_metadata (dict): Dataset metadata
            model_architecture (dict): Model architecture
            
        Returns:
            str or None: Error message if validation fails, None if valid
        """
        try:
            task_type = dataset_metadata.get('task_type', 'classification')
            feature_columns = dataset_metadata.get('feature_columns', [])
            class_labels = dataset_metadata.get('class_labels', [])
            
            num_features = len(feature_columns)
            num_classes = len(class_labels) if class_labels else 1
            
            nodes = model_architecture.get('nodes', [])
            if not nodes:
                return "Model architecture is empty"
            
            # Find input and output layers
            input_layer = next((node for node in nodes if node.get('type') == 'input'), None)
            output_layer = next((node for node in nodes if node.get('type') == 'output'), None)
            
            if not input_layer:
                return "Model must have an input layer"
            
            if not output_layer:
                return "Model must have an output layer"
            
            # Validate input layer
            input_units = input_layer.get('params', {}).get('units')
            if input_units and input_units != num_features:
                return f"Input layer units ({input_units}) don't match number of features ({num_features})"
            
            # Validate output layer
            output_units = output_layer.get('params', {}).get('units')
            if output_units:
                if task_type == 'classification':
                    expected_output = num_classes
                    if output_units != expected_output:
                        return f"Output layer units ({output_units}) don't match number of classes ({expected_output})"
                elif task_type == 'regression':
                    if output_units != 1:
                        return f"Regression output layer should have 1 unit, got {output_units}"
            
            # Check for common architecture issues
            dense_layers = [node for node in nodes if node.get('type') == 'dense']
            if len(dense_layers) == 0:
                return "Model should have at least one dense layer"
            
            # Validate activation functions
            if task_type == 'classification' and output_layer:
                output_activation = output_layer.get('params', {}).get('activation', '').lower()
                if num_classes > 2 and output_activation not in ['softmax', 'none', '']:
                    logger.warning(f"Multi-class classification typically uses softmax activation, got '{output_activation}'")
                elif num_classes == 2 and output_activation not in ['sigmoid', 'softmax', 'none', '']:
                    logger.warning(f"Binary classification typically uses sigmoid activation, got '{output_activation}'")
            
            return None  # No errors found
            
        except Exception as e:
            logger.error(f"Error validating model architecture: {str(e)}")
            return f"Architecture validation error: {str(e)}"

def enhanced_feature_type_detection(df: pd.DataFrame, feature_columns: List[str]) -> Tuple[List[str], Dict[str, Any]]:
    """
    Enhanced feature type detection with quality assessment.
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature_columns (list): List of feature column names
        
    Returns:
        tuple: (feature_types, quality_info)
    """
    feature_types = []
    quality_info = {
        'type_changes': [],
        'warnings': [],
        'confidence_scores': []
    }
    
    validator = DataQualityValidator()
    
    for col in feature_columns:
        if col not in df.columns:
            feature_types.append('other')
            quality_info['warnings'].append(f"Column '{col}' not found in data")
            quality_info['confidence_scores'].append(0.0)
            continue
        
        detected_type = validator.detect_feature_type(df[col], col)
        original_pandas_type = str(df[col].dtype)
        
        # Calculate confidence based on various factors
        confidence = 0.8  # Base confidence
        
        # Increase confidence for clear cases
        unique_ratio = df[col].nunique() / len(df[col].dropna()) if len(df[col].dropna()) > 0 else 0
        
        if detected_type == 'categorical' and unique_ratio < 0.05:
            confidence += 0.15
        elif detected_type in ['int', 'float'] and df[col].dtype.kind in 'biufc':
            confidence += 0.1
        elif detected_type == 'boolean' and df[col].nunique() == 2:
            confidence += 0.15
        
        confidence = min(confidence, 1.0)
        
        feature_types.append(detected_type)
        quality_info['confidence_scores'].append(confidence)
        
        # Track type changes from pandas inference
        if ((original_pandas_type.startswith('int') and detected_type not in ['int', 'categorical']) or
            (original_pandas_type.startswith('float') and detected_type not in ['float', 'categorical']) or
            (original_pandas_type == 'object' and detected_type not in ['categorical', 'text'])):
            quality_info['type_changes'].append({
                'column': col,
                'original': original_pandas_type,
                'detected': detected_type,
                'confidence': confidence
            })
    
    return feature_types, quality_info 