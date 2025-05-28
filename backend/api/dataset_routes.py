import os
import json
import pandas as pd
import numpy as np
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import tempfile
import shutil
from datetime import datetime
from backend.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# Create blueprint for dataset routes
dataset_blueprint = Blueprint('datasets', __name__, url_prefix='/api/datasets')

# Configuration constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_ROWS = 100000  # 100K rows
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

def get_file_size(file):
    """Get file size in bytes."""
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset to beginning
    return size

def safe_read_file(file, filename):
    """Safely read CSV or Excel file with proper error handling."""
    try:
        file_ext = os.path.splitext(filename.lower())[1]
        
        if file_ext == '.csv':
            # Try different encodings for CSV files
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=encoding, nrows=MAX_ROWS + 1)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Unable to decode CSV file with supported encodings")
                
        elif file_ext in ['.xlsx', '.xls']:
            file.seek(0)
            df = pd.read_excel(file, nrows=MAX_ROWS + 1)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
        return df
        
    except Exception as e:
        logger.error(f"Error reading file {filename}: {str(e)}")
        raise

def analyze_dataframe(df, filename):
    """Analyze dataframe and return comprehensive information."""
    try:
        # Check row limit
        if len(df) > MAX_ROWS:
            df = df.head(MAX_ROWS)
            row_limit_exceeded = True
        else:
            row_limit_exceeded = False
            
        # Basic info
        shape = df.shape
        columns = df.columns.tolist()
        
        # Data types analysis
        dtypes_info = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            # Categorize data types
            if dtype in ['int64', 'int32', 'float64', 'float32']:
                if df[col].dtype in ['int64', 'int32']:
                    dtypes_info[col] = 'integer'
                else:
                    dtypes_info[col] = 'float'
            elif dtype == 'object':
                # Check if it's categorical or text
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.1 and df[col].nunique() < 50:
                    dtypes_info[col] = 'categorical'
                else:
                    dtypes_info[col] = 'text'
            elif dtype == 'bool':
                dtypes_info[col] = 'boolean'
            else:
                dtypes_info[col] = 'other'
        
        # Missing values
        missing_values = df.isnull().sum().to_dict()
        missing_percentages = (df.isnull().sum() / len(df) * 100).to_dict()
        
        # Sample data (first 5 rows)
        sample_data = df.head(5).to_dict('records')
        
        # Basic statistics for numeric columns
        numeric_stats = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            try:
                stats = df[col].describe()
                numeric_stats[col] = {
                    'mean': float(stats['mean']) if not pd.isna(stats['mean']) else None,
                    'std': float(stats['std']) if not pd.isna(stats['std']) else None,
                    'min': float(stats['min']) if not pd.isna(stats['min']) else None,
                    'max': float(stats['max']) if not pd.isna(stats['max']) else None,
                    'median': float(stats['50%']) if not pd.isna(stats['50%']) else None
                }
            except Exception as e:
                logger.warning(f"Could not compute stats for column {col}: {str(e)}")
                numeric_stats[col] = None
        
        # Categorical column info
        categorical_info = {}
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            try:
                unique_values = df[col].value_counts().head(10).to_dict()
                categorical_info[col] = {
                    'unique_count': int(df[col].nunique()),
                    'top_values': {str(k): int(v) for k, v in unique_values.items()}
                }
            except Exception as e:
                logger.warning(f"Could not analyze categorical column {col}: {str(e)}")
                categorical_info[col] = None
        
        return {
            'filename': filename,
            'shape': shape,
            'columns': columns,
            'data_types': dtypes_info,
            'missing_values': missing_values,
            'missing_percentages': missing_percentages,
            'sample_data': sample_data,
            'numeric_statistics': numeric_stats,
            'categorical_info': categorical_info,
            'row_limit_exceeded': row_limit_exceeded,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing dataframe: {str(e)}")
        raise

@dataset_blueprint.route('/preview', methods=['POST'])
def preview_dataset():
    """
    Accept file upload and return data preview with comprehensive analysis.
    """
    logger.info("Dataset preview endpoint called")
    
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        # Validate file extension
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Unsupported file format. Allowed formats: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
            
        # Check file size
        file_size = get_file_size(file)
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                'error': f'File size ({file_size / (1024*1024):.1f}MB) exceeds maximum allowed size ({MAX_FILE_SIZE / (1024*1024)}MB)'
            }), 400
            
        # Secure filename
        filename = secure_filename(file.filename)
        
        # Read and analyze the file
        df = safe_read_file(file, filename)
        analysis = analyze_dataframe(df, filename)
        
        logger.info(f"Successfully analyzed dataset: {filename}, shape: {analysis['shape']}")
        
        return jsonify({
            'success': True,
            'message': 'Dataset preview generated successfully',
            'data': analysis
        }), 200
        
    except RequestEntityTooLarge:
        return jsonify({'error': 'File size exceeds maximum allowed size'}), 413
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error in preview_dataset: {str(e)}", exc_info=True)
        return jsonify({'error': f'Failed to process file: {str(e)}'}), 500

@dataset_blueprint.route('/validate', methods=['POST'])
def validate_preprocessing():
    """
    Accept preprocessing configuration and return validation results.
    """
    logger.info("Dataset validation endpoint called")
    
    try:
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON payload received'}), 400
            
        # Extract configuration
        config = data.get('preprocessing_config', {})
        file_info = data.get('file_info', {})
        
        if not config or not file_info:
            return jsonify({'error': 'Missing preprocessing_config or file_info'}), 400
            
        # Validate configuration structure
        required_fields = ['target_column', 'feature_columns', 'task_type']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
            
        # Validation results
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Validate target column
        target_column = config['target_column']
        available_columns = file_info.get('columns', [])
        
        if target_column not in available_columns:
            validation_results['errors'].append(f'Target column "{target_column}" not found in dataset')
            validation_results['valid'] = False
            
        # Validate feature columns
        feature_columns = config['feature_columns']
        if not isinstance(feature_columns, list) or len(feature_columns) == 0:
            validation_results['errors'].append('At least one feature column must be selected')
            validation_results['valid'] = False
        else:
            invalid_features = [col for col in feature_columns if col not in available_columns]
            if invalid_features:
                validation_results['errors'].append(f'Feature columns not found: {", ".join(invalid_features)}')
                validation_results['valid'] = False
                
        # Validate task type
        task_type = config['task_type']
        valid_task_types = ['classification', 'regression']
        if task_type not in valid_task_types:
            validation_results['errors'].append(f'Invalid task type. Must be one of: {", ".join(valid_task_types)}')
            validation_results['valid'] = False
            
        # Check for missing values in target column
        missing_values = file_info.get('missing_values', {})
        if target_column in missing_values and missing_values[target_column] > 0:
            validation_results['warnings'].append(f'Target column "{target_column}" has {missing_values[target_column]} missing values')
            
        # Check for high missing value percentages in features
        missing_percentages = file_info.get('missing_percentages', {})
        high_missing_features = []
        for col in feature_columns:
            if col in missing_percentages and missing_percentages[col] > 50:
                high_missing_features.append(f'{col} ({missing_percentages[col]:.1f}%)')
                
        if high_missing_features:
            validation_results['warnings'].append(f'Features with high missing values: {", ".join(high_missing_features)}')
            
        # Task-specific validations
        data_types = file_info.get('data_types', {})
        
        if task_type == 'classification':
            # Check if target column is appropriate for classification
            if target_column in data_types:
                target_type = data_types[target_column]
                if target_type not in ['categorical', 'integer', 'boolean']:
                    validation_results['warnings'].append(f'Target column "{target_column}" type ({target_type}) may not be suitable for classification')
                    
        elif task_type == 'regression':
            # Check if target column is numeric for regression
            if target_column in data_types:
                target_type = data_types[target_column]
                if target_type not in ['integer', 'float']:
                    validation_results['warnings'].append(f'Target column "{target_column}" type ({target_type}) may not be suitable for regression')
                    
        # Recommendations
        if len(feature_columns) < 2:
            validation_results['recommendations'].append('Consider using more feature columns for better model performance')
            
        if file_info.get('shape', [0, 0])[0] < 1000:
            validation_results['recommendations'].append('Dataset has fewer than 1000 rows, which may limit model performance')
            
        # Check for categorical features that might need encoding
        categorical_features = [col for col in feature_columns if data_types.get(col) == 'categorical']
        if categorical_features:
            validation_results['recommendations'].append(f'Categorical features may need encoding: {", ".join(categorical_features)}')
            
        logger.info(f"Validation completed. Valid: {validation_results['valid']}, Warnings: {len(validation_results['warnings'])}, Errors: {len(validation_results['errors'])}")
        
        return jsonify({
            'success': True,
            'validation_results': validation_results
        }), 200
        
    except Exception as e:
        logger.error(f"Error in validate_preprocessing: {str(e)}", exc_info=True)
        return jsonify({'error': f'Validation failed: {str(e)}'}), 500

@dataset_blueprint.route('/create', methods=['POST'])
def create_custom_dataset():
    """
    Create final dataset with user configuration and save it.
    """
    logger.info("Create custom dataset endpoint called")
    
    try:
        # Check if file and config are present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        config_json = request.form.get('config')
        
        if not config_json:
            return jsonify({'error': 'No configuration provided'}), 400
            
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid JSON configuration'}), 400
            
        # Validate file
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
            
        # Check file size
        file_size = get_file_size(file)
        if file_size > MAX_FILE_SIZE:
            return jsonify({'error': 'File size exceeds maximum allowed size'}), 400
            
        # Read the file
        filename = secure_filename(file.filename)
        df = safe_read_file(file, filename)
        
        # Apply preprocessing based on configuration
        dataset_name = config.get('dataset_name', f'custom_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        target_column = config['target_column']
        feature_columns = config['feature_columns']
        task_type = config['task_type']
        
        # Create processed dataset
        processed_data = {
            'name': dataset_name,
            'task_type': task_type,
            'target_column': target_column,
            'feature_columns': feature_columns,
            'original_filename': filename,
            'shape': df.shape,
            'created_at': datetime.now().isoformat()
        }
        
        # Handle missing values if specified
        missing_value_strategy = config.get('missing_value_strategy', 'drop')
        if missing_value_strategy == 'drop':
            df_processed = df.dropna(subset=[target_column] + feature_columns)
        elif missing_value_strategy == 'fill_mean':
            df_processed = df.copy()
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
        elif missing_value_strategy == 'fill_mode':
            df_processed = df.copy()
            for col in feature_columns + [target_column]:
                if col in df_processed.columns:
                    mode_value = df_processed[col].mode()
                    if len(mode_value) > 0:
                        df_processed[col] = df_processed[col].fillna(mode_value[0])
        else:
            df_processed = df.copy()
            
        # Extract features and target
        X = df_processed[feature_columns]
        y = df_processed[target_column]
        
        # Basic preprocessing for categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            # Simple label encoding for categorical variables
            for col in categorical_columns:
                X[col] = pd.Categorical(X[col]).codes
                
        # For classification, encode target if it's categorical
        if task_type == 'classification' and y.dtype == 'object':
            y_encoded = pd.Categorical(y)
            processed_data['class_labels'] = y_encoded.categories.tolist()
            y = y_encoded.codes
        else:
            processed_data['class_labels'] = None
            
        # Save the dataset
        datasets_dir = os.path.join(current_app.config.get('PROJECT_ROOT', ''), 'datasets', 'custom')
        os.makedirs(datasets_dir, exist_ok=True)
        
        # Save processed data
        dataset_file = os.path.join(datasets_dir, f'{dataset_name}.npz')
        np.savez(dataset_file, X=X.values, y=y.values if hasattr(y, 'values') else y)
        
        # Save metadata
        metadata_file = os.path.join(datasets_dir, f'{dataset_name}_metadata.json')
        processed_data['file_path'] = dataset_file
        processed_data['processed_shape'] = [X.shape[0], X.shape[1]]
        
        with open(metadata_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
            
        # Register the dataset with the DatasetRegistry for immediate use
        try:
            from backend.dataset_loader import register_custom_dataset
            
            dataset_config = {
                'name': dataset_name,
                'file_path': dataset_file,
                'metadata_path': metadata_file,
                'task_type': task_type,
                'feature_count': len(feature_columns),
                'class_labels': processed_data['class_labels']
            }
            
            register_custom_dataset(dataset_config)
            logger.info(f"Successfully registered custom dataset '{dataset_name}' with DatasetRegistry")
            
        except Exception as e:
            logger.warning(f"Could not register dataset with DatasetRegistry: {str(e)}")
            # Continue anyway, as the dataset files are created successfully
            
        logger.info(f"Custom dataset created successfully: {dataset_name}")
        
        return jsonify({
            'success': True,
            'message': 'Custom dataset created successfully',
            'dataset': {
                'name': dataset_name,
                'shape': processed_data['processed_shape'],
                'task_type': task_type,
                'feature_count': len(feature_columns),
                'class_labels': processed_data['class_labels']
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error in create_custom_dataset: {str(e)}", exc_info=True)
        return jsonify({'error': f'Failed to create dataset: {str(e)}'}), 500

@dataset_blueprint.route('/custom', methods=['GET'])
def list_custom_datasets():
    """
    List all custom datasets with their metadata.
    """
    logger.info("List custom datasets endpoint called")
    
    try:
        datasets_dir = os.path.join(current_app.config.get('PROJECT_ROOT', ''), 'datasets', 'custom')
        
        if not os.path.exists(datasets_dir):
            return jsonify({
                'success': True,
                'datasets': []
            }), 200
            
        datasets = []
        
        # Find all metadata files
        for filename in os.listdir(datasets_dir):
            if filename.endswith('_metadata.json'):
                metadata_file = os.path.join(datasets_dir, filename)
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        
                    # Check if the corresponding data file exists
                    data_file = metadata.get('file_path', '')
                    if os.path.exists(data_file):
                        datasets.append({
                            'name': metadata.get('name', 'Unknown'),
                            'task_type': metadata.get('task_type', 'Unknown'),
                            'shape': metadata.get('processed_shape', [0, 0]),
                            'feature_count': len(metadata.get('feature_columns', [])),
                            'target_column': metadata.get('target_column', 'Unknown'),
                            'created_at': metadata.get('created_at', 'Unknown'),
                            'original_filename': metadata.get('original_filename', 'Unknown'),
                            'class_labels': metadata.get('class_labels')
                        })
                except Exception as e:
                    logger.warning(f"Could not load metadata from {filename}: {str(e)}")
                    continue
                    
        # Sort by creation date (newest first)
        datasets.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        logger.info(f"Found {len(datasets)} custom datasets")
        
        return jsonify({
            'success': True,
            'datasets': datasets
        }), 200
        
    except Exception as e:
        logger.error(f"Error in list_custom_datasets: {str(e)}", exc_info=True)
        return jsonify({'error': f'Failed to list datasets: {str(e)}'}), 500

@dataset_blueprint.route('/available', methods=['GET'])
def get_available_datasets():
    """
    Get all available datasets (built-in and custom) for frontend selection.
    """
    logger.info("Get available datasets endpoint called")
    
    try:
        from backend.dataset_loader import dataset_registry
        
        # Get all available datasets
        all_datasets = dataset_registry.get_available_datasets()
        
        # Get detailed info for custom datasets
        custom_datasets = dataset_registry.get_custom_datasets()
        custom_dataset_names = {ds['name'] for ds in custom_datasets}
        
        # Build response with dataset categories
        dataset_list = {
            'built_in': [],
            'custom': custom_datasets,
            'all_names': all_datasets
        }
        
        # Categorize datasets
        for dataset_name in all_datasets:
            if dataset_name not in custom_dataset_names:
                # This is a built-in dataset
                dataset_list['built_in'].append({
                    'name': dataset_name,
                    'type': 'built_in',
                    'description': f'Built-in {dataset_name} dataset'
                })
        
        logger.info(f"Found {len(dataset_list['built_in'])} built-in and {len(dataset_list['custom'])} custom datasets")
        
        return jsonify({
            'success': True,
            'datasets': dataset_list
        }), 200
        
    except Exception as e:
        logger.error(f"Error in get_available_datasets: {str(e)}", exc_info=True)
        return jsonify({'error': f'Failed to get available datasets: {str(e)}'}), 500

# Error handlers for the blueprint
@dataset_blueprint.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File size exceeds maximum allowed size'}), 413

@dataset_blueprint.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'Bad request'}), 400

@dataset_blueprint.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500 