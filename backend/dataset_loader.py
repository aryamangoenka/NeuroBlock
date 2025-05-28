import numpy as np
import os
from flask import current_app
from backend.utils.logging import get_logger
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Dataset loaders
from backend.datasets.iris import load_iris_dataset
from backend.datasets.mnist import load_mnist_dataset
from backend.datasets.cifar10 import load_cifar10_dataset
from backend.datasets.california import load_california_housing_dataset
from backend.datasets.breastcancer import load_breast_cancer_dataset

# Initialize logger
logger = get_logger(__name__)

class DatasetRegistry:
    """Registry of available datasets and their loader functions"""
    
    def __init__(self):
        self.loaders = {
            "Iris": load_iris_dataset,
            "MNIST": load_mnist_dataset,
            "CIFAR-10": load_cifar10_dataset,
            "California Housing": load_california_housing_dataset,
            "Breast Cancer": load_breast_cancer_dataset
        }
        self.custom_datasets = {}  # Store custom dataset configurations
        
    def register_dataset(self, name, loader_function):
        """
        Register a new dataset loader function
        
        Args:
            name (str): Name of the dataset
            loader_function (callable): Function that loads the dataset
        """
        if name in self.loaders:
            logger.warning(f"Dataset {name} already registered. Overwriting.")
        self.loaders[name] = loader_function
        logger.info(f"Registered dataset: {name}")
    
    def register_custom_dataset(self, dataset_config):
        """
        Register a custom dataset with the registry
        
        Args:
            dataset_config (dict): Configuration dictionary containing:
                - name: Dataset name
                - file_path: Path to the .npz file
                - metadata_path: Path to the metadata JSON file
                - task_type: 'classification' or 'regression'
                - feature_count: Number of features
                - class_labels: List of class labels (for classification)
        
        Returns:
            str: Dataset ID (name) for future reference
        """
        dataset_name = dataset_config['name']
        
        # Validate required fields
        required_fields = ['name', 'file_path', 'metadata_path', 'task_type']
        missing_fields = [field for field in required_fields if field not in dataset_config]
        
        if missing_fields:
            raise ValueError(f"Missing required fields in dataset config: {missing_fields}")
        
        # Verify files exist
        if not os.path.exists(dataset_config['file_path']):
            raise FileNotFoundError(f"Dataset file not found: {dataset_config['file_path']}")
        
        if not os.path.exists(dataset_config['metadata_path']):
            raise FileNotFoundError(f"Metadata file not found: {dataset_config['metadata_path']}")
        
        # Store the configuration
        self.custom_datasets[dataset_name] = dataset_config
        
        # Create and register a loader function for this custom dataset
        def custom_loader():
            return self._load_custom_dataset_data(dataset_name)
        
        self.register_dataset(dataset_name, custom_loader)
        
        logger.info(f"Registered custom dataset: {dataset_name}")
        return dataset_name
    
    def load_custom_dataset(self, dataset_id):
        """
        Load a custom dataset for training
        
        Args:
            dataset_id (str): ID/name of the custom dataset
            
        Returns:
            tuple: ((x_train, y_train), (x_test, y_test)) format
        """
        # Try exact match first
        if dataset_id in self.custom_datasets:
            return self._load_custom_dataset_data(dataset_id)
        
        # Try case-insensitive match
        dataset_id_lower = dataset_id.lower()
        for registered_name in self.custom_datasets.keys():
            if registered_name.lower() == dataset_id_lower:
                logger.info(f"Found case-insensitive match for custom dataset: '{dataset_id}' -> '{registered_name}'")
                return self._load_custom_dataset_data(registered_name)
        
        # No match found
        raise ValueError(f"Custom dataset '{dataset_id}' not found in registry")
    
    def _load_custom_dataset_data(self, dataset_name):
        """
        Internal method to load custom dataset data
        
        Args:
            dataset_name (str): Name of the custom dataset
            
        Returns:
            tuple: ((x_train, y_train), (x_test, y_test)) format
        """
        config = self.custom_datasets[dataset_name]
        
        try:
            # Load the dataset from .npz file
            data = np.load(config['file_path'])
            X = data['X']
            y = data['y']
            
            # Load metadata for additional processing info
            with open(config['metadata_path'], 'r') as f:
                metadata = json.load(f)
            
            task_type = config['task_type']
            
            # Preprocessing based on task type
            if task_type == 'classification':
                # For classification, ensure labels are properly encoded
                if y.dtype == 'object' or len(np.unique(y)) != len(config.get('class_labels', [])):
                    # Re-encode if needed
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(y)
                
                # One-hot encode for classification
                n_classes = len(np.unique(y))
                
                # Always one-hot encode for consistency with model output
                # This ensures the target shape matches the model output shape
                encoder = OneHotEncoder(sparse_output=False)
                y = encoder.fit_transform(y.reshape(-1, 1))
                
                logger.info(f"One-hot encoded labels for '{dataset_name}': {n_classes} classes, shape: {y.shape}")
                    
            elif task_type == 'regression':
                # For regression, ensure target is float
                y = y.astype(np.float32).reshape(-1, 1)
            
            # Normalize features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            # Split into train/test sets
            test_size = 0.2
            random_state = 42
            
            x_train, x_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, 
                stratify=y if task_type == 'classification' and len(y.shape) == 1 else None
            )
            
            # Convert to TensorFlow tensors
            x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
            x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
            y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
            y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
            
            logger.info(f"Loaded custom dataset '{dataset_name}': "
                       f"train shape: {x_train.shape}, test shape: {x_test.shape}")
            
            return (x_train, y_train), (x_test, y_test)
            
        except Exception as e:
            logger.error(f"Error loading custom dataset '{dataset_name}': {str(e)}")
            raise
    
    def get_custom_datasets(self):
        """
        Get a list of all registered custom datasets
        
        Returns:
            list: List of dictionaries containing custom dataset information
        """
        custom_dataset_list = []
        
        for name, config in self.custom_datasets.items():
            try:
                # Try to load metadata for additional info
                with open(config['metadata_path'], 'r') as f:
                    metadata = json.load(f)
                
                dataset_info = {
                    'name': name,
                    'task_type': config['task_type'],
                    'feature_count': config.get('feature_count', metadata.get('processed_shape', [0, 0])[1]),
                    'shape': metadata.get('processed_shape', [0, 0]),
                    'target_column': metadata.get('target_column', 'Unknown'),
                    'created_at': metadata.get('created_at', 'Unknown'),
                    'original_filename': metadata.get('original_filename', 'Unknown'),
                    'class_labels': config.get('class_labels', metadata.get('class_labels'))
                }
                custom_dataset_list.append(dataset_info)
                
            except Exception as e:
                logger.warning(f"Could not load metadata for custom dataset '{name}': {str(e)}")
                # Fallback to basic info
                dataset_info = {
                    'name': name,
                    'task_type': config.get('task_type', 'Unknown'),
                    'feature_count': config.get('feature_count', 0),
                    'shape': [0, 0],
                    'target_column': 'Unknown',
                    'created_at': 'Unknown',
                    'original_filename': 'Unknown',
                    'class_labels': config.get('class_labels')
                }
                custom_dataset_list.append(dataset_info)
        
        return custom_dataset_list
    
    def _discover_custom_datasets(self):
        """
        Automatically discover and register custom datasets from the datasets/custom directory
        """
        try:
            # Get the custom datasets directory
            if hasattr(current_app, 'config'):
                project_root = current_app.config.get('PROJECT_ROOT', '')
            else:
                project_root = os.path.dirname(os.path.dirname(__file__))
            
            custom_datasets_dir = os.path.join(project_root, 'datasets', 'custom')
            
            if not os.path.exists(custom_datasets_dir):
                return
            
            # Find all metadata files
            for filename in os.listdir(custom_datasets_dir):
                if filename.endswith('_metadata.json'):
                    metadata_path = os.path.join(custom_datasets_dir, filename)
                    
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        dataset_name = metadata.get('name')
                        data_file_path = metadata.get('file_path', '')
                        
                        # Check if data file exists
                        if dataset_name and os.path.exists(data_file_path):
                            dataset_config = {
                                'name': dataset_name,
                                'file_path': data_file_path,
                                'metadata_path': metadata_path,
                                'task_type': metadata.get('task_type', 'classification'),
                                'feature_count': len(metadata.get('feature_columns', [])),
                                'class_labels': metadata.get('class_labels')
                            }
                            
                            # Only register if not already registered
                            if dataset_name not in self.custom_datasets:
                                self.register_custom_dataset(dataset_config)
                                
                    except Exception as e:
                        logger.warning(f"Could not load custom dataset from {filename}: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.warning(f"Error discovering custom datasets: {str(e)}")
        
    def get_loader(self, dataset_name):
        """
        Get the loader function for a dataset
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            callable: Loader function for the dataset
            
        Raises:
            ValueError: If dataset_name is not registered
        """
        # Auto-discover custom datasets if not already done
        if dataset_name not in self.loaders:
            self._discover_custom_datasets()
        
        # Try exact match first
        if dataset_name in self.loaders:
            return self.loaders[dataset_name]
        
        # Try case-insensitive match
        dataset_name_lower = dataset_name.lower()
        for registered_name, loader in self.loaders.items():
            if registered_name.lower() == dataset_name_lower:
                logger.info(f"Found case-insensitive match: '{dataset_name}' -> '{registered_name}'")
                return loader
        
        # No match found
        error_msg = f"Unknown dataset: {dataset_name}. Available datasets: {', '.join(self.loaders.keys())}"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    def get_available_datasets(self):
        """
        Get a list of available datasets (including custom datasets)
        
        Returns:
            list: Names of available datasets
        """
        # Auto-discover custom datasets
        self._discover_custom_datasets()
        return list(self.loaders.keys())

# Initialize the dataset registry
dataset_registry = DatasetRegistry()

def get_model_dataset():
    """
    Get the dataset name from the saved model architecture
    
    Returns:
        str or None: Name of the dataset used in the model, or None if not found
    """
    try:
        # Try to get the model architecture file path from the app config or fallback to default
        try:
            model_architecture_file = current_app.config.get('MODEL_ARCHITECTURE_FILE')
        except RuntimeError:  # When Flask app context is not available
            model_architecture_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_model.json")
        
        # Try to load model architecture if the file exists
        if os.path.exists(model_architecture_file):
            with open(model_architecture_file, "r") as f:
                model = json.load(f)
            return model.get("dataset")
    except Exception as e:
        logger.error(f"Error loading model architecture: {str(e)}")
    
    return None

def load_dataset(dataset_name):
    """
    Load and preprocess a dataset based on its name.
    
    Args:
        dataset_name (str): Name of the dataset to load
        
    Returns:
        tuple: ((x_train, y_train), (x_test, y_test)) - Training and test data
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Get the loader function for this dataset
    loader = dataset_registry.get_loader(dataset_name)
    
    # Load and return the dataset
    return loader()

# Convenience functions for custom dataset management
def register_custom_dataset(dataset_config):
    """
    Register a custom dataset with the global registry
    
    Args:
        dataset_config (dict): Dataset configuration
        
    Returns:
        str: Dataset ID
    """
    return dataset_registry.register_custom_dataset(dataset_config)

def get_custom_datasets():
    """
    Get all custom datasets from the global registry
    
    Returns:
        list: List of custom dataset information
    """
    return dataset_registry.get_custom_datasets()

def load_custom_dataset(dataset_id):
    """
    Load a custom dataset from the global registry
    
    Args:
        dataset_id (str): Dataset ID
        
    Returns:
        tuple: ((x_train, y_train), (x_test, y_test))
    """
    return dataset_registry.load_custom_dataset(dataset_id)

if __name__ == "__main__":
    # Use the dataset specified in the saved model, or default to MNIST
    dataset_name = get_model_dataset() or "MNIST"
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset_name)
    print(f"Dataset: {dataset_name}")
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
    
    # Demo custom datasets
    custom_datasets = get_custom_datasets()
    if custom_datasets:
        print(f"\nFound {len(custom_datasets)} custom datasets:")
        for dataset in custom_datasets:
            print(f"  - {dataset['name']} ({dataset['task_type']}, shape: {dataset['shape']})")
