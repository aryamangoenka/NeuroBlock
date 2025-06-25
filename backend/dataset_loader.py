import numpy as np
import os
from flask import current_app, session
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
        # Remove global custom_datasets - now handled per session
        
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
    
    def register_custom_dataset(self, dataset_config, session_id=None):
        """
        Register a custom dataset with the registry for the current session
        
        Args:
            dataset_config (dict): Configuration dictionary containing:
                - name: Dataset name
                - file_path: Path to the .npz file
                - metadata_path: Path to the metadata JSON file
                - task_type: 'classification' or 'regression'
                - feature_count: Number of features
                - class_labels: List of class labels (for classification)
            session_id (str, optional): Session ID. If None, uses current session.
        
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
        
        # Create and register a loader function for this custom dataset
        def custom_loader():
            return self._load_custom_dataset_data(dataset_name, session_id)
        
        self.register_dataset(dataset_name, custom_loader)
        
        logger.info(f"Registered custom dataset: {dataset_name}")
        return dataset_name
    
    def load_custom_dataset(self, dataset_id, session_id=None):
        """
        Load a custom dataset for training
        
        Args:
            dataset_id (str): ID/name of the custom dataset
            session_id (str, optional): Session ID. If None, uses current session.
            
        Returns:
            tuple: ((x_train, y_train), (x_test, y_test)) format
        """
        return self._load_custom_dataset_data(dataset_id, session_id)
    
    def _load_custom_dataset_data(self, dataset_name, session_id=None):
        """
        Internal method to load custom dataset data from session-specific storage
        
        Args:
            dataset_name (str): Name of the custom dataset
            session_id (str, optional): Session ID. If None, uses current session.
            
        Returns:
            tuple: ((x_train, y_train), (x_test, y_test)) format
        """
        try:
            # Get session-specific dataset directory
            from backend.utils.session_manager import get_session_datasets_dir, get_session_id
            
            if session_id is None:
                try:
                    session_id = get_session_id()
                except RuntimeError:
                    # No session context available, try to find dataset in any session
                    logger.warning(f"No session context for loading dataset '{dataset_name}', searching all sessions")
                    return self._load_dataset_from_any_session(dataset_name)
            
            datasets_dir = get_session_datasets_dir(session_id)
            
            # Find dataset files
            dataset_file = os.path.join(datasets_dir, f'{dataset_name}.npz')
            metadata_file = os.path.join(datasets_dir, f'{dataset_name}_metadata.json')
            
            if not os.path.exists(dataset_file):
                raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
            
            if not os.path.exists(metadata_file):
                raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
            
            # Load the dataset from .npz file
            data = np.load(dataset_file)
            X = data['X']
            y = data['y']
            
            # Load metadata for additional processing info
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            task_type = metadata.get('task_type', 'classification')
            dataset_type = metadata.get('dataset_type', 'tabular')
            
            # Handle image datasets differently from tabular datasets
            if dataset_type == 'image':
                logger.info(f"Loading image dataset '{dataset_name}': {X.shape}")
                
                # For image datasets, X is already in proper shape (N, H, W, C)
                # No need to flatten or apply StandardScaler normalization
                # Images are already normalized to [0,1] range during preprocessing
                
                # Ensure proper data types
                X = X.astype(np.float32)
                
                # For classification, ensure labels are properly encoded
                if task_type == 'classification':
                    class_labels = metadata.get('class_labels')
                    n_classes = len(np.unique(y))
                    
                    # One-hot encode for classification
                    encoder = OneHotEncoder(sparse_output=False)
                    y = encoder.fit_transform(y.reshape(-1, 1))
                    
                    logger.info(f"One-hot encoded labels for image dataset '{dataset_name}': {n_classes} classes, shape: {y.shape}")
                
                # No scaler needed for image datasets (already normalized)
                scaler_path = None
                
            else:
                # Handle tabular datasets (existing logic)
                logger.info(f"Loading tabular dataset '{dataset_name}': {X.shape}")
                
                # Preprocessing based on task type
                if task_type == 'classification':
                    # For classification, ensure labels are properly encoded
                    class_labels = metadata.get('class_labels')
                    
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
                
                # Normalize features and save scaler for validation use
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                
                # Save the scaler for validation predictions
                try:
                    from backend.utils.session_manager import get_session_id, get_session_datasets_dir
                    session_id = get_session_id() if session_id is None else session_id
                    datasets_dir = get_session_datasets_dir(session_id)
                    scaler_path = os.path.join(datasets_dir, f'{dataset_name}_scaler.pkl')
                    
                    import pickle
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(scaler, f)
                    logger.info(f"Saved scaler for custom dataset '{dataset_name}' to: {scaler_path}")
                except Exception as e:
                    logger.warning(f"Could not save scaler for dataset '{dataset_name}': {str(e)}")
            
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
    
    def _load_dataset_from_any_session(self, dataset_name):
        """
        Fallback method to load dataset from any session when no session context is available.
        This is used during training when the session context might not be available.
        
        Args:
            dataset_name (str): Name of the custom dataset
            
        Returns:
            tuple: ((x_train, y_train), (x_test, y_test)) format
        """
        try:
            project_root = current_app.config.get('PROJECT_ROOT', '')
            sessions_dir = os.path.join(project_root, 'datasets', 'sessions')
            
            if not os.path.exists(sessions_dir):
                raise FileNotFoundError(f"No sessions directory found")
            
            # Search through all session directories
            for session_id in os.listdir(sessions_dir):
                session_path = os.path.join(sessions_dir, session_id)
                
                if not os.path.isdir(session_path):
                    continue
                
                dataset_file = os.path.join(session_path, f'{dataset_name}.npz')
                metadata_file = os.path.join(session_path, f'{dataset_name}_metadata.json')
                
                if os.path.exists(dataset_file) and os.path.exists(metadata_file):
                    logger.info(f"Found dataset '{dataset_name}' in session: {session_id}")
                    return self._load_custom_dataset_data(dataset_name, session_id)
            
            raise FileNotFoundError(f"Dataset '{dataset_name}' not found in any session")
            
        except Exception as e:
            logger.error(f"Error searching for dataset '{dataset_name}' across sessions: {str(e)}")
            raise
    
    def get_custom_datasets(self, session_id=None):
        """
        Get a list of all custom datasets for the current session
        
        Args:
            session_id (str, optional): Session ID. If None, uses current session.
        
        Returns:
            list: List of dictionaries containing custom dataset information
        """
        custom_dataset_list = []
        
        try:
            from backend.utils.session_manager import get_session_datasets_dir, get_session_id
            
            if session_id is None:
                try:
                    session_id = get_session_id()
                except RuntimeError:
                    # No session context available
                    logger.warning("No session context available for getting custom datasets")
                    return []
            
            datasets_dir = get_session_datasets_dir(session_id)
            
            if not os.path.exists(datasets_dir):
                return []
            
            # Find all metadata files
            for filename in os.listdir(datasets_dir):
                if filename.endswith('_metadata.json'):
                    metadata_file = os.path.join(datasets_dir, filename)
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        # Check if the corresponding data file exists
                        dataset_name = metadata.get('name', 'Unknown')
                        data_file = os.path.join(datasets_dir, f'{dataset_name}.npz')
                        
                        if os.path.exists(data_file):
                            dataset_info = {
                                'name': dataset_name,
                                'task_type': metadata.get('task_type', 'Unknown'),
                                'feature_count': len(metadata.get('feature_columns', [])),
                                'shape': metadata.get('processed_shape', [0, 0]),
                                'target_column': metadata.get('target_column', 'Unknown'),
                                'created_at': metadata.get('created_at', 'Unknown'),
                                'original_filename': metadata.get('original_filename', 'Unknown'),
                                'class_labels': metadata.get('class_labels')
                            }
                            custom_dataset_list.append(dataset_info)
                            
                    except Exception as e:
                        logger.warning(f"Could not load metadata from {filename}: {str(e)}")
                        continue
        
        except Exception as e:
            logger.error(f"Error getting custom datasets: {str(e)}")
        
        return custom_dataset_list
    
    def _discover_custom_datasets(self):
        """
        This method is no longer used with session-based storage.
        Custom datasets are now discovered per session.
        """
        logger.info("Custom dataset discovery is now handled per session")
        pass
    
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
        # Try exact match first
        if dataset_name in self.loaders:
            return self.loaders[dataset_name]
        
        # Try case-insensitive match
        dataset_name_lower = dataset_name.lower()
        for registered_name, loader in self.loaders.items():
            if registered_name.lower() == dataset_name_lower:
                logger.info(f"Found case-insensitive match: '{dataset_name}' -> '{registered_name}'")
                return loader
        
        # Not found in registry - try to discover custom datasets from current session
        logger.info(f"Dataset '{dataset_name}' not in registry, searching session for custom datasets...")
        try:
            from backend.utils.session_manager import get_session_datasets_dir, get_session_id
            
            # Try to get session ID and datasets directory
            try:
                session_id = get_session_id()
                datasets_dir = get_session_datasets_dir(session_id)
            except RuntimeError:
                # No session context - search all sessions
                logger.warning("No session context, searching all sessions for custom datasets")
                datasets_dir = None
                session_id = None
            
            # Search for custom dataset
            if datasets_dir and os.path.exists(datasets_dir):
                # Search in current session
                for filename in os.listdir(datasets_dir):
                    if filename.endswith('_metadata.json'):
                        try:
                            metadata_file = os.path.join(datasets_dir, filename)
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            dataset_metadata_name = metadata.get('name', 'Unknown')
                            
                            # Check exact and case-insensitive matches
                            if (dataset_metadata_name == dataset_name or 
                                dataset_metadata_name.lower() == dataset_name_lower):
                                
                                # Found the dataset - register it
                                dataset_file = os.path.join(datasets_dir, f'{dataset_metadata_name}.npz')
                                
                                if os.path.exists(dataset_file):
                                    dataset_config = {
                                        'name': dataset_metadata_name,
                                        'file_path': dataset_file,
                                        'metadata_path': metadata_file,
                                        'task_type': metadata.get('task_type', 'classification'),
                                        'feature_count': len(metadata.get('feature_columns', [])),
                                        'class_labels': metadata.get('class_labels')
                                    }
                                    
                                    # Register the dataset
                                    self.register_custom_dataset(dataset_config, session_id)
                                    logger.info(f"Successfully discovered and registered custom dataset: '{dataset_metadata_name}'")
                                    
                                    # Return the loader
                                    return self.loaders[dataset_metadata_name]
                                    
                        except Exception as e:
                            logger.warning(f"Error processing metadata file {filename}: {str(e)}")
                            continue
            
            # If not found in current session, search all sessions and find the most recent
            if not datasets_dir or session_id is None:
                logger.info(f"Searching all sessions for dataset '{dataset_name}'...")
                project_root = current_app.config.get('PROJECT_ROOT', '') if hasattr(current_app, 'config') else ''
                all_sessions_dir = os.path.join(project_root, 'datasets', 'sessions') if project_root else 'sessions'
                
                found_datasets = []  # Store all matches with timestamps
                
                if os.path.exists(all_sessions_dir):
                    for session_folder in os.listdir(all_sessions_dir):
                        session_datasets_dir = os.path.join(all_sessions_dir, session_folder)
                        
                        if not os.path.isdir(session_datasets_dir):
                            continue
                            
                        for filename in os.listdir(session_datasets_dir):
                            if filename.endswith('_metadata.json'):
                                try:
                                    metadata_file = os.path.join(session_datasets_dir, filename)
                                    with open(metadata_file, 'r') as f:
                                        metadata = json.load(f)
                                    
                                    dataset_metadata_name = metadata.get('name', 'Unknown')
                                    
                                    # Check exact and case-insensitive matches
                                    if (dataset_metadata_name == dataset_name or 
                                        dataset_metadata_name.lower() == dataset_name_lower):
                                        
                                        # Found a matching dataset - check if data file exists
                                        dataset_file = os.path.join(session_datasets_dir, f'{dataset_metadata_name}.npz')
                                        
                                        if os.path.exists(dataset_file):
                                            found_datasets.append({
                                                'metadata': metadata,
                                                'metadata_file': metadata_file,
                                                'dataset_file': dataset_file,
                                                'session_folder': session_folder,
                                                'created_at': metadata.get('created_at', ''),
                                                'feature_count': len(metadata.get('feature_columns', []))
                                            })
                                            
                                except Exception as e:
                                    logger.warning(f"Error processing metadata file {filename} in session {session_folder}: {str(e)}")
                                    continue
                    
                    # If we found multiple datasets, use the most recent one
                    if found_datasets:
                        # Sort by creation time (most recent first)
                        found_datasets.sort(key=lambda x: x['created_at'], reverse=True)
                        best_match = found_datasets[0]
                        
                        logger.info(f"Found {len(found_datasets)} datasets named '{dataset_name}', using most recent from {best_match['created_at']}")
                        
                        dataset_config = {
                            'name': dataset_name,
                            'file_path': best_match['dataset_file'],
                            'metadata_path': best_match['metadata_file'],
                            'task_type': best_match['metadata'].get('task_type', 'classification'),
                            'feature_count': best_match['feature_count'],
                            'class_labels': best_match['metadata'].get('class_labels')
                        }
                        
                        # Register the most recent dataset
                        self.register_custom_dataset(dataset_config, best_match['session_folder'])
                        logger.info(f"Successfully discovered and registered most recent custom dataset: '{dataset_name}' from session: {best_match['session_folder']}")
                        
                        # Return the loader
                        return self.loaders[dataset_name]
                                    
        except Exception as e:
            logger.error(f"Error during custom dataset discovery: {str(e)}")
        
        # No match found after discovery attempt
        available_datasets = list(self.loaders.keys())
        error_msg = f"Unknown dataset: {dataset_name}. Available datasets: {', '.join(available_datasets)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    def get_available_datasets(self):
        """
        Get a list of available datasets (including custom datasets)
        
        Returns:
            list: Names of available datasets
        """
        # Start with current registry
        available_datasets = list(self.loaders.keys())
        
        # Also include discovered custom datasets
        try:
            custom_datasets = self.get_custom_datasets()
            for dataset in custom_datasets:
                dataset_name = dataset.get('name')
                if dataset_name and dataset_name not in available_datasets:
                    available_datasets.append(dataset_name)
        except Exception as e:
            logger.warning(f"Could not get custom datasets for available list: {str(e)}")
        
        return available_datasets

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
def register_custom_dataset(dataset_config, session_id=None):
    """
    Register a custom dataset with the global registry for the current session
    
    Args:
        dataset_config (dict): Dataset configuration
        session_id (str, optional): Session ID. If None, uses current session.
        
    Returns:
        str: Dataset ID
    """
    return dataset_registry.register_custom_dataset(dataset_config, session_id)

def get_custom_datasets(session_id=None):
    """
    Get all custom datasets for the current session
    
    Args:
        session_id (str, optional): Session ID. If None, uses current session.
    
    Returns:
        list: List of custom dataset information
    """
    return dataset_registry.get_custom_datasets(session_id)

def load_custom_dataset(dataset_id, session_id=None):
    """
    Load a custom dataset for the current session
    
    Args:
        dataset_id (str): Dataset ID
        session_id (str, optional): Session ID. If None, uses current session.
        
    Returns:
        tuple: ((x_train, y_train), (x_test, y_test))
    """
    return dataset_registry.load_custom_dataset(dataset_id, session_id)

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
