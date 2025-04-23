import numpy as np
import os
from flask import current_app
from backend.utils.logging import get_logger
import json

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
        if dataset_name not in self.loaders:
            error_msg = f"Unknown dataset: {dataset_name}. Available datasets: {', '.join(self.loaders.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        return self.loaders[dataset_name]
        
    def get_available_datasets(self):
        """
        Get a list of available datasets
        
        Returns:
            list: Names of available datasets
        """
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

if __name__ == "__main__":
    # Use the dataset specified in the saved model, or default to MNIST
    dataset_name = get_model_dataset() or "MNIST"
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset_name)
    print(f"Dataset: {dataset_name}")
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
