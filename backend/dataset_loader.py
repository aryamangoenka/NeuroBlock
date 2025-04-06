import numpy as np
from backend.utils.logging import get_logger
from backend.datasets.iris import load_iris_dataset
from backend.datasets.mnist import load_mnist_dataset
from backend.datasets.cifar10 import load_cifar10_dataset
from backend.datasets.california import load_california_housing_dataset
from backend.datasets.breastcancer import load_breast_cancer_dataset
import json

# Initialize logger
logger = get_logger(__name__)

DATASET_LOADERS = {
    "Iris": load_iris_dataset,
    "MNIST": load_mnist_dataset,
    "CIFAR-10": load_cifar10_dataset,
    "California Housing": load_california_housing_dataset,
    "Breast Cancer": load_breast_cancer_dataset
}

MODEL_ARCHITECTURE_FILE = "saved_model.json"
# Save model architecture to a file
with open(MODEL_ARCHITECTURE_FILE, "r") as f:
    model=json.load(f)
dataset_name=model.get("dataset")

def load_dataset(dataset_name):
    """
    Load and preprocess a dataset based on its name.
    
    Args:
        dataset_name (str): Name of the dataset to load ('Iris', 'MNIST', 'CIFAR-10', 'California Housing', 'Breast Cancer')
        
    Returns:
        tuple: ((x_train, y_train), (x_test, y_test)) - Training and test data
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    if dataset_name == "Iris":
        return load_iris_dataset()
    elif dataset_name == "MNIST":
        return load_mnist_dataset()
    elif dataset_name == "CIFAR-10":
        return load_cifar10_dataset()
    elif dataset_name == "California Housing":
        return load_california_housing_dataset()
    elif dataset_name == "Breast Cancer":
        return load_breast_cancer_dataset()
    else:
        error_msg = f"Unknown dataset: {dataset_name}. Only 'Iris', 'MNIST', 'CIFAR-10', 'California Housing', and 'Breast Cancer' are supported."
        logger.error(error_msg)
        raise ValueError(error_msg)

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset_name)
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
