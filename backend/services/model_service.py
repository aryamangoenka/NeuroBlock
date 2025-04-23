"""
Model service module providing operations for neural network models.

This module contains the business logic for saving, loading, exporting,
and managing neural network models.
"""

import os
import json
import tensorflow as tf
import shutil
from flask import current_app
from backend.utils.logging import get_logger
from backend.models.builder import build_model_from_architecture

# Initialize logger
logger = get_logger(__name__)

class ModelService:
    """Service for managing neural network models."""
    
    @staticmethod
    def save_model_architecture(model_data):
        """
        Save the model architecture received from the frontend.
        
        Args:
            model_data (dict): Model architecture data
            
        Returns:
            bool: True if saved successfully, False otherwise
            
        Raises:
            ValueError: If model_data is invalid
        """
        if not model_data:
            logger.warning("No model data received for saving")
            raise ValueError("No model data received")
            
        # Get the model architecture file path from the app config
        model_architecture_file = current_app.config.get('MODEL_ARCHITECTURE_FILE')
        
        # Save model architecture to the file
        with open(model_architecture_file, "w") as f:
            json.dump(model_data, f)
        
        logger.info("Model architecture saved successfully")
        return True
        
    @staticmethod
    def load_model_architecture():
        """
        Load the saved model architecture.
        
        Returns:
            dict: Model architecture data, or empty dict if not found
        """
        # Get the model architecture file path from the app config
        model_architecture_file = current_app.config.get('MODEL_ARCHITECTURE_FILE')
        
        # Check if the file exists
        if not os.path.exists(model_architecture_file):
            logger.warning("Model architecture file not found")
            return {}
            
        # Load the model architecture
        try:
            with open(model_architecture_file, "r") as f:
                model_architecture = json.load(f)
            logger.info("Model architecture loaded successfully")
            return model_architecture
        except Exception as e:
            logger.error(f"Error loading model architecture: {str(e)}", exc_info=True)
            return {}
    
    @staticmethod
    def clear_model_architecture():
        """
        Clear the saved model architecture.
        
        Returns:
            bool: True if cleared successfully, False if file not found
        """
        # Get the model architecture file path from the app config
        model_architecture_file = current_app.config.get('MODEL_ARCHITECTURE_FILE')
        
        # Check if the file exists
        if not os.path.exists(model_architecture_file):
            logger.warning("Model file not found for clearing")
            return False
            
        # Clear the file
        with open(model_architecture_file, "w") as file:
            file.write('{}')  # Overwrite with empty JSON
        logger.info("Model architecture has been cleared")
        return True
    
    @staticmethod
    def load_trained_model():
        """
        Load the trained model.
        
        Returns:
            tf.keras.Model: Trained model, or None if not found
        """
        # Get the trained model path from the app config
        trained_model_path = current_app.config.get('TRAINED_MODEL_PATH')
        
        # Check if the file exists
        if not os.path.exists(trained_model_path):
            logger.warning("Trained model file not found")
            return None
            
        # Load the model
        try:
            # Enable unsafe deserialization for Lambda layers
            tf.keras.config.enable_unsafe_deserialization()
            model = tf.keras.models.load_model(trained_model_path, compile=False)
            logger.info("Trained model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading trained model: {str(e)}", exc_info=True)
            return None
    
    @staticmethod
    def build_model_from_architecture(model_architecture, input_shape, dataset_name):
        """
        Build a model from architecture data.
        
        Args:
            model_architecture (dict): Model architecture data
            input_shape (tuple): Shape of the input data
            dataset_name (str): Name of the dataset
            
        Returns:
            tf.keras.Model: Built model
            
        Raises:
            ValueError: If model architecture is invalid
        """
        if not model_architecture or not model_architecture.get("nodes"):
            logger.error("Model architecture is empty or invalid")
            raise ValueError("Model architecture is invalid")
            
        # Build the model
        try:
            model = build_model_from_architecture(model_architecture, input_shape, dataset_name)
            logger.info("Model built successfully from architecture")
            return model
        except Exception as e:
            logger.error(f"Error building model from architecture: {str(e)}", exc_info=True)
            raise ValueError(f"Could not build model: {str(e)}")
    
    # Additional methods for model operations can be added here 