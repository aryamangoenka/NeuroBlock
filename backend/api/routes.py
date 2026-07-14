import os
import json
import tensorflow as tf
from flask import jsonify, request, send_file, Blueprint, current_app
import shutil
from backend.api.sockets import latest_training_config, x_train_shape

from backend.export.python_script import generate_python_script
from backend.export.notebook import generate_notebook
from backend.export.pytorch import generate_pytorch_script
from backend.utils.logging import get_logger

import pickle
import numpy as np
from PIL import Image
import io

# Initialize logger
logger = get_logger(__name__)

# Create a blueprint for API routes
api_blueprint = Blueprint('api', __name__, url_prefix='/api')

# Define the home route
@api_blueprint.route('/', endpoint='home')
def home():
    logger.debug("Home endpoint called")
    return jsonify({"message": "Hello, Flask!"})

@api_blueprint.route('/health', methods=['GET'])
def health_check():
    logger.debug("Health check endpoint called")
    return jsonify({"status": "running", "message": "Flask backend is operational!"})

# Model saving routes
@api_blueprint.route('/save_model', methods=["POST"])
def redirect_save_model():
    """
    Redirect /save_model to /api/save_model for API route consistency
    """
    logger.info("Redirecting from /save_model to /api/save_model")
    return save_model_api()

@api_blueprint.route('/api/save_model', methods=["POST"])
def save_model_api():
    """
    Save the model architecture and training configuration received from the frontend.
    """
    logger.info("Save model endpoint called")
    try:
        data = request.get_json()
        if not data:
            logger.warning("No JSON payload received in save_model request")
            return jsonify({"error": "No JSON payload received"}), 400

        # Get paths from app config
        model_architecture_file = current_app.config.get('MODEL_ARCHITECTURE_FILE')
        training_config_file = current_app.config.get('TRAINING_CONFIG_FILE')

        # Process and validate the architecture
        architecture = data.get('architecture', {})
        nodes = architecture.get('nodes', [])
        edges = architecture.get('edges', [])

        # Create a graph representation for topological sort
        graph = {}
        in_degree = {}
        for node in nodes:
            graph[node["id"]] = []
            in_degree[node["id"]] = 0
        
        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            if source in graph:
                graph[source].append(target)
                in_degree[target] = in_degree.get(target, 0) + 1

        # Perform topological sort
        queue = []
        for node_id, degree in in_degree.items():
            if degree == 0:
                queue.append(node_id)
        
        topo_order = []
        while queue:
            node_id = queue.pop(0)
            topo_order.append(node_id)
            
            for neighbor in graph[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(topo_order) != len(nodes):
            error_msg = "Graph contains cycles or disconnected components"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400

        # Create a mapping of node IDs to their data
        node_map = {node["id"]: node for node in nodes}

        # Reorder nodes based on topological sort
        ordered_nodes = [node_map[node_id] for node_id in topo_order]

        # Create the final architecture with ordered nodes and their connections
        final_architecture = {
            "nodes": ordered_nodes,
            "edges": edges,
            "dataset": architecture.get("dataset", ""),
            "topological_order": topo_order,  # Include the topological order for reference
            "layer_connections": {  # Add explicit layer connections
                node_id: {
                    "inputs": [edge["source"] for edge in edges if edge["target"] == node_id],
                    "outputs": [edge["target"] for edge in edges if edge["source"] == node_id]
                }
                for node_id in topo_order
            }
        }
        
        # Validate model architecture against dataset if available
        dataset_name = architecture.get("dataset", "")
        if dataset_name:
            try:
                from backend.utils.data_quality import DataQualityValidator
                from backend.utils.session_manager import get_session_datasets_dir, get_session_id
                
                # Try to load dataset metadata for validation
                try:
                    session_id = get_session_id()
                    datasets_dir = get_session_datasets_dir(session_id)
                    metadata_file = os.path.join(datasets_dir, f'{dataset_name}_metadata.json')
                    
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            dataset_metadata = json.load(f)
                        
                        # Validate architecture
                        validation_error = DataQualityValidator.validate_model_architecture(
                            dataset_metadata, final_architecture
                        )
                        
                        if validation_error:
                            logger.warning(f"Model architecture validation warning for dataset '{dataset_name}': {validation_error}")
                            # Don't fail, just warn - user may have valid reasons for their architecture
                        else:
                            logger.info(f"Model architecture validated successfully for dataset '{dataset_name}'")
                            
                except Exception as e:
                    logger.debug(f"Could not validate architecture against dataset metadata: {str(e)}")
                    # This is not critical, continue normally
                    
            except Exception as e:
                logger.debug(f"Architecture validation skipped: {str(e)}")
                # This is optional validation, don't fail the save operation

        # Save model architecture to a file
        with open(model_architecture_file, "w") as f:
            json.dump(final_architecture, f, indent=2)
        
        # Save training configuration to a file
        training_config = data.get('training_config', {})
        if training_config:
            logger.info(f"Received training config from frontend: {training_config}")
            # Use the exact values from the frontend without defaults
            config = {
                "epochs": int(training_config["epochs"]),
                "batchSize": int(training_config["batchSize"]),
                "optimizer": str(training_config["optimizer"]),
                "lossFunction": str(training_config["lossFunction"]),
                "learningRate": float(training_config["learningRate"]),
                "validationSplit": float(training_config["validationSplit"])
            }
            
            logger.info(f"Processed training config to save: {config}")
            
            # Write the configuration with proper formatting
            with open(training_config_file, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Training configuration saved to {training_config_file}")
            
            # Verify the saved configuration
            with open(training_config_file, "r") as f:
                saved_config = json.load(f)
            logger.info(f"Verified saved training config: {saved_config}")
        
        # Print ResNet model structure if ResNet blocks are present
        resnet_blocks = [node for node in nodes if node.get("type") == "resnetblock"]
        
        return jsonify({
            "message": "Model architecture and training configuration saved successfully",
            "resnet_blocks": len(resnet_blocks)
        }), 200

    except Exception as e:
        logger.error(f"Error in save_model_api: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to save model: {str(e)}"}), 500

@api_blueprint.route('/export/<format>', methods=["GET"])
def export_model(format):
    """
    Export the trained model in the specified format.
    Supported formats: py, ipynb, savedmodel, hdf5
    """
    logger.info(f"Export model endpoint called for format: {format}")
    try:
        # Get paths from app config
        export_folder = current_app.config.get('EXPORT_FOLDER')
        trained_model_path = current_app.config.get('TRAINED_MODEL_PATH')
        model_architecture_file = current_app.config.get('MODEL_ARCHITECTURE_FILE')
        training_config_file = current_app.config.get('TRAINING_CONFIG_FILE')
        
        # Log paths for debugging
        logger.debug(f"Export paths: export_folder={export_folder}, trained_model_path={trained_model_path}, model_architecture_file={model_architecture_file}, training_config_file={training_config_file}")
        
        # Make sure the export directory exists
        os.makedirs(export_folder, exist_ok=True)
        
        # Initialize variables with safe defaults
        dataset_name = ""
        model_architecture = {}
        training_config = {}
        
        # First try to get dataset name from model architecture
        if os.path.exists(model_architecture_file):
            try:
                with open(model_architecture_file, "r") as f:
                    model_arch = json.load(f)
                    dataset_name = model_arch.get("dataset", "")
                    logger.debug(f"Got dataset name from model architecture: {dataset_name}")
            except Exception as e:
                logger.warning(f"Could not load dataset name from model architecture: {str(e)}")
        
        # Only if dataset name is not found in model architecture, try training config
        if not dataset_name and os.path.exists(training_config_file):
            with open(training_config_file, "r") as f:
                training_config = json.load(f)
                config_dataset = training_config.get("dataset", "")
                if config_dataset:
                    dataset_name = config_dataset
                    logger.debug(f"Got dataset name from training config: {dataset_name}")
        
        logger.debug(f"Final dataset name for export: {dataset_name}")
        
        # Check if the saved model exists
        trained_model_exists = os.path.exists(trained_model_path)
        logger.debug(f"Trained model exists: {trained_model_exists}")
        
        # If the trained model doesn't exist, try to use the saved architecture to create a model
        if not trained_model_exists:
            logger.warning("No trained model found, attempting to use saved architecture")
            
            # Check if model architecture exists
            if not os.path.exists(model_architecture_file):
                logger.error(f"No model architecture found for export at path: {model_architecture_file}")
                return jsonify({"error": "No model found. Please create and save a model first."}), 400
            
            # Load the model architecture
            with open(model_architecture_file, "r") as f:
                model_architecture = json.load(f)
            
            if not model_architecture or not model_architecture.get("nodes"):
                logger.error("Model architecture is empty or invalid")
                return jsonify({"error": "Model architecture is invalid. Please create a valid model."}), 400
            
            # Get dataset name
            dataset_name = model_architecture.get("dataset", "")
            
            # Create a default shape based on the dataset
            if dataset_name == "MNIST":
                default_shape = (None, 28, 28, 1)
            elif dataset_name == "CIFAR-10":
                default_shape = (None, 32, 32, 3)
            elif dataset_name in ["Iris", "Breast Cancer", "California Housing"]:
                # Get the input size from the model architecture
                input_size = next((node.get("params", {}).get("units", 4) 
                                for node in model_architecture.get("nodes", [])
                                if node.get("type") == "Input"), 4)
                default_shape = (None, input_size)
            else:
                default_shape = (None, 28, 28, 1)  # Default to MNIST-like shape
            
            # Use x_train_shape if available, otherwise use the default shape
            dummy_shape = x_train_shape if (x_train_shape and len(x_train_shape) > 0) else default_shape
            logger.debug(f"Using shape for model building: {dummy_shape}")
            
            # Build model from architecture
            try:
                from backend.models.builder import build_model_from_architecture
                model = build_model_from_architecture(model_architecture, dummy_shape, dataset_name)
                logger.info("Successfully created model from saved architecture")
            except Exception as build_error:
                logger.error(f"Error building model from architecture: {str(build_error)}", exc_info=True)
                return jsonify({"error": f"Could not build model: {str(build_error)}"}), 500
        else:
            # Load the trained model
            logger.debug("Loading trained model")
            # Enable unsafe deserialization for Lambda layers
            tf.keras.config.enable_unsafe_deserialization()
            model = tf.keras.models.load_model(trained_model_path, compile=False)
        
        # Get the dataset name from the latest training config or model architecture
        config_dataset = training_config.get("dataset", "")
        if config_dataset:
            dataset_name = config_dataset
        elif not dataset_name and os.path.exists(model_architecture_file):
            try:
                with open(model_architecture_file, "r") as f:
                    model_arch = json.load(f)
                    dataset_name = model_arch.get("dataset", "")
            except Exception as e:
                logger.warning(f"Could not load dataset name from model architecture: {str(e)}")
        
        logger.debug(f"Dataset for export: {dataset_name}")
        
        # Get model input shape, with fallback
        try:
            model_input_shape = model.input_shape
        except:
            model_input_shape = dummy_shape if 'dummy_shape' in locals() else default_shape if 'default_shape' in locals() else (None, 28, 28, 1)
        
        # Use the architecture and model for export operations
        # Export according to the requested format
        if format == "py":
            logger.info("Generating Python script")
            file_path = os.path.join(export_folder, "trained_model.py")
            logger.debug(f"File path for Python export: {file_path}")
            logger.debug(f"EXPORT_FOLDER: {export_folder}")
            logger.debug(f"Current working directory: {os.getcwd()}")
            
            # Load the latest training configuration
            if os.path.exists(training_config_file):
                with open(training_config_file, 'r') as f:
                    training_config = json.load(f)
                    logger.debug(f"Loaded training config from file: {training_config}")
            
            # Make sure the file exists
            with open(file_path, "w") as f:
                f.write(generate_python_script(model, training_config, model_input_shape, dataset_name))
            logger.info(f"Python script saved to {file_path}")
            
            # Check if file exists after writing
            if os.path.exists(file_path):
                logger.debug(f"Confirmed file exists at: {file_path}")
            else:
                logger.error(f"File was not created at: {file_path}")
            
            # Use absolute path for send_file and set as_attachment to True
            return send_file(os.path.abspath(file_path), as_attachment=True)

        elif format == "ipynb":
            logger.info("Generating Jupyter notebook")
            file_path = os.path.join(export_folder, "trained_model.ipynb")
            
            # Load the latest training configuration
            if os.path.exists(training_config_file):
                with open(training_config_file, 'r') as f:
                    training_config = json.load(f)
                    logger.debug(f"Loaded training config from file: {training_config}")
            
            with open(file_path, "w") as f:
                f.write(generate_notebook(model, training_config, model_input_shape, dataset_name))
            logger.info(f"Jupyter notebook saved to {file_path}")
            # Use absolute path for send_file
            return send_file(os.path.abspath(file_path), as_attachment=True)

        elif format == "savedmodel":
            logger.info("Exporting to TensorFlow SavedModel format")
            saved_model_dir = os.path.join(export_folder, "saved_model_tf2")
            if os.path.exists(saved_model_dir):
                shutil.rmtree(saved_model_dir)

            # Keras 3: model.export() writes a TF SavedModel for serving.
            model.export(saved_model_dir)

            # Include the native Keras file alongside for reloading/fine-tuning
            keras_copy = os.path.join(saved_model_dir, "model.keras")
            model.save(keras_copy)

            readme_path = os.path.join(saved_model_dir, "README.txt")
            with open(readme_path, "w") as f:
                f.write(
                    "NeuroBlock SavedModel export\n"
                    "============================\n\n"
                    "saved_model.pb + variables/  TensorFlow SavedModel (for serving,\n"
                    "                             TF Serving, or tf.saved_model.load)\n"
                    "model.keras                  Native Keras model (for reloading in\n"
                    "                             Python: keras.models.load_model)\n"
                )

            zip_path = shutil.make_archive(saved_model_dir, "zip", saved_model_dir)
            logger.info(f"SavedModel exported to {zip_path}")
            return send_file(
                zip_path,
                as_attachment=True,
                mimetype="application/zip",
            )

        elif format == "keras":
            logger.info("Exporting to Keras format")
            file_path = os.path.join(export_folder, "trained_model.keras")
            
            try:
                # First try direct save
                model.save(file_path, include_optimizer=False)
                logger.info(f"Model saved directly to {file_path}")
            except Exception as save_error:
                logger.warning(f"Could not save model directly: {str(save_error)}")
                
                # If we already built the model from architecture above, no need to rebuild
                if trained_model_exists:
                    # Alternative approach: Rebuild the model from architecture and copy weights
                    try:
                        logger.debug("Rebuilding model from architecture for Keras export")
                        # Get the model architecture from the saved file
                        with open(model_architecture_file, "r") as f:
                            model_architecture = json.load(f)
                        
                        # Build a fresh model from the architecture
                        from backend.models.builder import build_model_from_architecture
                        fresh_model = build_model_from_architecture(
                            model_architecture, 
                            x_train_shape or model.input_shape, 
                            dataset_name
                        )
                        
                        # Copy weights from the trained model to the fresh model
                        fresh_model.set_weights(model.get_weights())
                        
                        # Save the fresh model
                        fresh_model.save(file_path, include_optimizer=False)
                        logger.info(f"Rebuilt model saved to {file_path}")
                    except Exception as rebuild_error:
                        logger.error(f"Could not save model after rebuilding: {str(rebuild_error)}", exc_info=True)
                        return jsonify({"error": f"Could not save model: {str(rebuild_error)}"}), 500
            
            # Use absolute path for send_file
            return send_file(os.path.abspath(file_path), as_attachment=True)
        
        elif format == "tflite":
            logger.info("Exporting to TensorFlow Lite format")
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            # Broad op support so attention/custom layers convert cleanly
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            tflite_model = converter.convert()
            file_path = os.path.join(export_folder, "model.tflite")
            with open(file_path, "wb") as f:
                f.write(tflite_model)
            logger.info(f"TFLite model exported to {file_path} ({len(tflite_model)} bytes)")
            return send_file(
                os.path.abspath(file_path),
                as_attachment=True,
                mimetype="application/octet-stream",
            )

        elif format == "pytorch":
            logger.info("Exporting to PyTorch format")
            # Load the latest training configuration
            if os.path.exists(training_config_file):
                with open(training_config_file, 'r') as f:
                    training_config = json.load(f)
                    logger.debug(f"Loaded training config from file: {training_config}")
            
            # Convert Keras model to PyTorch manually
            pytorch_script = generate_pytorch_script(model, training_config, model_input_shape, dataset_name)

            # Save to a file
            file_path = os.path.join(export_folder, "trained_model_pytorch.py")
            with open(file_path, "w") as f:
                f.write(pytorch_script)
            logger.info(f"PyTorch script saved to {file_path}")
            
            # Use absolute path for send_file
            return send_file(os.path.abspath(file_path), as_attachment=True)
            
        else:
            logger.warning(f"Unsupported export format: {format}")
            return jsonify({"error": f"Unsupported export format: {format}"}), 400

    except Exception as e:
        logger.error(f"Error in export_model: {str(e)}", exc_info=True)
        return jsonify({"error": f"Export failed: {str(e)}"}), 500

@api_blueprint.route("/clear_model", methods=["POST"])
def redirect_clear_model():
    """
    Redirect /clear_model to /api/clear_model for consistency
    """
    logger.info("Redirecting from /clear_model to /api/clear_model")
    return clear_saved_model()

@api_blueprint.route("/api/clear_model", methods=["POST"])
def clear_saved_model():
    """
    Clear the saved_model.json file when requested by the frontend.
    """
    logger.info("Clear model endpoint called")
    try:
        # Get model_architecture_file path from app config
        model_architecture_file = current_app.config.get('MODEL_ARCHITECTURE_FILE')
        
        if os.path.exists(model_architecture_file):
            with open(model_architecture_file, "w") as file:
                file.write('{}')  # Overwrite with empty JSON
            logger.info("saved_model.json has been cleared")
            return jsonify({"status": "success", "message": "Model cleared successfully!"}), 200
        else:
            logger.warning("Model file not found for clearing")
            return jsonify({"status": "error", "message": "Model file not found."}), 404
    except Exception as e:
        logger.error(f"Error clearing model: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@api_blueprint.route('/predict', methods=['POST'])
def predict_with_saved_model():
    """
    Generic prediction endpoint for any trained model.
    Expects JSON with:
      - input_data: dict of {feature: value}
      - (optional) dataset_name: to select the dataset/model if needed
    """
    try:
        data = request.get_json()
        if not data or 'input_data' not in data:
            return jsonify({'error': 'Missing input_data'}), 400
        
        input_data = data['input_data']
        dataset_name = data.get('dataset_name')

        # Get model path from config
        model_path = current_app.config.get('TRAINED_MODEL_PATH')
        
        # For custom datasets, try to find dataset-specific scaler first
        scaler_path = None
        builtin_datasets = ['Iris', 'MNIST', 'CIFAR-10', 'California Housing', 'Breast Cancer']
        
        if dataset_name and dataset_name not in builtin_datasets:
            # Custom dataset - look for dataset-specific scaler
            try:
                from backend.utils.session_manager import get_session_datasets_dir, get_session_id
                session_id = get_session_id()
                datasets_dir = get_session_datasets_dir(session_id)
                custom_scaler_path = os.path.join(datasets_dir, f'{dataset_name}_scaler.pkl')
                
                if os.path.exists(custom_scaler_path):
                    scaler_path = custom_scaler_path
                    logger.info(f"Using custom dataset scaler: {scaler_path}")
            except Exception as e:
                logger.warning(f"Could not find custom dataset scaler: {str(e)}")
        
        # Fallback to model-based scaler path if no custom scaler found
        if scaler_path is None:
            scaler_path = model_path.replace('.keras', '_scaler.pkl').replace('.h5', '_scaler.pkl')

        # Load feature metadata using the new generic system
        from backend.utils.feature_metadata import FeatureMetadataManager
        
        feature_metadata = FeatureMetadataManager.load_feature_metadata(
            dataset_name=dataset_name, 
            model_path=model_path
        )
        
        if not feature_metadata:
            return jsonify({'error': 'No feature metadata found. Please retrain the model or check dataset configuration.'}), 400

        # Process input data using generic processor
        try:
            input_array = FeatureMetadataManager.process_validation_input(input_data, feature_metadata)
        except ValueError as ve:
            return jsonify({'error': str(ve)}), 400

        # SCALING FIX: Don't apply scaling for custom datasets since model was trained with raw data
        # The scaler exists but the model was actually trained with unscaled data due to a bug
        if os.path.exists(scaler_path) and dataset_name in builtin_datasets:
            # Only apply scaling for built-in datasets that actually need it
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            input_array = scaler.transform(input_array)
            logger.info(f"Applied scaling for built-in dataset: {dataset_name}")
        else:
            logger.info(f"Skipping scaling for custom dataset: {dataset_name} (model expects raw inputs)")

        # Load model and make prediction
        model = tf.keras.models.load_model(model_path)
        prediction = model.predict(input_array)

        # Format prediction result using generic formatter
        result = FeatureMetadataManager.format_prediction_result(prediction, feature_metadata)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@api_blueprint.route('/predict/features', methods=['GET'])
def get_prediction_features():
    """
    Generic endpoint to get feature information for any dataset.
    """
    dataset_name = request.args.get('dataset_name')
    if not dataset_name:
        return jsonify({'error': 'dataset_name is required'}), 400

    try:
        # Define built-in datasets with their predefined features
        builtin_datasets = {
            'Iris': {
                'feature_names': ["sepal_length", "sepal_width", "petal_length", "petal_width"],
                'feature_types': ["float", "float", "float", "float"],
                'class_labels': ["setosa", "versicolor", "virginica"]
            },
            'MNIST': {
                'feature_names': [],  # Image dataset - no manual features
                'feature_types': [],
                'class_labels': [str(i) for i in range(10)]
            },
            'CIFAR-10': {
                'feature_names': [],  # Image dataset - no manual features  
                'feature_types': [],
                'class_labels': ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
            },
            'Breast Cancer': {
                'feature_names': [f"feature_{i}" for i in range(30)],
                'feature_types': ["float"] * 30,
                'class_labels': ["malignant", "benign"]
            },
            'California Housing': {
                'feature_names': [
                    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
                    "Population", "AveOccup", "Latitude", "Longitude"
                ],
                'feature_types': ["float"] * 8,
                'class_labels': None
            }
        }
        
        # PRIORITIZE built-in datasets - return their predefined features immediately
        if dataset_name in builtin_datasets:
            logger.info(f"Returning predefined features for built-in dataset: {dataset_name}")
            return jsonify(builtin_datasets[dataset_name])
        
        # Only for custom datasets, load feature metadata using the generic system
        logger.info(f"Loading custom dataset metadata for: {dataset_name}")
        from backend.utils.feature_metadata import FeatureMetadataManager
        
        model_path = current_app.config.get('TRAINED_MODEL_PATH')
        feature_metadata = FeatureMetadataManager.load_feature_metadata(
            dataset_name=dataset_name, 
            model_path=model_path
        )
        
        if not feature_metadata:
            return jsonify({'error': f'No feature metadata found for custom dataset: {dataset_name}. Please retrain the model.'}), 400
        
        # Extract relevant information for frontend
        feature_info = {
            'feature_names': feature_metadata.get('feature_names', []),
            'feature_types': feature_metadata.get('feature_types', []),
            'class_labels': feature_metadata.get('class_labels', None),
            'categorical_options': {}
        }
        
        # Add categorical options for dropdowns
        categorical_encodings = feature_metadata.get('categorical_encodings', {})
        for feature_name, encoding_info in categorical_encodings.items():
            if 'values' in encoding_info:
                feature_info['categorical_options'][feature_name] = encoding_info['values']
        
        # Check feature types quality and add recommendations
        quality_check = FeatureMetadataManager.check_feature_types_quality(feature_metadata)
        if quality_check['has_issues']:
            feature_info['quality_warning'] = {
                'message': 'Some features may be incorrectly classified',
                'issues': quality_check['issues'],
                'recommendations': quality_check['recommendations']
            }
        
        # Try to get enhanced data quality information from dataset metadata
        try:
            from backend.utils.session_manager import get_session_datasets_dir, get_session_id
            session_id = get_session_id()
            datasets_dir = get_session_datasets_dir(session_id)
            metadata_path = os.path.join(datasets_dir, f'{dataset_name}_metadata.json')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    dataset_metadata = json.load(f)
                
                # Include enhanced data quality information if available
                data_quality = dataset_metadata.get('data_quality')
                if data_quality:
                    feature_info['data_quality'] = data_quality
                    logger.debug(f"Added enhanced data quality info for {dataset_name}")
                    
        except Exception as e:
            logger.debug(f"Could not load enhanced quality info for {dataset_name}: {str(e)}")
            # This is optional, continue without enhanced quality info
        
        return jsonify(feature_info)
        
    except Exception as e:
        logger.error(f"Error getting prediction features: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error retrieving feature information: {str(e)}'}), 500 

@api_blueprint.route('/predict/image', methods=['POST'])
def predict_image():
    dataset_name = request.form.get('dataset_name') or request.args.get('dataset_name')
    if not dataset_name:
        return jsonify({'error': 'dataset_name is required'}), 400
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    image_file = request.files['image']
    try:
        # Load image
        img = Image.open(image_file.stream)
        
        # Handle built-in image datasets
        if dataset_name == 'MNIST':
            img = img.convert('L').resize((28, 28))  # Grayscale
            img_array = np.array(img).astype('float32') / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)
        elif dataset_name == 'CIFAR-10':
            img = img.convert('RGB').resize((32, 32))
            img_array = np.array(img).astype('float32') / 255.0
            img_array = img_array.reshape(1, 32, 32, 3)
        else:
            # Handle custom image datasets
            try:
                # Load custom dataset metadata to get image processing parameters
                from backend.utils.session_manager import get_session_datasets_dir, get_session_id
                session_id = get_session_id()
                datasets_dir = get_session_datasets_dir(session_id)
                metadata_path = os.path.join(datasets_dir, f'{dataset_name}_metadata.json')
                
                if not os.path.exists(metadata_path):
                    return jsonify({'error': f'Custom dataset {dataset_name} metadata not found'}), 400
                
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Check if it's an image dataset
                if metadata.get('dataset_type') != 'image':
                    return jsonify({'error': f'Dataset {dataset_name} is not an image dataset'}), 400
                
                # Get image processing parameters from metadata
                target_size = metadata.get('target_size', [224, 224])
                channels = metadata.get('channels', 3)
                
                # Process image according to dataset configuration
                if channels == 1:
                    img = img.convert('L').resize(tuple(target_size))  # Grayscale
                else:
                    img = img.convert('RGB').resize(tuple(target_size))  # RGB
                
                img_array = np.array(img).astype('float32') / 255.0
                
                # Reshape for model input
                if channels == 1:
                    img_array = img_array.reshape(1, target_size[0], target_size[1], 1)
                else:
                    img_array = img_array.reshape(1, target_size[0], target_size[1], channels)
                
                logger.info(f"Processed custom image for dataset '{dataset_name}': shape {img_array.shape}")
                
            except Exception as e:
                logger.error(f"Error processing custom image dataset '{dataset_name}': {str(e)}")
                return jsonify({'error': f'Error processing image for custom dataset: {str(e)}'}), 400
        
        # Load model and make prediction
        model_path = current_app.config.get('TRAINED_MODEL_PATH')
        model = tf.keras.models.load_model(model_path)
        prediction = model.predict(img_array)
        pred_class = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][pred_class])
        
        # Get class labels for better result formatting
        class_labels = None
        if dataset_name in ['MNIST']:
            class_labels = [str(i) for i in range(10)]
        elif dataset_name == 'CIFAR-10':
            class_labels = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
        else:
            # For custom datasets, try to get class labels from metadata
            try:
                if 'metadata' in locals() and metadata.get('class_labels'):
                    class_labels = metadata['class_labels']
            except:
                pass
        
        # Format result
        result = {
            'prediction': pred_class,
            'confidence': confidence
        }
        
        if class_labels and pred_class < len(class_labels):
            result['predicted_class'] = class_labels[pred_class]
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in image prediction: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500 

@api_blueprint.route('/session_id', methods=['GET'])
def get_session_id_route():
    try:
        from backend.utils.session_manager import get_session_id
        session_id = get_session_id()
        return jsonify({'session_id': session_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500 