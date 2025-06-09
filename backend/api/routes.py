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
            logger.info("Exporting to SavedModel format")
            # Save the model in TensorFlow 2 SavedModel format
            saved_model_dir = os.path.join(export_folder, "saved_model_tf2")
            
            # Clear any existing directory to avoid conflicts
            if os.path.exists(saved_model_dir):
                logger.debug(f"Removing existing directory: {saved_model_dir}")
                shutil.rmtree(saved_model_dir)
            os.makedirs(saved_model_dir, exist_ok=True)
            
            # Create a separate directory for the weights-only approach as fallback
            weights_dir = os.path.join(saved_model_dir, "weights_only")
            os.makedirs(weights_dir, exist_ok=True)
            
            # First, save weights and config separately as a guaranteed fallback
            try:
                # Save weights
                weights_path = os.path.join(weights_dir, "weights.h5")
                model.save_weights(weights_path)
                
                # Save model configuration
                config_path = os.path.join(weights_dir, "model_config.json")
                with open(config_path, 'w') as f:
                    json.dump(json.loads(model.to_json()), f, indent=2)
                
                # Create a Python script to load the model
                loader_script = os.path.join(weights_dir, "load_model.py")
                with open(loader_script, 'w') as f:
                    f.write("""
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import json

# Define custom attention layer class
@tf.keras.utils.register_keras_serializable(package='custom_layers')
class CustomAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads=8, key_dim=64, dropout=0.0, **kwargs):
        super(CustomAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout = dropout
        self.attention = None
        
    def build(self, input_shape):
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout
        )
        super(CustomAttentionLayer, self).build(input_shape)
        
    def call(self, inputs, training=None):
        return self.attention(inputs, inputs, training=training)
        
    def get_config(self):
        config = super(CustomAttentionLayer, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout': self.dropout
        })
        return config

# Define custom attention function for Lambda layer
@tf.keras.utils.register_keras_serializable(package='custom_layers')
def apply_attention(x, num_heads=8, key_dim=64, dropout=0.0):
    attention_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout
    )
    return attention_layer(x, x)

# Load model architecture from saved_model.json
with open('saved_model.json', 'r') as f:
    model_arch = json.load(f)
    
# Convert architecture to model JSON format
model_json = model_arch.get('model_json', {})
model_json_str = json.dumps(model_json)

# Create model with custom objects
custom_objects = {
    'CustomAttentionLayer': CustomAttentionLayer,
    'apply_attention': apply_attention
}
model = model_from_json(model_json_str, custom_objects=custom_objects)

# Load weights
model.load_weights('weights.h5')

# Now the model is ready to use
print("Model loaded successfully!")
""")
                
                # Create a README
                readme_path = os.path.join(weights_dir, "README.txt")
                with open(readme_path, 'w') as f:
                    f.write("""
# SavedModel with Custom Layers

This SavedModel contains custom attention layers that required special handling.

## How to Load the Model

1. Use the provided `load_model.py` script which contains all necessary custom layer definitions.
2. Or manually load using the following steps:
   a. Define the CustomAttentionLayer class (see load_model.py)
   b. Load the model architecture from saved_model.json
   c. Load the weights from weights.h5

Example:
```python
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import json

# Define custom layers (see load_model.py for full definitions)
@tf.keras.utils.register_keras_serializable(package='custom_layers')
class CustomAttentionLayer(tf.keras.layers.Layer):
    # ... (copy from load_model.py)

# Load model from saved_model.json
with open('saved_model.json', 'r') as f:
    model_arch = json.load(f)
model = model_from_json(json.dumps(model_arch.get('model_json', {})), custom_objects={'CustomAttentionLayer': CustomAttentionLayer})

# Load weights
model.load_weights('weights.h5')
```
""")
                
                logger.info("Saved weights and configuration as fallback")
            except Exception as e:
                logger.warning(f"Could not save weights and config: {str(e)}")
            
            # Now try to save the full SavedModel with different approaches
            success = False
            
            # Approach 1: Try to save with tf.saved_model.save and custom signatures
            try:
                logger.debug("Attempting to save model with tf.saved_model.save and custom signatures")
                # Define serving signature with explicit tensor conversion to avoid descriptor issues
                @tf.function
                def serving_fn(inputs):
                    # Convert inputs to tensor explicitly to avoid descriptor issues
                    inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
                    return {"outputs": model(inputs_tensor, training=False)}
                
                # Create input signature separately
                input_signature = [tf.TensorSpec(shape=model_input_shape, dtype=tf.float32, name="inputs")]
                
                # Get the concrete function
                concrete_serving_fn = serving_fn.get_concrete_function(*input_signature)
                
                # Save with concrete function
                tf.saved_model.save(
                    model,
                    saved_model_dir,
                    signatures={
                        "serving_default": concrete_serving_fn
                    },
                    options=tf.saved_model.SaveOptions(
                        experimental_custom_gradients=False,
                        save_debug_info=False
                    )
                )
                logger.info("Model saved successfully with tf.saved_model.save and custom signatures")
                success = True
            except Exception as e:
                logger.warning(f"Could not save with approach 1: {str(e)}")
            
            # Approach 2: If approach 1 failed, try rebuilding the model
            if not success and os.path.exists(model_architecture_file):
                try:
                    logger.debug("Attempting to save model by rebuilding from architecture")
                    # Get the model architecture from the saved file
                    with open(model_architecture_file, "r") as f:
                        rebuild_arch = json.load(f)
                    
                    # Build a fresh model from the architecture
                    from backend.models.builder import build_model_from_architecture
                    fresh_model = build_model_from_architecture(
                        rebuild_arch, 
                        model_input_shape, 
                        dataset_name
                    )
                    
                    # Copy weights from the trained model to the fresh model
                    fresh_model.set_weights(model.get_weights())
                    
                    # Compile the model (important for SavedModel)
                    fresh_model.compile(
                        optimizer='adam',  # Doesn't matter for inference
                        loss='mse',        # Doesn't matter for inference
                        metrics=['accuracy']
                    )
                    
                    # Create concrete functions for the model with explicit tensor conversion
                    @tf.function
                    def serving_fn(inputs):
                        # Convert inputs to tensor explicitly
                        inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
                        return {"outputs": fresh_model(inputs_tensor, training=False)}
                    
                    # Create input signature separately
                    input_signature = [tf.TensorSpec(shape=fresh_model.input_shape, dtype=tf.float32, name="inputs")]
                    
                    # Get the concrete function
                    concrete_serving_fn = serving_fn.get_concrete_function(*input_signature)
                    
                    # Save with concrete function
                    tf.saved_model.save(
                        fresh_model,
                        saved_model_dir,
                        signatures={
                            "serving_default": concrete_serving_fn
                        },
                        options=tf.saved_model.SaveOptions(
                            experimental_custom_gradients=False,
                            save_debug_info=False
                        )
                    )
                    logger.info("Fresh model saved successfully with tf.saved_model.save and custom signatures")
                    success = True
                except Exception as e:
                    logger.warning(f"Could not save with approach 2: {str(e)}")
            
            # Approach 3: If approaches 1 and 2 failed, try saving with Keras model.save
            if not success:
                try:
                    logger.debug("Attempting to save model with Keras model.save")
                    # For Keras 3, use file extension to determine format instead of save_format parameter
                    keras_saved_model_path = os.path.join(saved_model_dir, "keras_model")
                    os.makedirs(keras_saved_model_path, exist_ok=True)
                    
                    # Try saving in TF SavedModel format by using a directory path
                    model.save(
                        keras_saved_model_path,
                        include_optimizer=False
                    )
                    logger.info("Model saved successfully with model.save to directory (SavedModel format)")
                    success = True
                except Exception as e:
                    logger.warning(f"Could not save with approach 3 (directory): {str(e)}")
                    
                    # Try saving in Keras format with .keras extension
                    try:
                        keras_file_path = os.path.join(saved_model_dir, "model.keras")
                        model.save(
                            keras_file_path,
                            include_optimizer=False
                        )
                        logger.info("Model saved successfully with model.save to .keras file")
                        success = True
                    except Exception as e2:
                        logger.warning(f"Could not save with approach 3 (.keras file): {str(e2)}")
                        
                        # Try saving in HDF5 format with .h5 extension as last resort
                        try:
                            h5_file_path = os.path.join(saved_model_dir, "model.h5")
                            model.save(
                                h5_file_path,
                                include_optimizer=False
                            )
                            logger.info("Model saved successfully with model.save to .h5 file")
                            success = True
                        except Exception as e3:
                            logger.warning(f"Could not save with approach 3 (.h5 file): {str(e3)}")
            
            # If all approaches failed, create a special README
            if not success:
                logger.error("All approaches to save the model failed")
                readme_path = os.path.join(saved_model_dir, "README.txt")
                with open(readme_path, 'w') as f:
                    f.write("""
# SavedModel Export Failed

The model could not be saved in the standard SavedModel format due to custom layers.
However, we've provided the model weights and architecture in the 'weights_only' directory.

Please use the 'load_model.py' script in the 'weights_only' directory to load the model.
""")
                logger.warning("Created fallback README for weights-only approach")
            
            # Zip the SavedModel directory
            logger.debug(f"Creating zip archive of SavedModel directory: {saved_model_dir}")
            zip_path = shutil.make_archive(saved_model_dir, 'zip', saved_model_dir)
            logger.info(f"Saved model exported to zip file: {zip_path}")

            # Send the zipped SavedModel
            return send_file(
                zip_path,
                as_attachment=True,
                mimetype='application/zip'
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
        if dataset_name == 'MNIST':
            img = img.convert('L').resize((28, 28))  # Grayscale
            img_array = np.array(img).astype('float32') / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)
        elif dataset_name == 'CIFAR-10':
            img = img.convert('RGB').resize((32, 32))
            img_array = np.array(img).astype('float32') / 255.0
            img_array = img_array.reshape(1, 32, 32, 3)
        else:
            return jsonify({'error': 'Unsupported image dataset'}), 400
        # Load model
        model_path = current_app.config.get('TRAINED_MODEL_PATH')
        model = tf.keras.models.load_model(model_path)
        prediction = model.predict(img_array)
        pred_class = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][pred_class])
        return jsonify({'prediction': pred_class, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500 

@api_blueprint.route('/session_id', methods=['GET'])
def get_session_id_route():
    try:
        from backend.utils.session_manager import get_session_id
        session_id = get_session_id()
        return jsonify({'session_id': session_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500 