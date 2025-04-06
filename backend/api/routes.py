import os
import json
import tensorflow as tf
from flask import jsonify, request, send_file
import shutil
from backend.api.sockets import MODEL_ARCHITECTURE_FILE, EXPORT_FOLDER, TRAINED_MODEL_PATH, latest_training_config, x_train_shape
from backend.export.python_script import generate_python_script
from backend.export.notebook import generate_notebook
from backend.export.pytorch import generate_pytorch_script
from backend.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

def register_routes(app):
    """Register all Flask routes."""
    
    @app.route("/")
    def home():
        logger.debug("Home endpoint called")
        return jsonify({"message": "Hello, Flask!"})

    @app.route('/api/health', methods=['GET'])
    def health_check():
        logger.debug("Health check endpoint called")
        return jsonify({"status": "running", "message": "Flask backend is operational!"})
        
    @app.route("/save_model", methods=["POST"])
    def save_model():
        """
        Save the model architecture received from the frontend.
        """
        logger.info("Save model endpoint called")
        try:
            data = request.get_json()
            if not data:
                logger.warning("No JSON payload received in save_model request")
                return jsonify({"error": "No JSON payload received"}), 400

            # Save model architecture to a file
            with open(MODEL_ARCHITECTURE_FILE, "w") as f:
                json.dump(data, f)
            
            logger.info("Model architecture saved successfully")
                
            # Print ResNet model structure if ResNet blocks are present
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])
            resnet_blocks = [node for node in nodes if node.get("type") == "resnetblock"]
            
            if resnet_blocks:
                logger.info("ResNet model structure detected", 
                           extra={"context": {"resnet_blocks_count": len(resnet_blocks)}})
                
                # Create a dictionary of nodes by ID
                nodes_by_id = {node["id"]: node for node in nodes}
                
                # Create an adjacency list representation of the graph
                adjacency_list = {node["id"]: [] for node in nodes}
                for edge in edges:
                    source_id = edge["source"]
                    target_id = edge["target"]
                    if source_id in adjacency_list:
                        adjacency_list[source_id].append(target_id)
                
                # Find the input node
                input_node = next((node for node in nodes if node["type"] == "input"), None)
                if not input_node:
                    logger.warning("Input node not found in the model")
                    return jsonify({"message": "Model architecture saved successfully"}), 200
                    
                # Build model structure using BFS traversal
                logger.debug("Analyzing model structure")
                
                visited = set()
                queue = [(input_node["id"], 0)]  # (node_id, depth)
                
                while queue:
                    node_id, depth = queue.pop(0)
                    
                    if node_id in visited:
                        continue
                        
                    visited.add(node_id)
                    node = nodes_by_id[node_id]
                    
                    # Log node information
                    if node["type"] == "resnetblock":
                        block_type = node["data"].get("blockType", "Basic")
                        in_channels = node["data"].get("inChannels", 64)
                        out_channels = node["data"].get("outChannels", 64)
                        stride = node["data"].get("stride", [1, 1])
                        use_skip = node["data"].get("useSkipConnection", True)
                        downsample = node["data"].get("downsampleType", "None")
                        
                        logger.debug(f"ResNet Block: type={block_type}, in_channels={in_channels}, out_channels={out_channels}",
                                   extra={"context": {
                                       "node_id": node_id,
                                       "depth": depth,
                                       "block_type": block_type,
                                       "in_channels": in_channels,
                                       "out_channels": out_channels,
                                       "stride": stride,
                                       "use_skip_connection": use_skip,
                                       "downsample_type": downsample
                                   }})
                    else:
                        logger.debug(f"{node['type'].capitalize()} Layer",
                                   extra={"context": {
                                       "node_id": node_id,
                                       "depth": depth,
                                       "layer_type": node['type']
                                   }})
                    
                    # Add children to the queue
                    for child_id in adjacency_list.get(node_id, []):
                        queue.append((child_id, depth + 1))
                
                # Analyze ResNet architecture
                logger.info("ResNet Architecture Analysis", 
                           extra={"context": {
                               "total_blocks": len(resnet_blocks),
                               "basic_blocks": sum(1 for node in resnet_blocks if node["data"].get("blockType") == "Basic"),
                               "bottleneck_blocks": sum(1 for node in resnet_blocks if node["data"].get("blockType") == "Bottleneck"),
                               "has_batch_norm": any(node["type"] == "batchnormalization" for node in nodes)
                           }})

            return jsonify({"message": "Model architecture saved successfully"}), 200
        except Exception as e:
            logger.error(f"Error saving model architecture: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    
    @app.route("/export/<format>", methods=["GET"])
    def export_model(format):
        """
        Export the trained model in the specified format.
        Supported formats: py, ipynb, savedmodel, hdf5
        """
        logger.info(f"Export model endpoint called for format: {format}")
        try:
            # Load the latest trained model, not the dummy one
            if not os.path.exists(TRAINED_MODEL_PATH):
                logger.warning("No trained model found for export")
                return jsonify({"error": "No trained model found. Please train the model first."}), 400

            # Enable unsafe deserialization for Lambda layers
            tf.keras.config.enable_unsafe_deserialization()
            
            logger.debug("Loading trained model")
            model = tf.keras.models.load_model(TRAINED_MODEL_PATH, compile=False)
            
            # Get the dataset name from the latest training config
            dataset_name = latest_training_config.get("dataset", "")
            logger.debug(f"Dataset for export: {dataset_name}")

            # Export according to the requested format
            if format == "py":
                logger.info("Generating Python script")
                file_path = os.path.join(EXPORT_FOLDER, "trained_model.py")
                with open(file_path, "w") as f:
                    f.write(generate_python_script(model, latest_training_config, x_train_shape))
                logger.info(f"Python script saved to {file_path}")
                return send_file(file_path, as_attachment=True)

            elif format == "ipynb":
                logger.info("Generating Jupyter notebook")
                file_path = os.path.join(EXPORT_FOLDER, "trained_model.ipynb")
                with open(file_path, "w") as f:
                    f.write(generate_notebook(model, latest_training_config, x_train_shape))
                logger.info(f"Jupyter notebook saved to {file_path}")
                return send_file(file_path, as_attachment=True)

            elif format == "savedmodel":
                logger.info("Exporting to SavedModel format")
                # Save the model in TensorFlow 2 SavedModel format
                saved_model_dir = os.path.join(EXPORT_FOLDER, "saved_model_tf2")
                
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

# Load model architecture from JSON
with open('model_config.json', 'r') as f:
    model_json = json.load(f)
    
# Convert back to string for model_from_json
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
   b. Load the model architecture from model_config.json
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

# Load model from JSON
with open('model_config.json', 'r') as f:
    model_json = json.load(f)
model = model_from_json(json.dumps(model_json), custom_objects={'CustomAttentionLayer': CustomAttentionLayer})

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
                    # Define serving signature with explicit tensor conversion to avoid __dict__ descriptor issues
                    @tf.function
                    def serving_fn(inputs):
                        # Convert inputs to tensor explicitly to avoid descriptor issues
                        inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
                        return {"outputs": model(inputs_tensor, training=False)}
                    
                    # Create input signature separately
                    input_signature = [tf.TensorSpec(shape=model.input_shape, dtype=tf.float32, name="inputs")]
                    
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
                if not success:
                    try:
                        logger.debug("Attempting to save model by rebuilding from architecture")
                        # Get the model architecture from the saved file
                        with open(MODEL_ARCHITECTURE_FILE, "r") as f:
                            model_architecture = json.load(f)
                        
                        # Build a fresh model from the architecture
                        from backend.models.builder import build_model_from_architecture
                        fresh_model = build_model_from_architecture(
                            model_architecture, 
                            x_train_shape, 
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
                file_path = os.path.join(EXPORT_FOLDER, "trained_model.keras")
                
                try:
                    # First try direct save
                    model.save(file_path, include_optimizer=False)
                    logger.info(f"Model saved directly to {file_path}")
                except Exception as save_error:
                    logger.warning(f"Could not save model directly: {str(save_error)}")
                    
                    # Alternative approach: Rebuild the model from architecture and copy weights
                    try:
                        logger.debug("Rebuilding model from architecture for Keras export")
                        # Get the model architecture from the saved file
                        with open(MODEL_ARCHITECTURE_FILE, "r") as f:
                            model_architecture = json.load(f)
                        
                        # Build a fresh model from the architecture
                        from backend.models.builder import build_model_from_architecture
                        fresh_model = build_model_from_architecture(
                            model_architecture, 
                            x_train_shape, 
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
                    
                return send_file(file_path, as_attachment=True)
            
            elif format == "pytorch":
                logger.info("Exporting to PyTorch format")
                # Convert Keras model to PyTorch manually
                pytorch_script = generate_pytorch_script(model, latest_training_config, x_train_shape)

                # Save to a file
                file_path = os.path.join(EXPORT_FOLDER, "trained_model_pytorch.py")
                with open(file_path, "w") as f:
                    f.write(pytorch_script)
                
                logger.info(f"PyTorch model saved to {file_path}")
                return send_file(file_path, as_attachment=True)
                
            else:
                logger.warning(f"Unsupported export format requested: {format}")
                return jsonify({"error": f"Unsupported format: {format}"}), 400

        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            logger.error(f"Export error: {str(e)}", extra={"context": {"traceback": traceback_str}}, exc_info=True)
            return jsonify({"error": str(e), "traceback": traceback_str}), 500
    
    @app.route("/api/clear_model", methods=["POST"])
    def clear_saved_model():
        """
        Clear the saved_model.json file when requested by the frontend.
        """
        logger.info("Clear model endpoint called")
        try:
            if os.path.exists(MODEL_ARCHITECTURE_FILE):
                with open(MODEL_ARCHITECTURE_FILE, "w") as file:
                    file.write('{}')  # Overwrite with empty JSON
                logger.info("saved_model.json has been cleared")
                return jsonify({"status": "success", "message": "Model cleared successfully!"}), 200
            else:
                logger.warning("Model file not found for clearing")
                return jsonify({"status": "error", "message": "Model file not found."}), 404
        except Exception as e:
            logger.error(f"Error clearing model: {str(e)}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500 