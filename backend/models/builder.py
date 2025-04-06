import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, Input, Reshape, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from backend.models.custom_layers import CustomAttentionLayer
from backend.models.resnet import create_resnet_block
from backend.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

def determine_output_units(dataset_name):
    """
    Determine the number of units for the output layer based on the dataset.

    Args:
        dataset_name (str): The name of the dataset (e.g., 'Iris', 'MNIST', 'CIFAR-10', 'California Housing', 'Breast Cancer').

    Returns:
        int: The number of units for the output layer.
    """
    logger.debug(f"Determining output units for dataset: {dataset_name}")
    
    if dataset_name == "Iris":
        return 3  # 3 classes
    elif dataset_name == "MNIST":
        return 10  # 10 digits
    elif dataset_name == "CIFAR-10":
        return 10  # 10 classes
    elif dataset_name == "California Housing":
        return 1  # Regression
    elif dataset_name == "Breast Cancer":
        return 1  # Binary classification
    else:
        error_msg = f"Unknown dataset: {dataset_name}. Only 'Iris', 'MNIST', 'CIFAR-10', 'California Housing', and 'Breast Cancer' are supported."
        logger.error(error_msg)
        raise ValueError(error_msg)

def build_model_from_architecture(architecture, input_shape, dataset_name):
    """
    Build a Keras model based on the architecture provided.

    Args:
        architecture (dict): The model architecture containing nodes and edges.
        input_shape (tuple): Shape of the input data.
        dataset_name (str): Name of the dataset.

    Returns:
        keras.Model: A compiled Keras model.
    """
    logger.info("Building model from architecture", 
                extra={
                    "context": {
                        "dataset": dataset_name,
                        "input_shape": str(input_shape)
                    }
                })
                
    nodes = architecture["nodes"]
    edges = architecture["edges"]

    # Validate input and output layers
    input_layer = next((node for node in nodes if node["type"] == "input"), None)
    output_layer = next((node for node in nodes if node["type"] == "output"), None)
    
    if not input_layer or not output_layer:
        error_msg = "Model must have both an input and an output layer."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Check if we need to use a functional API (for models with skip connections like ResNet)
    has_resnet_blocks = any(node["type"] == "resnetblock" for node in nodes)
    
    if has_resnet_blocks:
        logger.info("Using functional API for model with ResNet blocks")
        # Use Functional API for models with skip connections
        inputs = Input(shape=input_shape)
        
        # Create a dictionary of layers by node ID
        layers_by_id = {}
        layers_by_id[input_layer["id"]] = inputs
        
        # Create a dictionary to track which nodes have been processed
        processed_nodes = {input_layer["id"]: True}
        
        # Process nodes in topological order
        nodes_to_process = [node for node in nodes if node["id"] != input_layer["id"]]
        
        while nodes_to_process:
            for node in list(nodes_to_process):
                # Find all incoming edges to this node
                incoming_edges = [edge for edge in edges if edge["target"] == node["id"]]
                
                # Check if all source nodes of incoming edges have been processed
                if all(edge["source"] in processed_nodes for edge in incoming_edges):
                    # Get the input layer(s) for this node
                    if len(incoming_edges) == 1:
                        # Single input
                        input_layer = layers_by_id[incoming_edges[0]["source"]]
                    else:
                        # Multiple inputs (not typical in sequential models, but handle it)
                        input_layer = [layers_by_id[edge["source"]] for edge in incoming_edges]
                    
                    # Process the node based on its type
                    layer_type = node["type"]
                    layer_data = node["data"]
                    
                    logger.debug(f"Processing {layer_type} layer (id: {node['id']})")
                    
                    if layer_type == "resnetblock":
                        # Create ResNet block
                        x = create_resnet_block(
                            input_layer, 
                            block_type=layer_data.get("blockType", "Basic"),
                            in_channels=layer_data.get("inChannels", 64),
                            out_channels=layer_data.get("outChannels", 64),
                            stride=layer_data.get("stride", [1, 1]),
                            activation=layer_data.get("activation", "ReLU").lower(),
                            use_skip_connection=layer_data.get("useSkipConnection", True),
                            downsample_type=layer_data.get("downsampleType", "None")
                        )
                    elif layer_type == "dense":
                        x = Dense(
                            units=layer_data["neurons"],
                            activation=None if layer_data["activation"].lower() == "none" else layer_data["activation"].lower()
                        )(input_layer)
                    elif layer_type == "convolution":
                        x = Conv2D(
                            filters=layer_data["filters"],
                            kernel_size=tuple(layer_data["kernelSize"]),
                            strides=tuple(layer_data["stride"]),
                            padding=layer_data.get("padding", "same").lower(),
                            activation=None if layer_data["activation"].lower() == "none" else layer_data["activation"].lower()
                        )(input_layer)
                    elif layer_type == "maxpooling":
                        # Check for potential size issues with MaxPooling
                        pool_size = tuple(layer_data["poolSize"])
                        
                        # For ResNet models, handle large pool sizes that might cause issues
                        if has_resnet_blocks and (pool_size[0] > 3 or pool_size[1] > 3):
                            # Use safer pooling parameters
                            logger.warning(f"Large pool size {pool_size} detected in ResNet model, using safer parameters")
                            safe_pool_size = (min(pool_size[0], 3), min(pool_size[1], 3))
                            x = MaxPooling2D(
                                pool_size=safe_pool_size,
                                strides=tuple(layer_data["stride"]),
                                padding=layer_data.get("padding", "same").lower()
                            )(input_layer)
                        else:
                            # Use regular pooling
                            x = MaxPooling2D(
                                pool_size=pool_size,
                                strides=tuple(layer_data["stride"]),
                                padding=layer_data.get("padding", "same").lower()
                            )(input_layer)
                    elif layer_type == "globalaveragepool":
                        # Add support for Global Average Pooling
                        x = GlobalAveragePooling2D()(input_layer)
                    elif layer_type == "flatten":
                        x = Flatten()(input_layer)
                    elif layer_type == "dropout":
                        x = Dropout(rate=layer_data["rate"])(input_layer)
                    elif layer_type == "batchnormalization":
                        x = BatchNormalization(
                            momentum=layer_data["momentum"],
                            epsilon=layer_data["epsilon"]
                        )(input_layer)
                    elif layer_type == "attention":
                        # Get the attention parameters
                        num_heads = layer_data.get("heads", 8)
                        key_dim = layer_data.get("keyDim", 64)
                        dropout_rate = layer_data.get("dropout", 0.0)
                        
                        # For MNIST data, we need to reshape the input for attention
                        if dataset_name == "MNIST" or dataset_name == "CIFAR-10":
                            logger.debug(f"Applying attention layer with reshaping for {dataset_name} data")
                            # For image data, we need to reshape before applying attention
                            # First, add a reshape layer to convert 2D image data to sequence data
                            reshaped = Reshape((-1, input_shape[0]))(input_layer)  # Reshape to (sequence_length, features)
                            
                            # Use our custom attention layer
                            attention_output = CustomAttentionLayer(
                                num_heads=num_heads,
                                key_dim=key_dim,
                                dropout=dropout_rate,
                                name=f"attention_{node['id']}"
                            )(reshaped)
                            
                            # Reshape back to original shape for subsequent layers if needed
                            x = Reshape(input_shape)(attention_output)
                        else:
                            logger.debug("Applying attention layer directly")
                            # For non-image data, apply attention directly
                            x = CustomAttentionLayer(
                                num_heads=num_heads,
                                key_dim=key_dim,
                                dropout=dropout_rate,
                                name=f"attention_{node['id']}"
                            )(input_layer)
                    elif layer_type == "output":
                        # For ResNet models, check if we should add a GlobalAveragePooling layer
                        # before the output layer if there's a feature map
                        if has_resnet_blocks and len(K.int_shape(input_layer)) > 2:
                            # Get the shape of the input tensor
                            input_shape = K.int_shape(input_layer)
                            
                            # If the input has spatial dimensions (height and width), add GlobalAveragePooling
                            if len(input_shape) == 4 and input_shape[1] is not None and input_shape[2] is not None:
                                logger.info(f"Adding automatic GlobalAveragePooling2D before output for ResNet (input shape: {input_shape})")
                                pooled = GlobalAveragePooling2D()(input_layer)
                                input_layer = pooled
                                
                        # Configure the output layer dynamically based on the dataset
                        output_units = determine_output_units(dataset_name)
                        activation = layer_data["activation"].lower()
                        if activation == "none":
                            activation = None
                        x = Dense(
                            units=output_units,
                            activation=activation
                        )(input_layer)
                    
                    # Store the output layer in the dictionary
                    layers_by_id[node["id"]] = x
                    
                    # Mark this node as processed
                    processed_nodes[node["id"]] = True
                    
                    # Remove this node from the list of nodes to process
                    nodes_to_process.remove(node)
        
        # Build and return the model
        model = Model(inputs=inputs, outputs=layers_by_id[output_layer["id"]])
        
    else:
        logger.info("Using Sequential API for model without skip connections")
        # Start building the model using Sequential API (for models without skip connections)
        model = Sequential()
        model.add(Input(shape=input_shape))  # Always start with an input layer

        # Add layers based on the nodes and edges
        # First, create a dictionary of nodes by ID for easy lookup
        nodes_by_id = {node["id"]: node for node in nodes}
        
        # Create a dictionary to track which nodes have been processed
        processed_nodes = {input_layer["id"]: True}
        
        # Start with the input layer and follow the edges
        current_layer_ids = [input_layer["id"]]
        
        while current_layer_ids:
            next_layer_ids = []
            
            for layer_id in current_layer_ids:
                # Find all edges that start from this layer
                outgoing_edges = [edge for edge in edges if edge["source"] == layer_id]
                
                for edge in outgoing_edges:
                    target_id = edge["target"]
                    
                    # Skip if already processed
                    if target_id in processed_nodes:
                        continue
                    
                    target_node = nodes_by_id[target_id]
                    layer_type = target_node["type"]
                    layer_data = target_node["data"]
                    
                    logger.debug(f"Adding {layer_type} layer to sequential model (id: {target_id})")
                    
                    # Add the appropriate layer based on type
                    if layer_type == "dense":
                        model.add(Dense(
                            units=layer_data["neurons"],
                            activation=None if layer_data["activation"].lower() == "none" else layer_data["activation"].lower()
                        ))
                    elif layer_type == "convolution":
                        model.add(Conv2D(
                            filters=layer_data["filters"],
                            kernel_size=tuple(layer_data["kernelSize"]),
                            strides=tuple(layer_data["stride"]),
                            activation=None if layer_data["activation"].lower() == "none" else layer_data["activation"].lower()
                        ))
                    elif layer_type == "maxpooling":
                        model.add(MaxPooling2D(
                            pool_size=tuple(layer_data["poolSize"]),
                            strides=tuple(layer_data["stride"]),
                            padding=layer_data.get("padding", "same").lower()
                        ))
                    elif layer_type == "globalaveragepool":
                        model.add(GlobalAveragePooling2D())
                    elif layer_type == "flatten":
                        model.add(Flatten())
                    elif layer_type == "dropout":
                        model.add(Dropout(rate=layer_data["rate"]))
                    elif layer_type == "batchnormalization":
                        model.add(BatchNormalization(
                            momentum=layer_data["momentum"],
                            epsilon=layer_data["epsilon"]
                        ))
                    elif layer_type == "attention":
                        # Get the attention parameters
                        num_heads = layer_data.get("heads", 8)
                        key_dim = layer_data.get("keyDim", 64)
                        dropout_rate = layer_data.get("dropout", 0.0)
                        
                        # For MNIST or CIFAR-10, we need special handling with reshaping
                        if dataset_name == "MNIST" or dataset_name == "CIFAR-10":
                            logger.debug(f"Adding attention layer with reshaping for {dataset_name} data")
                            # For image data, we need to reshape before applying attention
                            # First, add a reshape layer to convert 2D image data to sequence data
                            model.add(tf.keras.layers.Reshape((-1, input_shape[0])))  # Reshape to (sequence_length, features)
                            
                            # Use our custom attention layer instead of Lambda
                            model.add(CustomAttentionLayer(
                                num_heads=num_heads,
                                key_dim=key_dim,
                                dropout=dropout_rate,
                                name=f"attention_{target_id}"
                            ))
                            
                            # Reshape back to original shape for subsequent layers if needed
                            model.add(tf.keras.layers.Reshape(input_shape))
                        else:
                            logger.debug("Adding attention layer directly")
                            # For non-image data, apply attention directly using our custom attention layer
                            model.add(CustomAttentionLayer(
                                num_heads=num_heads,
                                key_dim=key_dim,
                                dropout=dropout_rate,
                                name=f"attention_{target_id}"
                            ))
                    elif layer_type == "output":
                        # Configure the output layer dynamically based on the dataset
                        output_units = determine_output_units(dataset_name)
                        activation = layer_data["activation"].lower()
                        if activation == "none":
                            activation = None
                        model.add(Dense(
                            units=output_units,
                            activation=activation
                        ))
                    
                    # Mark this node as processed
                    processed_nodes[target_id] = True
                    
                    # Add to the next layer IDs to process
                    if layer_type != "output":  # Don't process beyond output layer
                        next_layer_ids.append(target_id)
            
            # Update current layer IDs for the next iteration
            current_layer_ids = next_layer_ids
    
    logger.info("Model building completed")
    
    return model 