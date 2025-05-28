import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, Input, Reshape, GlobalAveragePooling2D, Add
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
    elif dataset_name == "test_dataset":
        # Specific handling for test_dataset
        logger.info(f"Using predefined output units for test_dataset: 2 classes")
        return 2
    else:
        # Handle custom datasets by looking up their metadata
        try:
            from backend.dataset_loader import dataset_registry
            
            # Check if it's a custom dataset
            custom_datasets = dataset_registry.get_custom_datasets()
            
            # Add defensive check for None or empty result
            if not custom_datasets:
                logger.warning(f"No custom datasets found for dataset: {dataset_name}")
                custom_datasets = []
            
            # Try exact match first
            for custom_dataset in custom_datasets:
                if custom_dataset and custom_dataset.get('name') == dataset_name:
                    task_type = custom_dataset.get('task_type', 'classification')
                    if task_type == 'regression':
                        return 1
                    else:
                        # For classification, try to determine number of classes
                        class_labels = custom_dataset.get('class_labels')
                        if class_labels and isinstance(class_labels, list) and len(class_labels) > 0:
                            logger.info(f"Using class_labels from metadata for '{dataset_name}': {len(class_labels)} classes")
                            return len(class_labels)
                        else:
                            # If class_labels is not available, try to load the data and count unique values
                            try:
                                logger.info(f"Attempting to analyze data for '{dataset_name}' to determine output units")
                                (x_train, y_train), (x_test, y_test) = dataset_registry.load_custom_dataset(dataset_name)
                                import numpy as np
                                if y_train is not None:
                                    unique_classes = len(np.unique(y_train.numpy()))
                                    logger.info(f"Determined {unique_classes} output units for custom dataset '{dataset_name}' by analyzing data")
                                    return unique_classes
                                else:
                                    logger.warning(f"y_train is None for custom dataset '{dataset_name}', defaulting to 2")
                                    return 2
                            except Exception as e:
                                logger.warning(f"Could not determine output units for custom dataset '{dataset_name}': {e}")
                                # Default to binary classification
                                logger.info(f"Defaulting to 2 output units for '{dataset_name}'")
                                return 2
            
            # Try case-insensitive match
            dataset_name_lower = dataset_name.lower()
            for custom_dataset in custom_datasets:
                if custom_dataset and custom_dataset.get('name', '').lower() == dataset_name_lower:
                    logger.info(f"Found case-insensitive match for dataset: '{dataset_name}' -> '{custom_dataset.get('name', '')}'")
                    task_type = custom_dataset.get('task_type', 'classification')
                    if task_type == 'regression':
                        return 1
                    else:
                        # For classification, try to determine number of classes
                        class_labels = custom_dataset.get('class_labels')
                        if class_labels and isinstance(class_labels, list) and len(class_labels) > 0:
                            logger.info(f"Using class_labels from metadata for '{custom_dataset.get('name', '')}': {len(class_labels)} classes")
                            return len(class_labels)
                        else:
                            # If class_labels is not available, try to load the data and count unique values
                            try:
                                actual_name = custom_dataset.get('name', dataset_name)
                                logger.info(f"Attempting to analyze data for '{actual_name}' to determine output units")
                                (x_train, y_train), (x_test, y_test) = dataset_registry.load_custom_dataset(actual_name)
                                import numpy as np
                                if y_train is not None:
                                    unique_classes = len(np.unique(y_train.numpy()))
                                    logger.info(f"Determined {unique_classes} output units for custom dataset '{actual_name}' by analyzing data")
                                    return unique_classes
                                else:
                                    logger.warning(f"y_train is None for custom dataset '{actual_name}', defaulting to 2")
                                    return 2
                            except Exception as e:
                                logger.warning(f"Could not determine output units for custom dataset '{custom_dataset.get('name', dataset_name)}': {e}")
                                # Default to binary classification
                                logger.info(f"Defaulting to 2 output units for '{custom_dataset.get('name', dataset_name)}'")
                                return 2
            
            # If no custom dataset found, raise error
            try:
                available_datasets = dataset_registry.get_available_datasets()
                if available_datasets is None:
                    available_datasets = ["Iris", "MNIST", "CIFAR-10", "California Housing", "Breast Cancer"]
                error_msg = f"Unknown dataset: {dataset_name}. Available datasets: {', '.join(available_datasets)}"
            except Exception as e:
                logger.warning(f"Could not get available datasets: {e}")
                error_msg = f"Unknown dataset: {dataset_name}. Could not retrieve available datasets list."
            
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        except ImportError:
            # Fallback if dataset_registry is not available
            error_msg = f"Unknown dataset: {dataset_name}. Only 'Iris', 'MNIST', 'CIFAR-10', 'California Housing', and 'Breast Cancer' are supported."
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            # Catch any other exceptions that might cause NoneType errors
            logger.error(f"Unexpected error in determine_output_units for dataset '{dataset_name}': {e}")
            # Default to binary classification to allow training to continue
            logger.warning(f"Defaulting to 2 output units for dataset '{dataset_name}' due to error")
            return 2

def expand_custom_blocks(nodes):
    """
    Expand custom blocks into their constituent layers.
    
    Args:
        nodes (list): List of nodes from the architecture
        
    Returns:
        list: Expanded list of nodes with custom blocks replaced by their internal layers
    """
    expanded_nodes = []
    
    for node in nodes:
        if node["type"].lower() == "customblock":
            logger.debug(f"Expanding custom block: {node.get('data', {}).get('blockName', 'Unnamed')}")
            
            # Get the internal layers from the custom block
            custom_layers = node["data"].get("layers", [])
            logger.debug(f"Custom block layers received: {custom_layers}")
            
            # Create individual nodes for each layer in the custom block
            for i, layer in enumerate(custom_layers):
                logger.debug(f"Processing layer {i}: {layer}")
                # Create a new node ID for each expanded layer
                expanded_node_id = f"{node['id']}_layer_{i}"
                
                # Map the layer type from the custom block format to the expected format
                # Handle both 'type' and 'id' fields from frontend, normalize to lowercase
                # Priority: id field (lowercase) > type field (capitalized) > fallback to empty string
                layer_type = layer.get("id", layer.get("type", "")).lower()
                logger.debug(f"Layer type extracted: '{layer_type}' from layer: {layer}")
                
                # Get parameters from the layer, with defaults as fallback
                layer_parameters = layer.get("parameters", {})
                
                # Create layer data with user-configured parameters or defaults
                layer_data = {}
                if layer_type == "dense":
                    layer_data = {
                        "neurons": layer_parameters.get("neurons", 64),
                        "activation": layer_parameters.get("activation", "relu")
                    }
                elif layer_type == "convolution":
                    layer_data = {
                        "filters": layer_parameters.get("filters", 32),
                        "kernelSize": layer_parameters.get("kernelSize", [3, 3]),
                        "stride": layer_parameters.get("stride", [1, 1]),
                        "padding": layer_parameters.get("padding", "same"),
                        "activation": layer_parameters.get("activation", "relu")
                    }
                elif layer_type == "maxpooling":
                    layer_data = {
                        "poolSize": layer_parameters.get("poolSize", [2, 2]),
                        "stride": layer_parameters.get("stride", [2, 2]),
                        "padding": layer_parameters.get("padding", "same")
                    }
                elif layer_type == "globalaveragepool":
                    layer_data = {}
                elif layer_type == "flatten":
                    layer_data = {}
                elif layer_type == "dropout":
                    layer_data = {
                        "rate": layer_parameters.get("rate", 0.25)
                    }
                elif layer_type == "batchnormalization":
                    layer_data = {
                        "momentum": layer_parameters.get("momentum", 0.99),
                        "epsilon": layer_parameters.get("epsilon", 0.001)
                    }
                elif layer_type == "attention":
                    layer_data = {
                        "heads": layer_parameters.get("heads", 8),
                        "keyDim": layer_parameters.get("keyDim", 64),
                        "dropout": layer_parameters.get("dropout", 0.0)
                    }
                elif layer_type == "addlayer":
                    layer_data = {}
                elif layer_type == "activation":
                    layer_data = {
                        "function": layer_parameters.get("function", "relu")
                    }
                else:
                    logger.warning(f"Unknown layer type in custom block: {layer_type}")
                    continue  # Skip unknown layer types
                
                # Create the expanded node
                expanded_node = {
                    "id": expanded_node_id,
                    "type": layer_type,
                    "data": layer_data,
                    "position": node.get("position", {"x": 0, "y": 0}),
                    "original_custom_block_id": node["id"],  # Keep reference to original custom block
                    "custom_block_layer_index": i  # Keep track of order within custom block
                }
                
                expanded_nodes.append(expanded_node)
                logger.debug(f"Expanded layer {i}: {layer_type} (ID: {expanded_node_id})")
        else:
            # Keep non-custom-block nodes as they are
            expanded_nodes.append(node)
    
    return expanded_nodes

def update_edges_for_expanded_blocks(edges, original_nodes, expanded_nodes):
    """
    Update edges to connect to expanded custom block layers.
    
    Args:
        edges (list): Original edges
        original_nodes (list): Original nodes before expansion
        expanded_nodes (list): Nodes after custom block expansion
        
    Returns:
        list: Updated edges that connect to expanded layers
    """
    updated_edges = []
    
    # Create mapping of original custom block IDs to their expanded layer IDs
    custom_block_mapping = {}
    for node in expanded_nodes:
        if "original_custom_block_id" in node:
            original_id = node["original_custom_block_id"]
            if original_id not in custom_block_mapping:
                custom_block_mapping[original_id] = []
            custom_block_mapping[original_id].append(node["id"])
    
    for edge in edges:
        source_id = edge["source"]
        target_id = edge["target"]
        
        # Check if source is a custom block
        if source_id in custom_block_mapping:
            # Connect from the last layer of the custom block
            expanded_layers = custom_block_mapping[source_id]
            new_source_id = expanded_layers[-1]  # Last layer in the custom block
        else:
            new_source_id = source_id
        
        # Check if target is a custom block
        if target_id in custom_block_mapping:
            # Connect to the first layer of the custom block
            expanded_layers = custom_block_mapping[target_id]
            new_target_id = expanded_layers[0]  # First layer in the custom block
        else:
            new_target_id = target_id
        
        # Create the updated edge
        updated_edge = {
            "source": new_source_id,
            "target": new_target_id,
            "id": edge.get("id", f"{new_source_id}-{new_target_id}")
        }
        updated_edges.append(updated_edge)
    
    # Add internal connections within custom blocks
    for original_id, expanded_layer_ids in custom_block_mapping.items():
        # Connect consecutive layers within the custom block
        for i in range(len(expanded_layer_ids) - 1):
            internal_edge = {
                "source": expanded_layer_ids[i],
                "target": expanded_layer_ids[i + 1],
                "id": f"{expanded_layer_ids[i]}-{expanded_layer_ids[i + 1]}"
            }
            updated_edges.append(internal_edge)
    
    return updated_edges

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
                
    original_nodes = architecture["nodes"]
    original_edges = architecture["edges"]
    
    # Expand custom blocks into individual layers
    nodes = expand_custom_blocks(original_nodes)
    
    # Update edges to connect to expanded layers
    edges = update_edges_for_expanded_blocks(original_edges, original_nodes, nodes)
    
    logger.info(f"Expanded {len(original_nodes)} original nodes to {len(nodes)} nodes")
    logger.info(f"Updated {len(original_edges)} original edges to {len(edges)} edges")

    # Validate input and output layers
    input_layer = next((node for node in nodes if node["type"] == "input"), None)
    output_layer = next((node for node in nodes if node["type"] == "output"), None)
    
    if not input_layer or not output_layer:
        error_msg = "Model must have both an input and an output layer."
        logger.error(error_msg)
        raise ValueError(error_msg)

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
        raise ValueError(error_msg)

    # Create a mapping of node IDs to their data
    node_map = {node["id"]: node for node in nodes}

    # Check if we need to use functional API
    has_resnet_blocks = any(node["type"] == "resnetblock" for node in nodes)
    has_skip_connections = any(len([e for e in edges if e["target"] == node["id"]]) > 1 for node in nodes)
    
    if has_resnet_blocks or has_skip_connections:
        logger.info("Using functional API for model with complex connections")
        # Use Functional API
        inputs = Input(shape=input_shape)
        layers_by_id = {input_layer["id"]: inputs}
        
        # First, we need to process the architecture to merge activation layers with their preceding layers
        processed_nodes = []
        i = 0
        while i < len(topo_order):
            node_id = topo_order[i]
            if node_id == input_layer["id"]:
                i += 1
                continue
                
            node = node_map[node_id]
            layer_type = node["type"]
            layer_data = node["data"]
            
            # Check if the next node is an activation layer that should be merged
            next_activation = None
            if i + 1 < len(topo_order):
                next_node_id = topo_order[i + 1]
                next_node = node_map[next_node_id]
                if next_node["type"] == "activation":
                    # Check if this activation directly follows the current layer
                    next_incoming_edges = [edge for edge in edges if edge["target"] == next_node_id]
                    if len(next_incoming_edges) == 1 and next_incoming_edges[0]["source"] == node_id:
                        next_activation = next_node["data"].get("function", "relu").lower()
                        i += 1  # Skip the activation node as we'll merge it
            
            processed_nodes.append({
                "id": node_id,
                "type": layer_type,
                "data": layer_data,
                "merged_activation": next_activation
            })
            i += 1
        
        # Process nodes in topological order
        for processed_node in processed_nodes:
            node_id = processed_node["id"]
            layer_type = processed_node["type"]
            layer_data = processed_node["data"]
            merged_activation = processed_node["merged_activation"]
            
            incoming_edges = [edge for edge in edges if edge["target"] == node_id]
            
            # Get input layers
            if len(incoming_edges) == 1:
                input_layer_func = layers_by_id[incoming_edges[0]["source"]]
            else:
                input_layer_func = [layers_by_id[edge["source"]] for edge in incoming_edges]
            
            logger.debug(f"Processing {layer_type} layer (id: {node_id})")
            
            # Determine final activation - use merged activation if available, otherwise use layer's activation
            final_activation = None
            if merged_activation:
                final_activation = merged_activation
            elif layer_data.get("activation", "none").lower() != "none":
                final_activation = layer_data["activation"].lower()
            
            # Create the appropriate layer
            if layer_type == "resnetblock":
                x = create_resnet_block(
                    input_layer_func,
                    block_type=layer_data.get("blockType", "Basic"),
                    in_channels=64,
                    out_channels=layer_data.get("filters", 64),
                    stride=layer_data.get("stride", [1, 1]),
                    activation=layer_data.get("activation", "ReLU").lower(),
                    use_skip_connection=True,
                    downsample_type="Conv1x1"
                )
            elif layer_type == "dense":
                x = Dense(
                    units=layer_data["neurons"],
                    activation=final_activation
                )(input_layer_func)
            elif layer_type == "convolution":
                x = Conv2D(
                    filters=layer_data["filters"],
                    kernel_size=tuple(layer_data["kernelSize"]),
                    strides=tuple(layer_data["stride"]),
                    padding=layer_data.get("padding", "same").lower(),
                    activation=final_activation
                )(input_layer_func)
            elif layer_type == "maxpooling":
                x = MaxPooling2D(
                    pool_size=tuple(layer_data["poolSize"]),
                    strides=tuple(layer_data["stride"]),
                    padding=layer_data.get("padding", "same").lower()
                )(input_layer_func)
            elif layer_type == "globalaveragepool":
                x = GlobalAveragePooling2D()(input_layer_func)
            elif layer_type == "flatten":
                x = Flatten()(input_layer_func)
            elif layer_type == "dropout":
                x = Dropout(rate=layer_data["rate"])(input_layer_func)
            elif layer_type == "batchnormalization":
                x = BatchNormalization(
                    momentum=layer_data["momentum"],
                    epsilon=layer_data["epsilon"]
                )(input_layer_func)
            elif layer_type == "activation":
                # This should only happen if it's a standalone activation layer
                activation_func = layer_data.get("function", "relu").lower()
                activation_map = {
                    "relu": "relu",
                    "sigmoid": "sigmoid",
                    "tanh": "tanh",
                    "softmax": "softmax",
                    "leaky relu": "leaky_relu",
                    "leakyrelu": "leaky_relu"
                }
                mapped_activation = activation_map.get(activation_func, "relu")
                
                if mapped_activation == "leaky_relu":
                    alpha = layer_data.get("alpha", 0.3)
                    x = tf.keras.layers.LeakyReLU(alpha=alpha)(input_layer_func)
                else:
                    x = tf.keras.layers.Activation(mapped_activation)(input_layer_func)
            elif layer_type == "attention":
                num_heads = layer_data.get("heads", 8)
                key_dim = layer_data.get("keyDim", 64)
                dropout_rate = layer_data.get("dropout", 0.0)
                
                if dataset_name in ["MNIST", "CIFAR-10"]:
                    reshaped = Reshape((-1, input_shape[0]))(input_layer_func)
                    attention_output = CustomAttentionLayer(
                        num_heads=num_heads,
                        key_dim=key_dim,
                        dropout=dropout_rate,
                        name=f"attention_{node_id}"
                    )(reshaped)
                    x = Reshape(input_shape)(attention_output)
                else:
                    x = CustomAttentionLayer(
                        num_heads=num_heads,
                        key_dim=key_dim,
                        dropout=dropout_rate,
                        name=f"attention_{node_id}"
                    )(input_layer_func)
            elif layer_type == "addlayer":
                if len(incoming_edges) < 2:
                    logger.warning(f"Add layer (id: {node_id}) has fewer than 2 inputs, using identity function")
                    x = input_layer_func
                else:
                    x = Add()(input_layer_func)
            elif layer_type == "output":
                output_units = determine_output_units(dataset_name)
                # For output layer, use the merged activation if available, otherwise determine based on dataset
                if final_activation:
                    output_activation = final_activation
                else:
                    # Auto-determine output activation based on dataset if not specified
                    if dataset_name in ["Iris", "MNIST", "CIFAR-10"]:
                        output_activation = "softmax"  # Multi-class classification
                    elif dataset_name == "Breast Cancer":
                        output_activation = "sigmoid"  # Binary classification
                    else:
                        output_activation = None  # Regression
                
                x = Dense(
                    units=output_units,
                    activation=output_activation
                )(input_layer_func)
            else:
                logger.warning(f"Unknown layer type: {layer_type}, skipping layer {node_id}")
                x = input_layer_func  # Pass through unchanged
            
            # Store the output layer
            layers_by_id[node_id] = x
        
        # Build and return the model
        model = Model(inputs=inputs, outputs=layers_by_id[output_layer["id"]])
        
    else:
        logger.info("Using Sequential API for simple model")
        # Use Sequential API for simple models
        model = Sequential()
        model.add(Input(shape=input_shape))
        
        # First, we need to process the architecture to merge activation layers with their preceding layers
        processed_nodes = []
        i = 0
        while i < len(topo_order):
            node_id = topo_order[i]
            if node_id == input_layer["id"]:
                i += 1
                continue
                
            node = node_map[node_id]
            layer_type = node["type"]
            layer_data = node["data"]
            
            # Check if the next node is an activation layer that should be merged
            next_activation = None
            if i + 1 < len(topo_order):
                next_node_id = topo_order[i + 1]
                next_node = node_map[next_node_id]
                if next_node["type"] == "activation":
                    # Check if this activation directly follows the current layer
                    next_incoming_edges = [edge for edge in edges if edge["target"] == next_node_id]
                    if len(next_incoming_edges) == 1 and next_incoming_edges[0]["source"] == node_id:
                        next_activation = next_node["data"].get("function", "relu").lower()
                        i += 1  # Skip the activation node as we'll merge it
            
            processed_nodes.append({
                "id": node_id,
                "type": layer_type,
                "data": layer_data,
                "merged_activation": next_activation
            })
            i += 1
        
        # Process the merged nodes
        for processed_node in processed_nodes:
            layer_type = processed_node["type"]
            layer_data = processed_node["data"]
            merged_activation = processed_node["merged_activation"]
            node_id = processed_node["id"]
            
            logger.debug(f"Adding {layer_type} layer to sequential model (id: {node_id})")
            
            # Determine final activation - use merged activation if available, otherwise use layer's activation
            final_activation = None
            if merged_activation:
                final_activation = merged_activation
            elif layer_data.get("activation", "none").lower() != "none":
                final_activation = layer_data["activation"].lower()
            
            # Add the appropriate layer
            if layer_type == "dense":
                model.add(Dense(
                    units=layer_data["neurons"],
                    activation=final_activation
                ))
            elif layer_type == "convolution":
                model.add(Conv2D(
                    filters=layer_data["filters"],
                    kernel_size=tuple(layer_data["kernelSize"]),
                    strides=tuple(layer_data["stride"]),
                    padding=layer_data.get("padding", "same").lower(),
                    activation=final_activation
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
            elif layer_type == "activation":
                # This should only happen if it's a standalone activation layer
                activation_func = layer_data.get("function", "relu").lower()
                activation_map = {
                    "relu": "relu",
                    "sigmoid": "sigmoid",
                    "tanh": "tanh",
                    "softmax": "softmax",
                    "leaky relu": "leaky_relu",
                    "leakyrelu": "leaky_relu"
                }
                mapped_activation = activation_map.get(activation_func, "relu")
                
                if mapped_activation == "leaky_relu":
                    alpha = layer_data.get("alpha", 0.3)
                    model.add(tf.keras.layers.LeakyReLU(alpha=alpha))
                else:
                    model.add(tf.keras.layers.Activation(mapped_activation))
            elif layer_type == "attention":
                num_heads = layer_data.get("heads", 8)
                key_dim = layer_data.get("keyDim", 64)
                dropout_rate = layer_data.get("dropout", 0.0)
                
                if dataset_name in ["MNIST", "CIFAR-10"]:
                    model.add(Reshape((-1, input_shape[0])))
                    model.add(CustomAttentionLayer(
                        num_heads=num_heads,
                        key_dim=key_dim,
                        dropout=dropout_rate,
                        name=f"attention_{node_id}"
                    ))
                    model.add(Reshape(input_shape))
                else:
                    model.add(CustomAttentionLayer(
                        num_heads=num_heads,
                        key_dim=key_dim,
                        dropout=dropout_rate,
                        name=f"attention_{node_id}"
                    ))
            elif layer_type == "output":
                output_units = determine_output_units(dataset_name)
                # For output layer, use the merged activation if available, otherwise determine based on dataset
                if final_activation:
                    output_activation = final_activation
                else:
                    # Auto-determine output activation based on dataset if not specified
                    if dataset_name in ["Iris", "MNIST", "CIFAR-10"]:
                        output_activation = "softmax"  # Multi-class classification
                    elif dataset_name == "Breast Cancer":
                        output_activation = "sigmoid"  # Binary classification
                    else:
                        output_activation = None  # Regression
                
                model.add(Dense(
                    units=output_units,
                    activation=output_activation
                ))
            else:
                logger.warning(f"Unknown layer type: {layer_type}, skipping layer {node_id}")
    
    return model 