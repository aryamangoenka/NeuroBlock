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
        
        # Process nodes in topological order
        for node_id in topo_order:
            if node_id == input_layer["id"]:
                continue
                
            node = node_map[node_id]
            incoming_edges = [edge for edge in edges if edge["target"] == node_id]
            
            # Get input layers
            if len(incoming_edges) == 1:
                input_layer = layers_by_id[incoming_edges[0]["source"]]
            else:
                input_layer = [layers_by_id[edge["source"]] for edge in incoming_edges]
            
            # Process the node based on its type
            layer_type = node["type"]
            layer_data = node["data"]
            
            logger.debug(f"Processing {layer_type} layer (id: {node_id})")
            
            # Create the appropriate layer
            if layer_type == "resnetblock":
                x = create_resnet_block(
                    input_layer,
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
                x = MaxPooling2D(
                    pool_size=tuple(layer_data["poolSize"]),
                    strides=tuple(layer_data["stride"]),
                    padding=layer_data.get("padding", "same").lower()
                )(input_layer)
            elif layer_type == "globalaveragepool":
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
            elif layer_type == "activation":
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
                    x = tf.keras.layers.LeakyReLU(alpha=alpha)(input_layer)
                else:
                    x = tf.keras.layers.Activation(mapped_activation)(input_layer)
            elif layer_type == "attention":
                num_heads = layer_data.get("heads", 8)
                key_dim = layer_data.get("keyDim", 64)
                dropout_rate = layer_data.get("dropout", 0.0)
                
                if dataset_name in ["MNIST", "CIFAR-10"]:
                    reshaped = Reshape((-1, input_shape[0]))(input_layer)
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
                    )(input_layer)
            elif layer_type == "addlayer":
                if len(incoming_edges) < 2:
                    logger.warning(f"Add layer (id: {node_id}) has fewer than 2 inputs, using identity function")
                    x = input_layer
                else:
                    x = Add()(input_layer)
            elif layer_type == "output":
                output_units = determine_output_units(dataset_name)
                activation = layer_data["activation"].lower()
                if activation == "none":
                    activation = None
                x = Dense(
                    units=output_units,
                    activation=activation
                )(input_layer)
            
            # Store the output layer
            layers_by_id[node_id] = x
        
        # Build and return the model
        model = Model(inputs=inputs, outputs=layers_by_id[output_layer["id"]])
        
    else:
        logger.info("Using Sequential API for simple model")
        # Use Sequential API for simple models
        model = Sequential()
        model.add(Input(shape=input_shape))
        
        # Process nodes in topological order
        for node_id in topo_order:
            if node_id == input_layer["id"]:
                continue
                
            node = node_map[node_id]
            layer_type = node["type"]
            layer_data = node["data"]
            
            logger.debug(f"Adding {layer_type} layer to sequential model (id: {node_id})")
            
            # Add the appropriate layer
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
                    padding=layer_data.get("padding", "same").lower(),
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
            elif layer_type == "activation":
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
                activation = layer_data["activation"].lower()
                if activation == "none":
                    activation = None
                model.add(Dense(
                    units=output_units,
                    activation=activation
                ))
    
    return model 