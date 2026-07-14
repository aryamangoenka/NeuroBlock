def generate_python_script(model, training_config, x_train_shape, dataset_name):
    """
    Generate a Python script that recreates the model architecture and training process.
    
    Args:
        model: The trained Keras model
        training_config: Dictionary with training configuration
        x_train_shape: Shape of the training data
        dataset_name: Name of the dataset being used
        
    Returns:
        str: Generated Python script content
    """
    # Check if the model contains attention layers
    has_attention_layer = False
    for layer in model.layers:
        layer_name = getattr(layer, 'name', '')
        if ('attention' in layer_name.lower() or 
            'multihead' in str(layer.__class__.__name__).lower() or
            'multiheadattention' in str(type(layer)).lower()):
            has_attention_layer = True
            break
    
    script = [
        "import tensorflow as tf",
        "import numpy as np",
        "import json",
        "from tensorflow.keras.models import Sequential",
        "from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Input, Reshape, Activation"
    ]
    
    # Add attention layer imports if needed
    if has_attention_layer:
        script.extend([
            "from tensorflow.keras.layers import MultiHeadAttention, Lambda",
            "from tensorflow.keras import Model",
            "",
            "class CustomAttentionLayer(tf.keras.layers.Layer):",
            "    def __init__(self, num_heads, key_dim, dropout=0.0, **kwargs):",
            "        super(CustomAttentionLayer, self).__init__(**kwargs)",
            "        self.num_heads = num_heads",
            "        self.key_dim = key_dim",
            "        self.dropout = dropout",
            "        self.attention = MultiHeadAttention(",
            "            num_heads=num_heads,",
            "            key_dim=key_dim,",
            "            dropout=dropout",
            "        )",
            "",
            "    def call(self, inputs, training=None):",
            "        return self.attention(inputs, inputs, training=training)",
            "",
            "    def get_config(self):",
            "        config = super(CustomAttentionLayer, self).get_config()",
            "        config.update({",
            "            'num_heads': self.num_heads,",
            "            'key_dim': self.key_dim,",
            "            'dropout': self.dropout",
            "        })",
            "        return config",
            ""
        ])
    
    # Get the actual input shape from the model
    input_shape = model.input_shape[1:] if model.input_shape else x_train_shape[1:]
    input_shape_str = str(input_shape)
    
    # Extract training parameters from the passed config
    epochs = training_config.get("epochs", 10)
    batch_size = training_config.get("batchSize", 32)
    validation_split = training_config.get("validationSplit", 0.2)
    learning_rate = training_config.get("learningRate", 0.001)
    optimizer_name = training_config.get("optimizer", "Adam")
    loss_function = training_config.get("lossFunction", "Categorical Cross-Entropy")
    
    # Start building the script
    script.append("# Load and preprocess the dataset")
    
    # Check if this is a custom dataset
    is_custom_dataset = _is_custom_dataset(dataset_name)
    
    if is_custom_dataset:
        # Handle custom dataset
        custom_dataset_code = _generate_custom_dataset_code(dataset_name, input_shape)
        script.extend(custom_dataset_code)
    else:
        # Handle built-in datasets
        script.extend(_generate_builtin_dataset_code(dataset_name, input_shape))
    
    script.append("")
    script.append("# Define the model")
    script.append("model = Sequential()")
    
    # Add model layers with actual code instead of a placeholder comment
    # First, add the input layer
    script.append(f"model.add(Input(shape={input_shape_str}))")
    
    # Iterate through model layers and generate code for each one (skip input layer if it exists)
    start_idx = 1 if "input" in str(model.layers[0].__class__.__name__).lower() else 0
    
    # Check if we need to validate output layer
    has_output_layer = False
    expected_output_units = None
    
    # Determine expected output units based on dataset
    if is_custom_dataset:
        # For custom datasets, get expected output units from metadata
        metadata = _load_custom_dataset_metadata(dataset_name)
        if metadata:
            task_type = metadata.get('task_type', 'classification')
            class_labels = metadata.get('class_labels', [])
            
            if task_type == 'classification':
                if len(class_labels) > 2:
                    expected_output_units = len(class_labels)
                else:
                    # Binary classification - use 2 units for one-hot encoding (to match training)
                    expected_output_units = 2
            else:
                # Regression
                expected_output_units = 1
    else:
        # Built-in datasets
        if dataset_name == "Iris":
            expected_output_units = 3
        elif dataset_name == "Breast Cancer":
            expected_output_units = 1
        elif dataset_name == "California Housing":
            expected_output_units = 1
        elif dataset_name == "MNIST" or dataset_name == "CIFAR-10":
            expected_output_units = 10
    
    # Process layers to merge standalone activation layers with their preceding layers
    processed_layers = []
    i = start_idx
    while i < len(model.layers):
        layer = model.layers[i]
        layer_class = layer.__class__.__name__
        layer_name = getattr(layer, 'name', '').lower()
        layer_type = getattr(layer, 'type', '').lower()
        
        # Check if the next layer is a standalone activation layer
        merged_activation = None
        if i + 1 < len(model.layers):
            next_layer = model.layers[i + 1]
            next_layer_class = next_layer.__class__.__name__
            next_layer_name = getattr(next_layer, 'name', '').lower()
            next_layer_type = getattr(next_layer, 'type', '').lower()
            
            # Check if next layer is a standalone activation layer
            if ("activation" in next_layer_type or "activation" in next_layer_name or 
                next_layer_class == "Activation"):
                # Get the activation function
                if hasattr(next_layer, 'activation') and hasattr(next_layer.activation, '__name__'):
                    merged_activation = next_layer.activation.__name__
                elif hasattr(next_layer, 'data') and 'function' in next_layer.data:
                    merged_activation = next_layer.data['function'].lower()
                else:
                    merged_activation = 'relu'  # default
                i += 1  # Skip the activation layer as we'll merge it
        
        processed_layers.append({
            'layer': layer,
            'layer_class': layer_class,
            'layer_name': layer_name,
            'layer_type': layer_type,
            'merged_activation': merged_activation
        })
        i += 1
        
    for layer_info in processed_layers:
        layer = layer_info['layer']
        layer_class = layer_info['layer_class']
        layer_name = layer_info['layer_name']
        layer_type = layer_info['layer_type']
        merged_activation = layer_info['merged_activation']
        
        # Handle different layer types
        if "dense" in layer_type or "dense" in layer_name or layer_class == "Dense":
            # Only get the values that were explicitly set
            units = layer.units
            
            # Use merged activation if available, otherwise use layer's activation
            if merged_activation:
                activation = merged_activation
            else:
                activation = layer.activation.__name__ if hasattr(layer.activation, '__name__') else 'None'
            
            # Check if this is potentially an output layer (last layer)
            if layer == model.layers[-1] and expected_output_units is not None:
                if units != expected_output_units:
                    # This is an output layer with incorrect units
                    script.append(f"# Note: Modified output layer to match dataset requirements")
                    script.append(f"model.add(Dense({expected_output_units}, activation='{activation}'))")
                    has_output_layer = True
                else:
                    script.append(f"model.add(Dense({units}, activation='{activation}'))")
                    has_output_layer = True
            else:
                script.append(f"model.add(Dense({units}, activation='{activation}'))")
                
                # If this is the last layer, mark that we have an output layer
                if layer == model.layers[-1]:
                    has_output_layer = True
            
        elif "convolution" in layer_type or "conv2d" in layer_name or layer_class == "Conv2D":
            # Only get the values that were explicitly set
            filters = layer.filters
            
            # Use merged activation if available, otherwise use layer's activation
            if merged_activation:
                activation = merged_activation
            else:
                activation = layer.activation.__name__ if hasattr(layer.activation, '__name__') else 'None'
            
            # Get kernel size from layer data if available
            kernel_size = getattr(layer, 'kernel_size', (3,3))
            if hasattr(layer, 'data') and 'kernelSize' in layer.data:
                kernel_size = tuple(layer.data['kernelSize'])
            
            # Get stride from layer data if available
            strides = getattr(layer, 'strides', (1,1))
            if hasattr(layer, 'data') and 'stride' in layer.data:
                strides = tuple(layer.data['stride'])
            
            # Get padding from layer data if available
            padding = getattr(layer, 'padding', 'valid')
            if hasattr(layer, 'data') and 'padding' in layer.data:
                padding = layer.data['padding']
            
            # Build the layer string with only the parameters that were explicitly set
            layer_str = f"model.add(Conv2D({filters}, kernel_size={kernel_size}"
            if strides != (1,1):
                layer_str += f", strides={strides}"
            if padding != 'valid':
                layer_str += f", padding='{padding}'"
            if activation != 'None':
                layer_str += f", activation='{activation}'"
            layer_str += "))"
            
            script.append(layer_str)
            
        elif "maxpooling" in layer_type or "maxpooling2d" in layer_name or layer_class == "MaxPooling2D":
            # Get pool size from layer data if available
            pool_size = getattr(layer, 'pool_size', (2,2))
            if hasattr(layer, 'data') and 'poolSize' in layer.data:
                pool_size = tuple(layer.data['poolSize'])
            
            # Get stride from layer data if available
            strides = getattr(layer, 'strides', None)
            if hasattr(layer, 'data') and 'stride' in layer.data:
                strides = tuple(layer.data['stride'])
            
            # Get padding from layer data if available
            padding = getattr(layer, 'padding', 'valid')
            if hasattr(layer, 'data') and 'padding' in layer.data:
                padding = layer.data['padding']
            
            # Build the layer string with only the parameters that were explicitly set
            layer_str = f"model.add(MaxPooling2D(pool_size={pool_size}"
            if strides:
                layer_str += f", strides={strides}"
            if padding != 'valid':
                layer_str += f", padding='{padding}'"
            layer_str += "))"
            
            script.append(layer_str)
            
        elif "flatten" in layer_type or "flatten" in layer_name or layer_class == "Flatten":
            script.append(f"model.add(Flatten())")
            
        elif "dropout" in layer_type or "dropout" in layer_name or layer_class == "Dropout":
            # Only include rate if it's not the default 0.5
            rate = layer.rate if hasattr(layer, 'rate') else 0.5
            script.append(f"model.add(Dropout({rate}))")
            
        elif "batchnormalization" in layer_type or "batchnormalization" in layer_name or layer_class == "BatchNormalization":
            # Only get the values that were explicitly set
            momentum = layer.momentum if hasattr(layer, 'momentum') and layer.momentum != 0.99 else None
            epsilon = layer.epsilon if hasattr(layer, 'epsilon') and layer.epsilon != 0.001 else None
            
            # Build the layer string with only the parameters that were explicitly set
            layer_str = "model.add(BatchNormalization("
            if momentum is not None:
                layer_str += f"momentum={momentum}"
            if epsilon is not None:
                layer_str += f", epsilon={epsilon}"
            layer_str += "))"
            
            script.append(layer_str)
            
        elif "reshape" in layer_type or "reshape" in layer_name or layer_class == "Reshape":
            # Only include target_shape if it was explicitly set
            target_shape = layer.target_shape if hasattr(layer, 'target_shape') else None
            if target_shape:
                script.append(f"model.add(Reshape({target_shape}))")
            else:
                script.append(f"model.add(Reshape())")
            
        elif "activation" in layer_type or "activation" in layer_name or layer_class == "Activation":
            # Skip standalone activation layers as they should have been merged with preceding layers
            # This should only happen if it's truly a standalone activation layer
            activation = 'relu'  # default
            if hasattr(layer, 'activation') and hasattr(layer.activation, '__name__'):
                activation = layer.activation.__name__
            elif hasattr(layer, 'data') and 'function' in layer.data:
                activation = layer.data['function'].lower()
            script.append(f"model.add(Activation('{activation}'))")
            
        elif "attention" in layer_type or "attention" in layer_name or "multihead" in layer_name:
            # Only get the values that were explicitly set
            num_heads = layer.num_heads if hasattr(layer, 'num_heads') else None
            key_dim = layer.key_dim if hasattr(layer, 'key_dim') else None
            dropout = layer.dropout if hasattr(layer, 'dropout') and layer.dropout != 0.0 else None
            
            # Build the layer string with only the parameters that were explicitly set
            layer_str = "model.add(CustomAttentionLayer("
            if num_heads is not None:
                layer_str += f"num_heads={num_heads}"
            if key_dim is not None:
                layer_str += f", key_dim={key_dim}"
            if dropout is not None:
                layer_str += f", dropout={dropout}"
            layer_str += "))"
            
            script.append(layer_str)
            
        else:
            # For any other layer types, add a comment
            script.append(f"# Layer type '{layer_class}' is not explicitly handled - adjust if needed")
            script.append(f"# model.add({layer_class}(...))")
    
    # Add appropriate output layer if one is missing and we know what it should be
    if not has_output_layer and expected_output_units is not None:
        if dataset_name == "Breast Cancer" or dataset_name == "California Housing":
            activation = "sigmoid" if dataset_name == "Breast Cancer" else "linear"
            script.append(f"")
            script.append(f"# Adding appropriate output layer for {dataset_name}")
            script.append(f"model.add(Dense({expected_output_units}, activation='{activation}'))")
        else:
            script.append(f"")
            script.append(f"# Adding appropriate output layer for {dataset_name}")
            script.append(f"model.add(Dense({expected_output_units}, activation='softmax'))")
            
    # Add compilation and training
    loss_function = training_config.get("lossFunction", "categorical_crossentropy")
    optimizer = training_config.get("optimizer", "adam")
    
    # Map loss function names
    LOSS_FUNCTION_MAPPING = {
        "Categorical Cross-Entropy": "categorical_crossentropy",
        "Binary Cross-Entropy": "binary_crossentropy",
        "Mean Squared Error": "mse",
        "Mean Absolute Error": "mae",
        "Huber Loss": "huber"
    }
    loss_value = LOSS_FUNCTION_MAPPING.get(loss_function, loss_function.lower())
    
    # Configure optimizer with learning rate
    optimizer_config = f"tf.keras.optimizers.{optimizer.capitalize()}(learning_rate={learning_rate})"
    
    script.append(f"# Compile the model")
    script.append(f"model.compile(")
    script.append(f"    optimizer={optimizer_config},")
    script.append(f"    loss='{loss_value}',")
    script.append(f"    metrics=['accuracy']")
    script.append(f")")
    script.append("")
    script.append("# Display model summary")
    script.append("model.summary()")
    script.append("")
    
    script.append("# Train the model")
    script.append(f"history = model.fit(")
    script.append(f"    x_train, y_train,")
    script.append(f"    epochs={epochs},")
    script.append(f"    batch_size={batch_size},")
    script.append(f"    validation_split={validation_split}")
    script.append(f")")
    script.append("")
    script.append("# Evaluate the model")
    script.append("test_loss, test_acc = model.evaluate(x_test, y_test)")
    script.append("print(f'Test accuracy: {test_acc:.4f}')")
    script.append("")
    script.append("# Save the model")
    script.append("model.save('trained_model.keras')")
    
    return "\n".join(script)


def _is_custom_dataset(dataset_name):
    """
    Check if the dataset is a custom dataset by looking for its metadata file.
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        bool: True if it's a custom dataset, False otherwise
    """
    import os
    from flask import current_app
    
    # List of known built-in datasets
    builtin_datasets = {"MNIST", "CIFAR-10", "Iris", "Breast Cancer", "California Housing"}
    
    # If it's a known built-in dataset, return False
    if dataset_name in builtin_datasets:
        return False
    
    # Check if custom dataset metadata file exists
    try:
        if hasattr(current_app, 'config'):
            project_root = current_app.config.get('PROJECT_ROOT', '')
        else:
            # Fallback to relative path
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        custom_datasets_dir = os.path.join(project_root, 'datasets', 'custom')
        metadata_file = os.path.join(custom_datasets_dir, f'{dataset_name}_metadata.json')
        
        return os.path.exists(metadata_file)
    except Exception:
        return False


def _load_custom_dataset_metadata(dataset_name):
    """
    Load metadata for a custom dataset.
    
    Args:
        dataset_name (str): Name of the custom dataset
        
    Returns:
        dict: Dataset metadata or None if not found
    """
    import os
    import json
    from flask import current_app
    
    try:
        if hasattr(current_app, 'config'):
            project_root = current_app.config.get('PROJECT_ROOT', '')
        else:
            # Fallback to relative path
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        custom_datasets_dir = os.path.join(project_root, 'datasets', 'custom')
        metadata_file = os.path.join(custom_datasets_dir, f'{dataset_name}_metadata.json')
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    
    return None


def _generate_custom_dataset_code(dataset_name, input_shape):
    """
    Generate code to load and preprocess a custom dataset.
    
    Args:
        dataset_name (str): Name of the custom dataset
        input_shape (tuple): Expected input shape for the model
        
    Returns:
        list: List of code lines for loading the custom dataset
    """
    metadata = _load_custom_dataset_metadata(dataset_name)
    
    if not metadata:
        return [
            f"# Custom dataset '{dataset_name}' metadata not found",
            "# Please ensure the dataset files are available and update the paths below",
            "# x_train, y_train = ...",
            "# x_test, y_test = ...",
            f"# Make sure to reshape your data to match the model input shape: {input_shape}"
        ]
    
    task_type = metadata.get('task_type', 'classification')
    dataset_type = metadata.get('dataset_type', 'tabular')
    class_labels = metadata.get('class_labels', [])
    feature_columns = metadata.get('feature_columns', [])
    target_column = metadata.get('target_column', 'target')
    processed_shape = metadata.get('processed_shape', [0, 0])
    
    code = [
        f"# Load custom dataset: {dataset_name}",
        "import numpy as np",
        "from sklearn.model_selection import train_test_split",
        "",
        f"# Load the custom dataset '{dataset_name}'",
        f"# Note: Update the path below to point to your dataset file",
        f"dataset_file = '{dataset_name}.npz'  # Update this path as needed",
        "",
        "# Load data from .npz file",
        "data = np.load(dataset_file)",
        "X = data['X']",
        "y = data['y']",
        "",
        f"# Dataset info:",
        f"# - Task type: {task_type}",
        f"# - Dataset type: {dataset_type}",
    ]
    
    if dataset_type == 'image':
        # Handle image datasets
        target_size = metadata.get('target_size', [224, 224])
        channels = metadata.get('channels', 3)
        
        code.extend([
            f"# - Image size: {target_size[0]}x{target_size[1]}",
            f"# - Channels: {channels}",
            f"# - Classes: {len(class_labels)} ({', '.join(map(str, class_labels))})",
            "",
            "# Image preprocessing (already normalized to [0,1] range)",
            "# Images are already in proper shape (N, H, W, C)",
            "print(f'Image data shape: {X.shape}')",
            "print(f'Image value range: [{X.min():.3f}, {X.max():.3f}]')",
        ])
        
        if task_type == 'classification':
            if len(class_labels) > 2:
                # Multi-class classification - one-hot encode
                code.extend([
                    "",
                    "# One-hot encode labels for multi-class classification",
                    f"y_encoded = tf.keras.utils.to_categorical(y, {len(class_labels)})",
                ])
            else:
                # Binary classification - one-hot encode to match training
                code.extend([
                    "",
                    "# One-hot encode labels for binary classification (to match training)",
                    "y_encoded = tf.keras.utils.to_categorical(y, 2)",
                ])
        
        code.extend([
            "",
            "# Split the data (80% train, 20% test)",
            "x_train, x_test, y_train, y_test = train_test_split(",
            "    X, y_encoded, test_size=0.2, random_state=42",
        ])
        
        if task_type == 'classification':
            code.append("    , stratify=y_encoded")
        
        code.extend([
            ")",
            "",
            "# No additional preprocessing needed for images (already normalized)",
            f"# Data is already in correct shape for model: {input_shape}",
            "",
            "print(f'Dataset loaded: {x_train.shape[0]} training samples, {x_test.shape[0]} test samples')",
            "print(f'Image shape: {x_train.shape[1:]}')",
            "print(f'Target shape: {y_train.shape[1:] if len(y_train.shape) > 1 else \"scalar\"}')"
        ])
        
    else:
        # Handle tabular datasets (existing logic)
        code.extend([
            f"# - Features: {len(feature_columns)} ({', '.join(feature_columns)})",
            f"# - Target: {target_column}",
        ])
        
        if class_labels:
            code.extend([
                f"# - Classes: {len(class_labels)} ({', '.join(map(str, class_labels))})",
            ])
        
        code.extend([
            f"# - Processed shape: {processed_shape}",
            "",
            "# Preprocessing (same as used during training)",
            "from sklearn.preprocessing import StandardScaler, LabelEncoder",
        ])
        
        if task_type == 'classification':
            if len(class_labels) > 2:
                # Multi-class classification - one-hot encode
                code.extend([
                    "# One-hot encode labels for multi-class classification",
                    f"y_encoded = tf.keras.utils.to_categorical(y, {len(class_labels)})",
                ])
            else:
                # Binary classification - one-hot encode to match training
                code.extend([
                    "# One-hot encode labels for binary classification (to match training)",
                    "y_encoded = tf.keras.utils.to_categorical(y, 2)",
                ])
        else:
            # Regression
            code.extend([
                "# For regression, use labels as-is",
                "y_encoded = y.reshape(-1, 1) if len(y.shape) == 1 else y",
            ])
        
        code.extend([
            "",
            "# Split the data (80% train, 20% test)",
            "x_train, x_test, y_train, y_test = train_test_split(",
            "    X, y_encoded, test_size=0.2, random_state=42",
        ])
        
        if task_type == 'classification':
            code.append("    , stratify=y_encoded")
        
        code.extend([
            ")",
            "",
            "# Standardize features (same as used during training)",
            "scaler = StandardScaler()",
            "x_train = scaler.fit_transform(x_train)",
            "x_test = scaler.transform(x_test)",
            "",
            f"# Reshape for model input {input_shape}",
            f"x_train = x_train.reshape(x_train.shape[0], {', '.join(str(dim) for dim in input_shape)})",
            f"x_test = x_test.reshape(x_test.shape[0], {', '.join(str(dim) for dim in input_shape)})",
            "",
            "print(f'Dataset loaded: {x_train.shape[0]} training samples, {x_test.shape[0]} test samples')",
            "print(f'Feature shape: {x_train.shape[1:]}')",
            "print(f'Target shape: {y_train.shape[1:] if len(y_train.shape) > 1 else \"scalar\"}')"
        ])
    
    return code


def _generate_builtin_dataset_code(dataset_name, input_shape):
    """
    Generate code to load and preprocess built-in datasets.
    
    Args:
        dataset_name (str): Name of the built-in dataset
        input_shape (tuple): Expected input shape for the model
        
    Returns:
        list: List of code lines for loading the built-in dataset
    """
    if dataset_name == "MNIST":
        return [
            "# Load MNIST dataset",
            "(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()",
            "# Normalize pixel values to be between 0 and 1",
            "x_train, x_test = x_train / 255.0, x_test / 255.0",
            f"# Reshape for model input {input_shape}",
            f"x_train = x_train.reshape(x_train.shape[0], {', '.join(str(dim) for dim in input_shape)})",
            f"x_test = x_test.reshape(x_test.shape[0], {', '.join(str(dim) for dim in input_shape)})",
            "# Convert class vectors to binary class matrices (one-hot encoding)",
            "y_train = tf.keras.utils.to_categorical(y_train, 10)",
            "y_test = tf.keras.utils.to_categorical(y_test, 10)"
        ]
    elif dataset_name == "CIFAR-10":
        return [
            "# Load CIFAR-10 dataset",
            "(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()",
            "# Normalize pixel values to be between 0 and 1",
            "x_train, x_test = x_train / 255.0, x_test / 255.0",
            f"# Reshape for model input {input_shape}",
            f"x_train = x_train.reshape(x_train.shape[0], {', '.join(str(dim) for dim in input_shape)})",
            f"x_test = x_test.reshape(x_test.shape[0], {', '.join(str(dim) for dim in input_shape)})",
            "# Convert class vectors to binary class matrices (one-hot encoding)",
            "y_train = tf.keras.utils.to_categorical(y_train, 10)",
            "y_test = tf.keras.utils.to_categorical(y_test, 10)"
        ]
    elif dataset_name == "Iris":
        return [
            "# Load Iris dataset",
            "from sklearn.datasets import load_iris",
            "from sklearn.model_selection import train_test_split",
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder",
            "",
            "iris = load_iris()",
            "X = iris.data",
            "y = iris.target.reshape(-1, 1)",
            "",
            "# Scale features",
            "scaler = StandardScaler()",
            "X_scaled = scaler.fit_transform(X)",
            "",
            "# One-hot encode the labels",
            "encoder = OneHotEncoder(sparse_output=False)",
            "y_encoded = encoder.fit_transform(y)",
            "",
            "# Split the data",
            "x_train, x_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)",
            f"# Reshape for model input {input_shape}",
            f"x_train = x_train.reshape(x_train.shape[0], {', '.join(str(dim) for dim in input_shape)})",
            f"x_test = x_test.reshape(x_test.shape[0], {', '.join(str(dim) for dim in input_shape)})"
        ]
    elif dataset_name == "Breast Cancer":
        return [
            "# Load Breast Cancer dataset",
            "from sklearn.datasets import load_breast_cancer",
            "from sklearn.model_selection import train_test_split",
            "from sklearn.preprocessing import StandardScaler",
            "",
            "cancer = load_breast_cancer()",
            "X = cancer.data",
            "y = cancer.target",
            "",
            "# Scale features",
            "scaler = StandardScaler()",
            "X_scaled = scaler.fit_transform(X)",
            "",
            "# Split the data",
            "x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)",
            f"# Reshape for model input {input_shape}",
            f"x_train = x_train.reshape(x_train.shape[0], {', '.join(str(dim) for dim in input_shape)})",
            f"x_test = x_test.reshape(x_test.shape[0], {', '.join(str(dim) for dim in input_shape)})"
        ]
    elif dataset_name == "California Housing":
        return [
            "# Load California Housing dataset",
            "from sklearn.datasets import fetch_california_housing",
            "from sklearn.model_selection import train_test_split",
            "from sklearn.preprocessing import StandardScaler",
            "",
            "housing = fetch_california_housing()",
            "X = housing.data",
            "y = housing.target",
            "",
            "# Scale features",
            "scaler = StandardScaler()",
            "X_scaled = scaler.fit_transform(X)",
            "",
            "# Split the data",
            "x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)",
            f"# Reshape for model input {input_shape}",
            f"x_train = x_train.reshape(x_train.shape[0], {', '.join(str(dim) for dim in input_shape)})",
            f"x_test = x_test.reshape(x_test.shape[0], {', '.join(str(dim) for dim in input_shape)})"
        ]
    else:
        return [
            "# Replace with your own dataset loading code",
            "# x_train, y_train = ...",
            "# x_test, y_test = ...",
            f"# Make sure to reshape your data to match the model input shape: {input_shape}"
        ] 