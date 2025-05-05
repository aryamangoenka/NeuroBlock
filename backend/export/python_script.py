def generate_python_script(model, training_config, x_train_shape):
    """
    Generate a Python script that recreates the model architecture and training process.
    
    Args:
        model: The trained Keras model
        training_config: Dictionary with training configuration
        x_train_shape: Shape of the training data
        
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
    
    # Get dataset name from training config
    dataset_name = training_config.get("dataset", "Unknown")
    
    # Get the actual input shape from the model
    input_shape = model.input_shape[1:] if model.input_shape else x_train_shape[1:]
    input_shape_str = str(input_shape)
    
    # Extract model parameters from training config
    epochs = training_config.get("epochs", 10)
    batch_size = training_config.get("batchSize", 32)
    validation_split = training_config.get("validationSplit", 0.2)
    learning_rate = training_config.get("learningRate", 0.001)
    
    # Start building the script
    script.append("")
    script.append("# Load and preprocess the dataset")
    
    # Add dataset-specific code
    if dataset_name == "MNIST":
        script.extend([
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
        ])
    elif dataset_name == "CIFAR-10":
        script.extend([
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
        ])
    elif dataset_name == "Iris":
        script.extend([
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
        ])
    elif dataset_name == "Breast Cancer":
        script.extend([
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
        ])
    elif dataset_name == "California Housing":
        script.extend([
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
        ])
    else:
        script.extend([
            "# Replace with your own dataset loading code",
            "# x_train, y_train = ...",
            "# x_test, y_test = ...",
            f"# Make sure to reshape your data to match the model input shape: {input_shape}"
        ])
    
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
    if dataset_name == "Iris":
        expected_output_units = 3
    elif dataset_name == "Breast Cancer":
        expected_output_units = 1
    elif dataset_name == "California Housing":
        expected_output_units = 1
    elif dataset_name == "MNIST" or dataset_name == "CIFAR-10":
        expected_output_units = 10
        
    for layer in model.layers[start_idx:]:
        layer_class = layer.__class__.__name__
        layer_name = getattr(layer, 'name', '').lower()
        layer_type = getattr(layer, 'type', '').lower()
        
        # Handle different layer types
        if "dense" in layer_type or "dense" in layer_name or layer_class == "Dense":
            # Only get the values that were explicitly set
            units = layer.units
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
            layer_str = f"model.add(Conv2D({filters}"
            if kernel_size != (3,3):
                layer_str += f", kernel_size={kernel_size}"
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
            layer_str = "model.add(MaxPooling2D("
            if pool_size != (2,2):
                layer_str += f"pool_size={pool_size}"
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
            rate = layer.rate if hasattr(layer, 'rate') and layer.rate != 0.5 else None
            if rate is not None:
                script.append(f"model.add(Dropout({rate}))")
            else:
                script.append(f"model.add(Dropout())")
            
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
            
        elif "activation" in layer_type or "activation" in layer_name:
            # Get activation function from layer data if available
            activation = 'relu'  # default
            if hasattr(layer, 'data') and 'function' in layer.data:
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
    script.append("model.save('trained_model.h5')")
    
    return "\n".join(script) 