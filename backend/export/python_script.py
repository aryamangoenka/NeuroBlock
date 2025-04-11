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
        "from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Input, Reshape"
    ]
    
    # Only add MultiHeadAttention and Lambda imports if needed
    if has_attention_layer:
        script[-1] = "from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Input, MultiHeadAttention, Lambda, Reshape"
        script.extend([
            "",
            "# Enable unsafe deserialization for Lambda layers",
            "tf.keras.config.enable_unsafe_deserialization()",
            "",
            "# Define custom attention function for Lambda layer",
            "@tf.keras.utils.register_keras_serializable(package='custom_layers')",
            "def apply_attention(x, num_heads=8, key_dim=64, dropout=0.0):",
            "    attention_layer = MultiHeadAttention(",
            "        num_heads=num_heads,",
            "        key_dim=key_dim,",
            "        dropout=dropout",
            "    )",
            "    return attention_layer(x, x)",
            "",
            "# Define custom attention layer class",
            "@tf.keras.utils.register_keras_serializable(package='custom_layers')",
            "class CustomAttentionLayer(tf.keras.layers.Layer):",
            "    def __init__(self, num_heads=8, key_dim=64, dropout=0.0, **kwargs):",
            "        super(CustomAttentionLayer, self).__init__(**kwargs)",
            "        self.num_heads = num_heads",
            "        self.key_dim = key_dim",
            "        self.dropout = dropout",
            "        self.attention = None  # Will be initialized in build()",
            "        ",
            "    def build(self, input_shape):",
            "        self.attention = MultiHeadAttention(",
            "            num_heads=self.num_heads,",
            "            key_dim=self.key_dim,",
            "            dropout=self.dropout",
            "        )",
            "        super(CustomAttentionLayer, self).build(input_shape)",
            "        ",
            "    def call(self, inputs, training=None):",
            "        return self.attention(inputs, inputs, training=training)",
            "        ",
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
    
    # Determine input shape based on dataset
    input_shape_str = str(x_train_shape[1:]) if len(x_train_shape) > 1 else "(None,)"
    
    # Extract model parameters
    epochs = training_config.get("epochs", 10)
    batch_size = training_config.get("batchSize", 32)
    validation_split = 0.2
    
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
            "# Reshape for CNN input (28, 28, 1)",
            "x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)",
            "x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)",
            "# Convert class vectors to binary class matrices (one-hot encoding)",
            "y_train = tf.keras.utils.to_categorical(y_train, 10)",
            "y_test = tf.keras.utils.to_categorical(y_test, 10)"
        ])
        input_shape_str = "(28, 28, 1)"
    elif dataset_name == "CIFAR-10":
        script.extend([
            "# Load CIFAR-10 dataset",
            "(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()",
            "# Normalize pixel values to be between 0 and 1",
            "x_train, x_test = x_train / 255.0, x_test / 255.0",
            "# Convert class vectors to binary class matrices (one-hot encoding)",
            "y_train = tf.keras.utils.to_categorical(y_train, 10)",
            "y_test = tf.keras.utils.to_categorical(y_test, 10)"
        ])
        input_shape_str = "(32, 32, 3)"
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
            "encoder = OneHotEncoder(sparse=False)",
            "y_encoded = encoder.fit_transform(y)",
            "",
            "# Split the data",
            "x_train, x_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)"
        ])
        input_shape_str = "(4,)"
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
            "x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)"
        ])
        input_shape_str = "(30,)"
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
            "x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)"
        ])
        input_shape_str = "(8,)"
    else:
        script.extend([
            "# Replace with your own dataset loading code",
            "# x_train, y_train = ...",
            "# x_test, y_test = ..."
        ])
    
    script.append("")
    script.append("# Define the model")
    script.append("model = Sequential()")
    
    # Add model layers with actual code instead of a placeholder comment
    # First, add the input layer
    script.append(f"model.add(Input(shape={input_shape_str}))")
    
    # Iterate through model layers and generate code for each one (skip input layer if it exists)
    start_idx = 1 if "input" in str(model.layers[0].__class__.__name__).lower() else 0
    
    for layer in model.layers[start_idx:]:
        layer_class = layer.__class__.__name__
        
        # Handle different layer types
        if layer_class == "Dense":
            activation = layer.activation.__name__ if hasattr(layer.activation, '__name__') else 'None'
            script.append(f"model.add(Dense({layer.units}, activation='{activation}'))")
            
        elif layer_class == "Conv2D":
            kernel_size = tuple(layer.kernel_size) if hasattr(layer, 'kernel_size') else (3, 3)
            strides = tuple(layer.strides) if hasattr(layer, 'strides') else (1, 1)
            padding = "'" + layer.padding + "'" if hasattr(layer, 'padding') else "'valid'"
            activation = layer.activation.__name__ if hasattr(layer.activation, '__name__') else 'None'
            script.append(f"model.add(Conv2D({layer.filters}, kernel_size={kernel_size}, strides={strides}, padding={padding}, activation='{activation}'))")
            
        elif layer_class == "MaxPooling2D":
            pool_size = tuple(layer.pool_size) if hasattr(layer, 'pool_size') else (2, 2)
            strides = tuple(layer.strides) if hasattr(layer, 'strides') else None
            padding = "'" + layer.padding + "'" if hasattr(layer, 'padding') else "'valid'"
            
            if strides is None:
                script.append(f"model.add(MaxPooling2D(pool_size={pool_size}, padding={padding}))")
            else:
                script.append(f"model.add(MaxPooling2D(pool_size={pool_size}, strides={strides}, padding={padding}))")
            
        elif layer_class == "Flatten":
            script.append(f"model.add(Flatten())")
            
        elif layer_class == "Dropout":
            script.append(f"model.add(Dropout({layer.rate}))")
            
        elif layer_class == "BatchNormalization":
            momentum = layer.momentum if hasattr(layer, 'momentum') else 0.99
            epsilon = layer.epsilon if hasattr(layer, 'epsilon') else 0.001
            script.append(f"model.add(BatchNormalization(momentum={momentum}, epsilon={epsilon}))")
            
        elif layer_class == "Reshape":
            target_shape = layer.target_shape if hasattr(layer, 'target_shape') else (None,)
            script.append(f"model.add(Reshape({target_shape}))")
            
        elif "attention" in layer_class.lower() or "multihead" in layer_class.lower():
            # Add attention layer with custom parameters if available
            num_heads = layer.num_heads if hasattr(layer, 'num_heads') else 8
            key_dim = layer.key_dim if hasattr(layer, 'key_dim') else 64
            dropout = layer.dropout if hasattr(layer, 'dropout') else 0.0
            script.append(f"model.add(CustomAttentionLayer(num_heads={num_heads}, key_dim={key_dim}, dropout={dropout}))")
            
        else:
            # For any other layer types, add a comment
            script.append(f"# Layer type '{layer_class}' is not explicitly handled - adjust if needed")
            script.append(f"# model.add({layer_class}(...))")
    
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
    
    script.append(f"# Compile the model")
    script.append(f"model.compile(")
    script.append(f"    optimizer='{optimizer}',")
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
    script.append("")
    script.append("# Visualize training history")
    script.append("import matplotlib.pyplot as plt")
    script.append("")
    script.append("plt.figure(figsize=(12, 4))")
    script.append("plt.subplot(1, 2, 1)")
    script.append("plt.plot(history.history['loss'], label='Training Loss')")
    script.append("plt.plot(history.history['val_loss'], label='Validation Loss')")
    script.append("plt.title('Loss over Epochs')")
    script.append("plt.xlabel('Epoch')")
    script.append("plt.ylabel('Loss')")
    script.append("plt.legend()")
    script.append("")
    script.append("plt.subplot(1, 2, 2)")
    script.append("plt.plot(history.history['accuracy'], label='Training Accuracy')")
    script.append("plt.plot(history.history['val_accuracy'], label='Validation Accuracy')")
    script.append("plt.title('Accuracy over Epochs')")
    script.append("plt.xlabel('Epoch')")
    script.append("plt.ylabel('Accuracy')")
    script.append("plt.legend()")
    script.append("plt.tight_layout()")
    script.append("plt.show()")
    
    return "\n".join(script) 