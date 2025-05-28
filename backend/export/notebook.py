import json

def generate_notebook(model, training_config, x_train_shape, dataset_name):
    """
    Generate a Jupyter notebook that recreates the model architecture and training process.
    Args:
        model: The trained Keras model
        training_config: Dictionary with training configuration
        x_train_shape: Shape of the training data
        dataset_name: Name of the dataset being used
    Returns:
        str: Generated notebook content as JSON string
    """
    # Import helper functions from python_script module
    from .python_script import _is_custom_dataset, _generate_custom_dataset_code, _generate_builtin_dataset_code, _load_custom_dataset_metadata
    
    # Check if the model contains attention layers
    has_attention_layer = False
    for layer in model.layers:
        layer_name = getattr(layer, 'name', '')
        if ('attention' in layer_name.lower() or 
            'multihead' in str(layer.__class__.__name__).lower() or
            'multiheadattention' in str(type(layer)).lower()):
            has_attention_layer = True
            break

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
    optimizer_config = f"tf.keras.optimizers.{optimizer_name.capitalize()}(learning_rate={learning_rate})"

    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# Model Architecture and Training\n",
            "This notebook recreates the model architecture and training process from the DND-Neural-Network application."
        ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [
            "import tensorflow as tf",
            "import numpy as np",
            "import json",
            "from tensorflow.keras.models import Sequential",
            "from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Input, Reshape, Activation"
        ], "outputs": []}
    ]

    if has_attention_layer:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
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
                "        return config"
            ],
            "outputs": []
        })

    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Load and Preprocess Dataset"]})

    # Check if this is a custom dataset and generate appropriate code
    is_custom_dataset = _is_custom_dataset(dataset_name)
    
    if is_custom_dataset:
        # Handle custom dataset
        dataset_code = _generate_custom_dataset_code(dataset_name, input_shape)
    else:
        # Handle built-in datasets
        dataset_code = _generate_builtin_dataset_code(dataset_name, input_shape)

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": dataset_code,
        "outputs": []
    })

    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Define Model Architecture"]})

    model_code = [
        "model = Sequential()",
        f"model.add(Input(shape={input_shape_str}))"
    ]

    start_idx = 1 if "input" in str(model.layers[0].__class__.__name__).lower() else 0
    has_output_layer = False
    expected_output_units = None
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
        if "dense" in layer_type or "dense" in layer_name or layer_class == "Dense":
            units = layer.units
            activation = layer.activation.__name__ if hasattr(layer.activation, '__name__') else 'None'
            if layer == model.layers[-1] and expected_output_units is not None:
                if units != expected_output_units:
                    model_code.append(f"# Note: Modified output layer to match dataset requirements")
                    model_code.append(f"model.add(Dense({expected_output_units}, activation='{activation}'))")
                    has_output_layer = True
                else:
                    model_code.append(f"model.add(Dense({units}, activation='{activation}'))")
                    has_output_layer = True
            else:
                model_code.append(f"model.add(Dense({units}, activation='{activation}'))")
                if layer == model.layers[-1]:
                    has_output_layer = True
        elif "convolution" in layer_type or "conv2d" in layer_name or layer_class == "Conv2D":
            filters = layer.filters
            activation = layer.activation.__name__ if hasattr(layer.activation, '__name__') else 'None'
            kernel_size = getattr(layer, 'kernel_size', (3,3))
            if hasattr(layer, 'data') and 'kernelSize' in layer.data:
                kernel_size = tuple(layer.data['kernelSize'])
            strides = getattr(layer, 'strides', (1,1))
            if hasattr(layer, 'data') and 'stride' in layer.data:
                strides = tuple(layer.data['stride'])
            padding = getattr(layer, 'padding', 'valid')
            if hasattr(layer, 'data') and 'padding' in layer.data:
                padding = layer.data['padding']
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
            model_code.append(layer_str)
        elif "maxpooling" in layer_type or "maxpooling2d" in layer_name or layer_class == "MaxPooling2D":
            pool_size = getattr(layer, 'pool_size', (2,2))
            if hasattr(layer, 'data') and 'poolSize' in layer.data:
                pool_size = tuple(layer.data['poolSize'])
            strides = getattr(layer, 'strides', None)
            if hasattr(layer, 'data') and 'stride' in layer.data:
                strides = tuple(layer.data['stride'])
            padding = getattr(layer, 'padding', 'valid')
            if hasattr(layer, 'data') and 'padding' in layer.data:
                padding = layer.data['padding']
            layer_str = "model.add(MaxPooling2D("
            if pool_size != (2,2):
                layer_str += f"pool_size={pool_size}"
            if strides:
                layer_str += f", strides={strides}"
            if padding != 'valid':
                layer_str += f", padding='{padding}'"
            layer_str += "))"
            model_code.append(layer_str)
        elif "flatten" in layer_type or "flatten" in layer_name or layer_class == "Flatten":
            model_code.append("model.add(Flatten())")
        elif "dropout" in layer_type or "dropout" in layer_name or layer_class == "Dropout":
            rate = layer.rate if hasattr(layer, 'rate') and layer.rate != 0.5 else None
            if rate is not None:
                model_code.append(f"model.add(Dropout({rate}))")
            else:
                model_code.append("model.add(Dropout())")
        elif "batchnormalization" in layer_type or "batchnormalization" in layer_name or layer_class == "BatchNormalization":
            momentum = layer.momentum if hasattr(layer, 'momentum') and layer.momentum != 0.99 else None
            epsilon = layer.epsilon if hasattr(layer, 'epsilon') and layer.epsilon != 0.001 else None
            layer_str = "model.add(BatchNormalization("
            if momentum is not None:
                layer_str += f"momentum={momentum}"
            if epsilon is not None:
                layer_str += f", epsilon={epsilon}"
            layer_str += "))"
            model_code.append(layer_str)
        elif "reshape" in layer_type or "reshape" in layer_name or layer_class == "Reshape":
            target_shape = layer.target_shape if hasattr(layer, 'target_shape') else None
            if target_shape:
                model_code.append(f"model.add(Reshape({target_shape}))")
            else:
                model_code.append("model.add(Reshape())")
        elif "activation" in layer_type or "activation" in layer_name:
            activation = 'relu'  # default
            if hasattr(layer, 'data') and 'function' in layer.data:
                activation = layer.data['function'].lower()
            model_code.append(f"model.add(Activation('{activation}'))")
        elif "attention" in layer_type or "attention" in layer_name or "multihead" in layer_name:
            num_heads = layer.num_heads if hasattr(layer, 'num_heads') else None
            key_dim = layer.key_dim if hasattr(layer, 'key_dim') else None
            dropout = layer.dropout if hasattr(layer, 'dropout') and layer.dropout != 0.0 else None
            layer_str = "model.add(CustomAttentionLayer("
            if num_heads is not None:
                layer_str += f"num_heads={num_heads}"
            if key_dim is not None:
                layer_str += f", key_dim={key_dim}"
            if dropout is not None:
                layer_str += f", dropout={dropout}"
            layer_str += "))"
            model_code.append(layer_str)
        else:
            model_code.append(f"# Layer type '{layer_class}' is not explicitly handled - adjust if needed")
            model_code.append(f"# model.add({layer_class}(...))")
    if not has_output_layer and expected_output_units is not None:
        if dataset_name == "Breast Cancer" or dataset_name == "California Housing":
            activation = "sigmoid" if dataset_name == "Breast Cancer" else "linear"
            model_code.append("")
            model_code.append(f"# Adding appropriate output layer for {dataset_name}")
            model_code.append(f"model.add(Dense({expected_output_units}, activation='{activation}'))")
        else:
            model_code.append("")
            model_code.append(f"# Adding appropriate output layer for {dataset_name}")
            model_code.append(f"model.add(Dense({expected_output_units}, activation='softmax'))")
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": model_code,
        "outputs": []
    })

    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Compile and Train Model"]})
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Compile the model",
            "model.compile(",
            f"    optimizer={optimizer_config},",
            f"    loss='{loss_value}',",
            "    metrics=['accuracy']",
            ")"
        ],
        "outputs": []
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Display model summary",
            "model.summary()"
        ],
        "outputs": []
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Train the model",
            f"history = model.fit(",
            f"    x_train, y_train,",
            f"    epochs={epochs},",
            f"    batch_size={batch_size},",
            f"    validation_split={validation_split}",
            ")"
        ],
        "outputs": []
    })
    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Evaluate Model"]})
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Evaluate the model",
            "test_loss, test_acc = model.evaluate(x_test, y_test)",
            "print(f'Test accuracy: {test_acc:.4f}')"
        ],
        "outputs": []
    })
    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Save Model"]})
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Save the model",
            "model.save('trained_model.h5')"
        ],
        "outputs": []
    })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    return json.dumps(notebook, indent=2) 