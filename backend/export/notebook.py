import json

def generate_notebook(model, training_config, x_train_shape):
    """
    Generate a Jupyter notebook that recreates the model architecture and training process.
    
    Args:
        model: The trained Keras model
        training_config: Dictionary with training configuration
        x_train_shape: Shape of the training data
        
    Returns:
        str: Generated Jupyter notebook content as JSON string
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
    
    dataset_name = training_config.get("dataset", "Unknown")
    epochs = training_config.get("epochs", 10)
    batch_size = training_config.get("batchSize", 32)
    validation_split = 0.2
    
    # Map loss function names
    LOSS_FUNCTION_MAPPING = {
        "Categorical Cross-Entropy": "categorical_crossentropy",
        "Binary Cross-Entropy": "binary_crossentropy",
        "Mean Squared Error": "mse",
        "Mean Absolute Error": "mae",
        "Huber Loss": "huber"
    }
    loss_function = training_config.get("lossFunction", "categorical_crossentropy")
    loss_value = LOSS_FUNCTION_MAPPING.get(loss_function, loss_function.lower())
    
    optimizer = training_config.get("optimizer", "adam").lower()
    
    # Dataset-specific preprocessing code
    preprocessing_code = ""
    if dataset_name.lower() == "iris":
        preprocessing_code = """
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load Iris dataset
data = load_iris()
X, y = data.data, data.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""
    elif dataset_name.lower() == "breast cancer":
        preprocessing_code = """
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Breast Cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
"""
    elif dataset_name.lower() == "california housing":
        preprocessing_code = """
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load California Housing dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""
    elif dataset_name.lower() == "mnist":
        preprocessing_code = """
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
"""
    elif dataset_name.lower() == "cifar-10":
        preprocessing_code = """
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
"""
    
    # Custom attention layer definition
    custom_attention_code = """
# Define custom attention function for Lambda layer
@tf.keras.utils.register_keras_serializable(package='custom_layers')
def apply_attention(x, num_heads=8, key_dim=64, dropout=0.0):
    attention_layer = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout
    )
    return attention_layer(x, x)

# Define custom attention layer class
@tf.keras.utils.register_keras_serializable(package='custom_layers')
class CustomAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads=8, key_dim=64, dropout=0.0, **kwargs):
        super(CustomAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout = dropout
        self.attention = None  # Will be initialized in build()
        
    def build(self, input_shape):
        self.attention = MultiHeadAttention(
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
"""
    
    # Base imports without attention
    imports_code = """
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Input, Reshape
import numpy as np
import matplotlib.pyplot as plt
import wandb
"""

    # Add attention imports if needed
    if has_attention_layer:
        imports_code = """
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Input, MultiHeadAttention, Lambda, Reshape
import numpy as np
import matplotlib.pyplot as plt
import wandb

# Enable unsafe deserialization for Lambda layers
tf.keras.config.enable_unsafe_deserialization()
"""

    # Add WandB initialization
    wandb_init_code = f"""
# Initialize Weights & Biases for experiment tracking
wandb.init(project="dnd-neural-network", 
           name="{dataset_name}-model",
           config={{
               "dataset": "{dataset_name}",
               "optimizer": "{optimizer}",
               "loss_function": "{loss_value}",
               "batch_size": {batch_size},
               "epochs": {epochs},
               "validation_split": {validation_split}
           }})
"""

    # Generate model definition using the actual model architecture
    model_definition_code = f"""
# Define the model
model = Sequential()
model.add(Input(shape={x_train_shape[1:]}))

"""
    
    # Iterate through model layers and generate code for each one (skip input layer if it exists)
    start_idx = 1 if "input" in str(model.layers[0].__class__.__name__).lower() else 0
    
    # Determine expected output units based on dataset
    expected_output_units = None
    if dataset_name.lower() == "iris":
        expected_output_units = 3
    elif dataset_name.lower() == "breast cancer":
        expected_output_units = 1
    elif dataset_name.lower() == "california housing":
        expected_output_units = 1
    elif dataset_name.lower() == "mnist" or dataset_name.lower() == "cifar-10":
        expected_output_units = 10
    
    # Flag to track if we've processed the output layer
    has_output_layer = False
    
    for layer in model.layers[start_idx:]:
        layer_class = layer.__class__.__name__
        
        # Handle different layer types
        if layer_class == "Dense":
            activation = layer.activation.__name__ if hasattr(layer.activation, '__name__') else 'None'
            
            # Check if this is potentially an output layer (last layer)
            if layer == model.layers[-1] and expected_output_units is not None:
                if layer.units != expected_output_units:
                    # This is an output layer with incorrect units
                    model_definition_code += f"# Note: Modified output layer to match dataset requirements\n"
                    model_definition_code += f"model.add(Dense({expected_output_units}, activation='{activation}'))\n"
                    has_output_layer = True
                else:
                    model_definition_code += f"model.add(Dense({layer.units}, activation='{activation}'))\n"
                    has_output_layer = True
            else:
                model_definition_code += f"model.add(Dense({layer.units}, activation='{activation}'))\n"
                
                # If this is the last layer, mark that we have an output layer
                if layer == model.layers[-1]:
                    has_output_layer = True
            
        elif layer_class == "Conv2D":
            kernel_size = tuple(layer.kernel_size) if hasattr(layer, 'kernel_size') else (3, 3)
            strides = tuple(layer.strides) if hasattr(layer, 'strides') else (1, 1)
            padding = "'" + layer.padding + "'" if hasattr(layer, 'padding') else "'valid'"
            activation = layer.activation.__name__ if hasattr(layer.activation, '__name__') else 'None'
            model_definition_code += f"model.add(Conv2D({layer.filters}, kernel_size={kernel_size}, strides={strides}, padding={padding}, activation='{activation}'))\n"
            
        elif layer_class == "MaxPooling2D":
            pool_size = tuple(layer.pool_size) if hasattr(layer, 'pool_size') else (2, 2)
            strides = tuple(layer.strides) if hasattr(layer, 'strides') else None
            padding = "'" + layer.padding + "'" if hasattr(layer, 'padding') else "'valid'"
            
            if strides is None:
                model_definition_code += f"model.add(MaxPooling2D(pool_size={pool_size}, padding={padding}))\n"
            else:
                model_definition_code += f"model.add(MaxPooling2D(pool_size={pool_size}, strides={strides}, padding={padding}))\n"
            
        elif layer_class == "Flatten":
            model_definition_code += f"model.add(Flatten())\n"
            
        elif layer_class == "Dropout":
            model_definition_code += f"model.add(Dropout({layer.rate}))\n"
            
        elif layer_class == "BatchNormalization":
            momentum = layer.momentum if hasattr(layer, 'momentum') else 0.99
            epsilon = layer.epsilon if hasattr(layer, 'epsilon') else 0.001
            model_definition_code += f"model.add(BatchNormalization(momentum={momentum}, epsilon={epsilon}))\n"
            
        elif layer_class == "Reshape":
            target_shape = layer.target_shape if hasattr(layer, 'target_shape') else (None,)
            model_definition_code += f"model.add(Reshape({target_shape}))\n"
            
        elif "attention" in layer_class.lower() or "multihead" in layer_class.lower():
            # Add attention layer with custom parameters if available
            num_heads = layer.num_heads if hasattr(layer, 'num_heads') else 8
            key_dim = layer.key_dim if hasattr(layer, 'key_dim') else 64
            dropout = layer.dropout if hasattr(layer, 'dropout') else 0.0
            model_definition_code += f"model.add(CustomAttentionLayer(num_heads={num_heads}, key_dim={key_dim}, dropout={dropout}))\n"
            
        else:
            # For any other layer types, add a comment
            model_definition_code += f"# Layer type '{layer_class}' is not explicitly handled - adjust if needed\n"
            model_definition_code += f"# model.add({layer_class}(...))\n"
    
    # Add appropriate output layer if one is missing and we know what it should be
    if not has_output_layer and expected_output_units is not None:
        if dataset_name.lower() == "breast cancer" or dataset_name.lower() == "california housing":
            activation = "sigmoid" if dataset_name.lower() == "breast cancer" else "linear"
            model_definition_code += f"\n# Adding appropriate output layer for {dataset_name}\n"
            model_definition_code += f"model.add(Dense({expected_output_units}, activation='{activation}'))\n"
        else:
            model_definition_code += f"\n# Adding appropriate output layer for {dataset_name}\n"
            model_definition_code += f"model.add(Dense({expected_output_units}, activation='softmax'))\n"
    
    # Add model compilation and W&B model watching
    model_definition_code += f"""
# Compile the model
model.compile(optimizer='{optimizer}', loss='{loss_value}', metrics=['accuracy'])
model.summary()

# Log model architecture with Weights & Biases
wandb.watch(model, log='all')
"""

    training_code = f"""
# Define Weights & Biases callback for logging
wandb_callback = wandb.keras.WandbCallback()

# Train the model
history = model.fit(
    x_train, y_train, 
    epochs={epochs}, 
    batch_size={batch_size}, 
    validation_split={validation_split},
    callbacks=[wandb_callback]
)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {{loss}}")
print(f"Test Accuracy: {{accuracy}}")

# Save the model
model.save('trained_model.keras')
wandb.save('trained_model.keras')  # Also save to W&B
"""

    visualization_code = """
# Visualize training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()

# Log visualizations to W&B
wandb.log({"training_curves": wandb.Image(plt)})
plt.show()

# Log additional metrics for classification tasks
if any(dataset in "{dataset_name}".lower() for dataset in ["iris", "mnist", "cifar-10", "breast cancer"]):
    # Get predictions
    predictions = model.predict(x_test)
    
    # Process predictions based on dataset type
    if "breast cancer" in "{dataset_name}".lower():
        y_pred = (predictions > 0.5).astype(int)
        y_true = y_test
    else:
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Log to W&B
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.show()

# Finish the W&B run
wandb.finish()
"""

    # Create the notebook in JSON format
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Neural Network Model\n", f"Dataset: {dataset_name}\n"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [imports_code]
            }
        ]
    }

    # Add W&B initialization
    notebook["cells"].extend([
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Initialize Weights & Biases"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [wandb_init_code]
        }
    ])
    
    # Only add the attention layer cell if needed
    if has_attention_layer:
        notebook["cells"].extend([
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Custom Attention Layer Definition"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [custom_attention_code]
            }
        ])
    
    # Add the rest of the cells
    notebook["cells"].extend([
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Data Preprocessing"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [preprocessing_code]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Model Definition"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [model_definition_code]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Model Training"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [training_code]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Visualize Training Results"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [visualization_code]
        }
    ])
    
    # Add metadata
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    }
    notebook["nbformat"] = 4
    notebook["nbformat_minor"] = 5
    
    return json.dumps(notebook, indent=2) 