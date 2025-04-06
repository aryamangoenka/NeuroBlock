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
"""

    # Add attention imports if needed
    if has_attention_layer:
        imports_code = """
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Input, MultiHeadAttention, Lambda, Reshape
import numpy as np
import matplotlib.pyplot as plt

# Enable unsafe deserialization for Lambda layers
tf.keras.config.enable_unsafe_deserialization()
"""

    # Generate placeholder model structure
    model_definition_code = f"""
# Define the model
model = Sequential()
model.add(Input(shape={x_train_shape[1:]}))

# ... Additional layers would be added here based on your model ...

# Placeholder for final layer
model.add(Dense(units=10, activation='softmax'))  # Replace with actual output size and activation

model.compile(optimizer='{optimizer}', loss='{loss_value}', metrics=['accuracy'])
model.summary()
"""

    training_code = f"""
# Train the model
history = model.fit(x_train, y_train, epochs={epochs}, batch_size={batch_size}, validation_split={validation_split})

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {{loss}}")
print(f"Test Accuracy: {{accuracy}}")

# Save the model
model.save('trained_model.keras')
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
plt.show()
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