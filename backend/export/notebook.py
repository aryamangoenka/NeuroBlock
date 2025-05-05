import json

def generate_notebook(model, training_config, x_train_shape):
    """
    Generate a Jupyter notebook that recreates the model architecture and training process.
    
    Args:
        model: The trained Keras model
        training_config: Dictionary with training configuration
        x_train_shape: Shape of the training data
        
    Returns:
        str: Generated notebook content as JSON string
    """
    # Get the actual input shape from the model
    input_shape = model.input_shape[1:] if model.input_shape else x_train_shape[1:]
    input_shape_str = str(input_shape)
    
    # Get dataset name from training config
    dataset_name = training_config.get("dataset", "Unknown")
    
    # Extract model parameters
    epochs = training_config.get("epochs", 10)
    batch_size = training_config.get("batchSize", 32)
    validation_split = 0.2
    
    # Create notebook cells
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Model Architecture and Training\n",
                "This notebook recreates the model architecture and training process from the DND-Neural-Network application."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "import tensorflow as tf\n",
                "import numpy as np\n",
                "from tensorflow.keras.models import Sequential\n",
                "from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Input, Reshape"
            ],
            "outputs": []
        }
    ]
    
    # Add dataset loading cell
    dataset_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Load and Preprocess Dataset"]
    }
    cells.append(dataset_cell)
    
    dataset_code = []
    if dataset_name == "MNIST":
        dataset_code.extend([
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
        dataset_code.extend([
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
        dataset_code.extend([
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
        dataset_code.extend([
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
        dataset_code.extend([
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
        dataset_code.extend([
            "# Replace with your own dataset loading code",
            "# x_train, y_train = ...",
            "# x_test, y_test = ...",
            f"# Make sure to reshape your data to match the model input shape: {input_shape}"
        ])
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": dataset_code,
        "outputs": []
    })
    
    # Add model definition cell
    cells.extend([
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Define Model Architecture"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "model = Sequential()",
                f"model.add(Input(shape={input_shape_str}))",
                "# Add your model layers here",
                "# Example:",
                "# model.add(Dense(64, activation='relu'))",
                "# model.add(Dense(10, activation='softmax'))"
            ],
            "outputs": []
        }
    ])
    
    # Add model compilation and training cells
    cells.extend([
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Compile and Train Model"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Compile the model",
                "model.compile(",
                "    optimizer='adam',",
                "    loss='categorical_crossentropy',",
                "    metrics=['accuracy']",
                ")"
            ],
            "outputs": []
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Train the model",
                f"history = model.fit(",
                f"    x_train, y_train,",
                f"    epochs={epochs},",
                f"    batch_size={batch_size},",
                f"    validation_split={validation_split},",
                "    verbose=1",
                ")"
            ],
            "outputs": []
        }
    ])
    
    # Add model evaluation cell
    cells.extend([
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Evaluate Model"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Evaluate the model on test data",
                "test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)",
                "print(f'Test accuracy: {test_accuracy:.4f}')",
                "print(f'Test loss: {test_loss:.4f}')"
            ],
            "outputs": []
        }
    ])
    
    # Add model saving cell
    cells.extend([
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Save Model"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Save the model",
                "model.save('trained_model.h5')",
                "print('Model saved as trained_model.h5')"
            ],
            "outputs": []
        }
    ])
    
    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
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
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Convert the notebook dictionary to a JSON string
    return json.dumps(notebook, indent=2) 