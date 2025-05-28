def generate_pytorch_script(model, training_config, x_train_shape, dataset_name):
    """
    Generate a PyTorch script that recreates the model architecture and training process.
    
    Args:
        model: The trained Keras model
        training_config: Dictionary with training configuration
        x_train_shape: Shape of the training data
        dataset_name: Name of the dataset being used
        
    Returns:
        str: Generated PyTorch script content
    """
    # Import helper functions from python_script module
    from .python_script import _is_custom_dataset, _load_custom_dataset_metadata
    
    # Extract training parameters
    epochs = training_config.get("epochs", 10)
    batch_size = training_config.get("batchSize", 32)
    learning_rate = training_config.get("learningRate", 0.001)
    optimizer_name = training_config.get("optimizer", "Adam").lower()
    loss_function = training_config.get("lossFunction", "Categorical Cross-Entropy")

    # Map loss function names to PyTorch equivalents
    LOSS_FUNCTION_MAPPING = {
        "Categorical Cross-Entropy": "nn.CrossEntropyLoss()",
        "Binary Cross-Entropy": "nn.BCEWithLogitsLoss()",
        "Mean Squared Error": "nn.MSELoss()",
        "Mean Absolute Error": "nn.L1Loss()",
        "Huber Loss": "nn.SmoothL1Loss()"
    }
    loss_value = LOSS_FUNCTION_MAPPING.get(loss_function, "nn.CrossEntropyLoss()")

    # Start building the PyTorch script
    pytorch_code = f"""import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {{device}}')

"""

    # Check if this is a custom dataset
    is_custom_dataset = _is_custom_dataset(dataset_name)
    
    if is_custom_dataset:
        # Handle custom dataset
        pytorch_code += _generate_pytorch_custom_dataset_code(dataset_name, batch_size)
    else:
        # Handle built-in datasets
        pytorch_code += _generate_pytorch_builtin_dataset_code(dataset_name, batch_size)

    # Check if the model contains attention layers
    has_attention_layer = False
    for layer in model.layers:
        layer_name = getattr(layer, 'name', '')
        if ('attention' in layer_name.lower() or 
            'multihead' in str(layer.__class__.__name__).lower() or
            'multiheadattention' in str(type(layer)).lower()):
            has_attention_layer = True
            break
    
    # Use the passed dataset_name parameter instead of getting it from training_config
    dataset_name = dataset_name.lower() if dataset_name else ""
    
    # Add the dataset name as a variable for the evaluation logic
    pytorch_code += f"\n# Define the dataset name for evaluation logic\ndataset_name = \"{dataset_name}\"\n"

    # Add attention-specific imports if needed
    if has_attention_layer:
        pytorch_code += """
from torch.nn import MultiheadAttention
"""

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
                    # Binary classification - use 1 unit for PyTorch (BCEWithLogitsLoss)
                    expected_output_units = 1
            else:
                # Regression
                expected_output_units = 1
        else:
            expected_output_units = 1  # default
    else:
        # Built-in datasets
        if dataset_name == "iris":
            expected_output_units = 3
        elif dataset_name == "breast cancer":
            expected_output_units = 1
        elif dataset_name == "california housing":
            expected_output_units = 1
        elif dataset_name == "mnist" or dataset_name == "cifar-10":
            expected_output_units = 10
        else:
            expected_output_units = 1  # default

    # Map loss function names
    LOSS_FUNCTION_MAPPING = {
        "Categorical Cross-Entropy": "nn.CrossEntropyLoss()",
        "Binary Cross-Entropy": "nn.BCELoss()",
        "Mean Squared Error": "nn.MSELoss()",
        "Mean Absolute Error": "nn.L1Loss()",
        "Huber Loss": "nn.SmoothL1Loss()"
    }
    
    loss_value = LOSS_FUNCTION_MAPPING.get(loss_function, "nn.CrossEntropyLoss()")
    
    # Base imports
    pytorch_code += """
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
"""

    # Dataset-specific preprocessing code
    if dataset_name == "iris":
        pytorch_code += f"""
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
X_train, X_test = torch.tensor(x_train, dtype=torch.float32), torch.tensor(x_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
"""

    elif dataset_name == "breast cancer":
        pytorch_code += f"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to tensors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
"""

    elif dataset_name == "california housing":
        pytorch_code += f"""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
"""

    elif dataset_name == "mnist":
        pytorch_code += f"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False)
"""

    elif dataset_name == "cifar-10":
        pytorch_code += f"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False)
"""
    else:
        # Generic case for tabular datasets
        pytorch_code += f"""
# Dataset loading code would go here based on your specific dataset
# Convert your data to PyTorch tensors
X_train = torch.tensor(x_train, dtype=torch.float32)
X_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
"""

    # Wrap tabular datasets in DataLoader
    if dataset_name in ["iris", "breast cancer", "california housing"]:
        pytorch_code += f"""
# Create TensorDatasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False)
"""

    # Model definition
    pytorch_code += """
# Define the PyTorch model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
"""

    # Placeholder for layers based on model type
    if 'Conv2D' in str(model.layers):
        # It's a CNN-like model
        pytorch_code += f"""        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Adjust parameters as needed
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjust input size based on your model
        self.fc2 = nn.Linear(128, {expected_output_units})  # Output layer sized for dataset: {dataset_name}
        self.dropout = nn.Dropout(0.25)
"""
    else:
        # It's a simpler MLP-like model
        input_size = x_train_shape[0] if len(x_train_shape) == 1 else x_train_shape[1]
        pytorch_code += f"""        # Fully connected layers
        self.fc1 = nn.Linear({input_size}, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, {expected_output_units})  # Output layer sized for dataset: {dataset_name}
        self.dropout = nn.Dropout(0.25)
"""

    # Forward function
    if 'Conv2D' in str(model.layers):
        pytorch_code += """
    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten and fully connected layers
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
"""
    else:
        pytorch_code += """
    def forward(self, x):
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
"""

    # Training code
    opt_map = {"adam": "Adam", "sgd": "SGD", "rmsprop": "RMSprop", "adagrad": "Adagrad"}
    opt_name = opt_map.get(optimizer_name, "Adam")
    
    # Adjust criterion based on dataset type for better compatibility
    if dataset_name == "iris" or dataset_name == "mnist" or dataset_name == "cifar-10":
        adjusted_loss = loss_value  # Use the mapped loss (usually CrossEntropyLoss for classification)
    elif dataset_name == "breast cancer":
        adjusted_loss = "nn.BCEWithLogitsLoss()"  # Better for binary classification
    elif dataset_name == "california housing":
        adjusted_loss = "nn.MSELoss()"  # Better for regression
    else:
        adjusted_loss = loss_value
    
    pytorch_code += f"""
# Initialize the model
model = NeuralNetwork()

# Define loss function and optimizer
criterion = {adjusted_loss}
optimizer = optim.{opt_name}(model.parameters(), lr={learning_rate})

# Training loop
for epoch in range({epochs}):
    # Training
    model.train()
    running_loss = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        if dataset_name == "iris" and criterion.__class__.__name__ == "CrossEntropyLoss":
            # Convert one-hot encoded targets to class indices for CrossEntropyLoss
            _, target_indices = torch.max(targets, 1)
            loss = criterion(outputs, target_indices)
        else:
            loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            
            # Calculate test loss
            if dataset_name == "iris" and criterion.__class__.__name__ == "CrossEntropyLoss":
                _, target_indices = torch.max(targets, 1)
                test_batch_loss = criterion(outputs, target_indices)
            else:
                test_batch_loss = criterion(outputs, targets)
                
            test_loss += test_batch_loss.item()
            
            # Calculate accuracy (classification) or other metrics (regression)
            if dataset_name == "breast cancer":
                # Binary classification
                predicted = (outputs > 0).float()  # Apply sigmoid in loss function
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            elif dataset_name == "california housing":
                # Regression - no accuracy calculation, use MSE
                pass
            elif dataset_name == "mnist" or dataset_name == "cifar-10":
                # MNIST/CIFAR-10 have class indices, not one-hot encoded labels
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            elif dataset_name == "iris":
                # Iris dataset is one-hot encoded in our preprocessing
                _, predicted = torch.max(outputs.data, 1)
                _, target_classes = torch.max(targets, 1)
                total += targets.size(0)
                correct += (predicted == target_classes).sum().item()
            else:
                # Generic classification approach
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                if targets.size(1) > 1:  # One-hot encoded
                    _, target_classes = torch.max(targets, 1)
                    correct += (predicted == target_classes).sum().item()
                else:  # Class indices
                    correct += (predicted == targets).sum().item()
                
    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_test_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total if total > 0 else 0
                
    # Print epoch statistics
    print(f"Epoch {{epoch+1}}/{{{epochs}}}, Loss: {{epoch_loss:.4f}}, Test Loss: {{epoch_test_loss:.4f}}")
    if total > 0 and dataset_name != "california housing":
        print(f"Accuracy: {{accuracy:.2f}}%")
    elif dataset_name == "california housing":
        print(f"Regression task - accuracy metric not applicable")

# Final evaluation
model.eval()
predictions = []
actual_labels = []

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        
        # Process predictions based on dataset type
        if dataset_name == "breast cancer":
            preds = (outputs > 0).float()
            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(targets.cpu().numpy())
        elif dataset_name == "california housing":
            predictions.extend(outputs.cpu().numpy())
            actual_labels.extend(targets.cpu().numpy())
        else:
            _, preds = torch.max(outputs, 1)
            if targets.size(1) > 1:  # One-hot encoded
                _, target_classes = torch.max(targets, 1)
                predictions.extend(preds.cpu().numpy())
                actual_labels.extend(target_classes.cpu().numpy())
            else:
                predictions.extend(preds.cpu().numpy())
                actual_labels.extend(targets.cpu().numpy())

# Plot confusion matrix for classification tasks
if dataset_name != "california housing":
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(actual_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

# Save the model
torch.save(model.state_dict(), "pytorch_model.pth")
print("Training completed!")
"""

    return pytorch_code


def _generate_pytorch_custom_dataset_code(dataset_name, batch_size):
    """
    Generate PyTorch code to load and preprocess a custom dataset.
    
    Args:
        dataset_name (str): Name of the custom dataset
        batch_size (int): Batch size for DataLoader
        
    Returns:
        str: PyTorch code for loading the custom dataset
    """
    from .python_script import _load_custom_dataset_metadata
    
    metadata = _load_custom_dataset_metadata(dataset_name)
    
    if not metadata:
        return f"""
# Custom dataset '{dataset_name}' metadata not found
# Please ensure the dataset files are available and update the paths below
# X_train = torch.tensor(x_train, dtype=torch.float32)
# X_test = torch.tensor(x_test, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32)
# y_test = torch.tensor(y_test, dtype=torch.float32)
"""
    
    task_type = metadata.get('task_type', 'classification')
    class_labels = metadata.get('class_labels', [])
    feature_columns = metadata.get('feature_columns', [])
    target_column = metadata.get('target_column', 'target')
    processed_shape = metadata.get('processed_shape', [0, 0])
    
    code = f"""
# Load custom dataset: {dataset_name}
# Note: Update the path below to point to your dataset file
dataset_file = '{dataset_name}.npz'  # Update this path as needed

# Load data from .npz file
data = np.load(dataset_file)
X = data['X']
y = data['y']

# Dataset info:
# - Task type: {task_type}
# - Features: {len(feature_columns)} ({', '.join(feature_columns)})
# - Target: {target_column}
"""
    
    if class_labels:
        code += f"# - Classes: {len(class_labels)} ({', '.join(map(str, class_labels))})\n"
    
    code += f"""# - Processed shape: {processed_shape}

# Preprocessing (same as used during training)
"""
    
    if task_type == 'classification':
        if len(class_labels) > 2:
            # Multi-class classification - use class indices for PyTorch CrossEntropyLoss
            code += f"""# For multi-class classification, use class indices (not one-hot)
# PyTorch CrossEntropyLoss expects class indices, not one-hot encoded labels
y_processed = y  # Assuming y contains class indices
"""
        else:
            # Binary classification - use single values for BCEWithLogitsLoss
            code += f"""# For binary classification, use single values (not one-hot)
# PyTorch BCEWithLogitsLoss expects single values, not one-hot encoded labels
y_processed = y.astype(np.float32)
"""
    else:
        # Regression
        code += f"""# For regression, use labels as-is
y_processed = y.reshape(-1, 1) if len(y.shape) == 1 else y
y_processed = y_processed.astype(np.float32)
"""
    
    code += f"""
# Split the data (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(
    X, y_processed, test_size=0.2, random_state=42"""
    
    if task_type == 'classification':
        code += ", stratify=y_processed"
    
    code += f""")

# Standardize features (same as used during training)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Convert to PyTorch tensors
X_train = torch.tensor(x_train, dtype=torch.float32)
X_test = torch.tensor(x_test, dtype=torch.float32)
"""
    
    if task_type == 'classification':
        if len(class_labels) > 2:
            code += "y_train = torch.tensor(y_train, dtype=torch.long)  # Class indices for CrossEntropyLoss\n"
            code += "y_test = torch.tensor(y_test, dtype=torch.long)\n"
        else:
            code += "y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Single values for BCEWithLogitsLoss\n"
            code += "y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)\n"
    else:
        code += "y_train = torch.tensor(y_train, dtype=torch.float32)\n"
        code += "y_test = torch.tensor(y_test, dtype=torch.float32)\n"
    
    code += f"""
# Create TensorDatasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False)

print(f'Dataset loaded: {{X_train.shape[0]}} training samples, {{X_test.shape[0]}} test samples')
print(f'Feature shape: {{X_train.shape[1:]}}')
print(f'Target shape: {{y_train.shape[1:] if len(y_train.shape) > 1 else "scalar"}}')
"""
    
    return code


def _generate_pytorch_builtin_dataset_code(dataset_name, batch_size):
    """
    Generate PyTorch code to load and preprocess built-in datasets.
    
    Args:
        dataset_name (str): Name of the built-in dataset
        batch_size (int): Batch size for DataLoader
        
    Returns:
        str: PyTorch code for loading the built-in dataset
    """
    if dataset_name == "iris":
        return f"""
from sklearn.datasets import load_iris

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
X_train, X_test = torch.tensor(x_train, dtype=torch.float32), torch.tensor(x_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# Create TensorDatasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False)
"""

    elif dataset_name == "breast cancer":
        return f"""
from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to tensors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Create TensorDatasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False)
"""

    elif dataset_name == "california housing":
        return f"""
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Create TensorDatasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False)
"""

    elif dataset_name == "mnist":
        return f"""
from torchvision import datasets, transforms

# Define transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False)
"""

    elif dataset_name == "cifar-10":
        return f"""
from torchvision import datasets, transforms

# Define transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False)
"""
    else:
        return f"""
# Dataset loading code would go here based on your specific dataset
# Convert your data to PyTorch tensors
X_train = torch.tensor(x_train, dtype=torch.float32)
X_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create TensorDatasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False)
""" 