def generate_pytorch_script(model, training_config, x_train_shape):
    """
    Generates a PyTorch script equivalent to the trained Keras model.
    
    Args:
        model: The trained Keras model
        training_config: Dictionary with training configuration
        x_train_shape: Shape of the training data
        
    Returns:
        str: Generated PyTorch script content
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
    
    # Map loss function names
    LOSS_FUNCTION_MAPPING = {
        "Categorical Cross-Entropy": "nn.CrossEntropyLoss()",
        "Binary Cross-Entropy": "nn.BCELoss()",
        "Mean Squared Error": "nn.MSELoss()",
        "Mean Absolute Error": "nn.L1Loss()",
        "Huber Loss": "nn.SmoothL1Loss()"
    }
    
    loss_function = training_config.get("lossFunction", "categorical_crossentropy")
    loss_value = LOSS_FUNCTION_MAPPING.get(loss_function, "nn.CrossEntropyLoss()")
    
    optimizer = training_config.get("optimizer", "adam").lower()
    epochs = training_config.get("epochs", 10)
    batch_size = training_config.get("batchSize", 32)
    dataset_name = training_config.get("dataset", "").lower()

    # Base imports
    pytorch_code = """
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
"""

    # Add attention-specific imports if needed
    if has_attention_layer:
        pytorch_code += """
from torch.nn import MultiheadAttention
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
        pytorch_code += """        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Adjust parameters as needed
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjust input size based on your model
        self.fc2 = nn.Linear(128, 10)  # Adjust output size based on your model
        self.dropout = nn.Dropout(0.25)
"""
    else:
        # It's a simpler MLP-like model
        input_size = x_train_shape[0] if len(x_train_shape) == 1 else x_train_shape[1]
        pytorch_code += f"""        # Fully connected layers
        self.fc1 = nn.Linear({input_size}, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # Adjust output size based on your model
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
    opt_name = opt_map.get(optimizer, "Adam")
    
    pytorch_code += f"""
# Initialize the model
model = NeuralNetwork()

# Define loss function and optimizer
criterion = {loss_value}
optimizer = optim.{opt_name}(model.parameters(), lr=0.001)

# Training loop
for epoch in range({epochs}):
    # Training
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            
            # Calculate accuracy (classification) or other metrics (regression)
            if criterion.__class__.__name__ in ["CrossEntropyLoss", "BCELoss"]:
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
    # Print epoch statistics
    print(f"Epoch {{epoch+1}}/{{{epochs}}}, Loss: {{running_loss/len(train_loader):.4f}}")
    if total > 0:
        print(f"Accuracy: {{100 * correct / total:.2f}}%")

# Save the model
torch.save(model.state_dict(), "pytorch_model.pth")

print("Training completed!")
"""

    return pytorch_code 