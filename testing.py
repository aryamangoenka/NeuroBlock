
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Define the dataset name for evaluation logic
dataset_name = "iris"

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

# Create TensorDatasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=31, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=31, shuffle=False)

# Define the PyTorch model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # Output layer sized for dataset: iris
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# Initialize the model
model = NeuralNetwork()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(13):
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
    print(f"Epoch {epoch+1}/{13}, Loss: {epoch_loss:.4f}, Test Loss: {epoch_test_loss:.4f}")
    if total > 0 and dataset_name != "california housing":
        print(f"Accuracy: {accuracy:.2f}%")
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
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend to prevent threading issues on macOS
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
