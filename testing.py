import tensorflow as tf
import numpy as np
import wandb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Input, Reshape

# Initialize Weights & Biases for experiment tracking
wandb.init(project="dnd-neural-network", name="Iris-model", config={
    "dataset": "Iris",
    "optimizer": "adam",
    "loss_function": "categorical_crossentropy",
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 0.001
})

# Load and preprocess the dataset
# Load Iris dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Input(shape=(4,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))
# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Log model architecture with Weights & Biases
wandb.watch(model, log='all')

# Define Weights & Biases callback for logging
wandb_callback = wandb.keras.WandbCallback()

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[wandb_callback]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# Log additional evaluation metrics to W&B
if dataset_name in ['Iris', 'MNIST', 'CIFAR-10', 'Breast Cancer']:
    # Get predictions
    predictions = model.predict(x_test)
    if dataset_name == 'Breast Cancer':
        y_pred = (predictions > 0.5).astype(int)
        y_true = y_test
    else:
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
    
    # Log confusion matrix
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    wandb.log({'confusion_matrix': wandb.Image(plt)})

# Save the model
model.save('trained_model.keras')
wandb.save('trained_model.keras')  # Also save to W&B

# Visualize training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['aaccuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
wandb.log({'training_curves': wandb.Image(plt)})
plt.show()

# Finish the W&B run
wandb.finish()