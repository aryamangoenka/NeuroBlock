import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Input, AveragePooling2D, Add, Activation, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Import the create_resnet_block function from main.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
from main import create_resnet_block

print("Testing ResNet Block Implementation")

# Load and preprocess CIFAR-10 dataset
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define image shape
input_shape = (32, 32, 3)

# Test 1: Simple ResNet model with Basic blocks
def build_resnet_basic_model():
    inputs = Input(shape=input_shape)
    
    # Initial convolution
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    # ResNet Basic Block 1
    x = create_resnet_block(
        x, 
        block_type="Basic",
        in_channels=64,
        out_channels=64,
        stride=[1, 1],
        activation="relu",
        use_skip_connection=True,
        downsample_type="None"
    )
    
    # ResNet Basic Block 2
    x = create_resnet_block(
        x, 
        block_type="Basic",
        in_channels=64,
        out_channels=128,
        stride=[2, 2],
        activation="relu",
        use_skip_connection=True,
        downsample_type="Conv1x1"
    )
    
    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    
    # Dense output layer
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Test 2: ResNet model with Bottleneck blocks
def build_resnet_bottleneck_model():
    inputs = Input(shape=input_shape)
    
    # Initial convolution
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    # Bottleneck Block 1
    x = create_resnet_block(
        x, 
        block_type="Bottleneck",
        in_channels=64,
        out_channels=256,
        stride=[1, 1],
        activation="relu",
        use_skip_connection=True,
        downsample_type="Conv1x1"
    )
    
    # Bottleneck Block 2
    x = create_resnet_block(
        x, 
        block_type="Bottleneck",
        in_channels=256,
        out_channels=512,
        stride=[2, 2],
        activation="relu",
        use_skip_connection=True,
        downsample_type="Conv1x1"
    )
    
    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    
    # Dense output layer
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Test 3: ResNet model without skip connections (for comparison)
def build_plain_model():
    inputs = Input(shape=input_shape)
    
    # Initial convolution
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    # Plain Block 1 (no skip connection)
    x = create_resnet_block(
        x, 
        block_type="Basic",
        in_channels=64,
        out_channels=64,
        stride=[1, 1],
        activation="relu",
        use_skip_connection=False,
        downsample_type="None"
    )
    
    # Plain Block 2 (no skip connection)
    x = create_resnet_block(
        x, 
        block_type="Basic",
        in_channels=64,
        out_channels=128,
        stride=[2, 2],
        activation="relu",
        use_skip_connection=False,
        downsample_type="None"
    )
    
    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    
    # Dense output layer
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Train the models
def train_and_evaluate(model, name):
    print(f"\nTraining {name} model...")
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Define early stopping
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        x_train[:5000], y_train[:5000],  # Use subset for faster testing
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    print(f"\nEvaluating {name} model...")
    test_loss, test_acc = model.evaluate(x_test[:1000], y_test[:1000], verbose=0)
    print(f"{name} Test accuracy: {test_acc:.4f}")
    
    return history, test_acc

# Build and train models
models = [
    ("ResNet Basic", build_resnet_basic_model()),
    ("ResNet Bottleneck", build_resnet_bottleneck_model()),
    ("Plain Network", build_plain_model())
]

results = {}

for name, model in models:
    history, test_acc = train_and_evaluate(model, name)
    results[name] = {
        'history': history,
        'test_acc': test_acc
    }

# Compare results
print("\nModel Comparison:")
for name, result in results.items():
    print(f"{name}: Test accuracy = {result['test_acc']:.4f}")

# Visualize results
plt.figure(figsize=(12, 5))

# Plot training accuracy
plt.subplot(1, 2, 1)
for name, result in results.items():
    plt.plot(result['history'].history['accuracy'], label=f'{name} Training')
    plt.plot(result['history'].history['val_accuracy'], label=f'{name} Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Plot training loss
plt.subplot(1, 2, 2)
for name, result in results.items():
    plt.plot(result['history'].history['loss'], label=f'{name} Training')
    plt.plot(result['history'].history['val_loss'], label=f'{name} Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('resnet_test_results.png')
print("\nResults plot saved as 'resnet_test_results.png'")
print("ResNet testing complete!") 