import os
import json
import tensorflow as tf
from flask import request
from flask_socketio import emit
from backend.models.builder import build_model_from_architecture
from backend.training.callbacks import RealTimeUpdateCallback
from backend.dataset_loader import load_dataset
import numpy as np
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to prevent threading issues on macOS
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import pandas as pd
from backend.utils.logging import get_logger
from backend.utils.wandb_integration import wandb_logger

# Initialize logger
logger = get_logger(__name__)

# Dictionary to track stop flags for each client
stop_flags = {}

# Store the latest training configuration globally
latest_training_config = {}
x_train_shape=()

# Path to store the trained model - use absolute paths based on project root
EXPORT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "exports")
os.makedirs(EXPORT_FOLDER, exist_ok=True)  # Ensure the folder exists
TRAINED_MODEL_PATH = os.path.join(EXPORT_FOLDER, "trained_model.keras")
MODEL_ARCHITECTURE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "saved_model.json")

def get_stop_flag(client_id):
    """Get the stop flag for a client."""
    return stop_flags.get(client_id, False)

def set_stop_flag(client_id, value):
    """Set the stop flag for a client."""
    stop_flags[client_id] = value

def register_socket_events(socketio):
    """Register all socket event handlers."""
    
    @socketio.on("connect")
    def handle_connect():
        """Handle WebSocket connection."""
        client_id = request.sid  # Unique client ID
        set_stop_flag(client_id, False)  # Initialize stop flag for the client
        
        logger.info(f"Client connected to WebSocket", extra={"context": {"client_id": client_id}})
        emit("message", {"type": "info", "data": "Connected to WebSocket!"})

    @socketio.on("disconnect")
    def handle_disconnect():
        """Handle WebSocket disconnection."""
        client_id = request.sid  # Unique client ID
        set_stop_flag(client_id, True)  # Set stop flag for the client
        
        logger.info(f"Client disconnected from WebSocket", extra={"context": {"client_id": client_id}})

    @socketio.on("start_training")
    def start_training(data):
        """
        Handle the WebSocket event to start training and send real-time updates.
        """
        global latest_training_config
        global x_train_shape
        
        LOSS_FUNCTION_MAPPING = {
            "Categorical Cross-Entropy": "categorical_crossentropy",
            "Binary Cross-Entropy": "binary_crossentropy",
            "Mean Squared Error": "mse",
            "Mean Absolute Error": "mae",
            "Huber Loss": "huber"
        }
        client_id = request.sid  # Unique client ID
        
        logger.info("Start training request received", 
                   extra={"context": {"client_id": client_id}})
        
        try:
            # Ensure model architecture exists
            if not os.path.exists(MODEL_ARCHITECTURE_FILE):
                logger.error("Model architecture not found", 
                           extra={"context": {"client_id": client_id}})
                emit("training_error", {"error": "Model architecture not found. Please save it first."})
                return

            # Load the saved model architecture
            with open(MODEL_ARCHITECTURE_FILE, "r") as f:
                model_architecture = json.load(f)

            # Extract dataset and training configuration
            dataset = model_architecture.get("dataset")
            if not dataset:
                logger.error("Dataset information missing in model architecture", 
                           extra={"context": {"client_id": client_id}})
                emit("training_error", {"error": "Dataset information missing in model architecture"})
                return

            training_config = data
            logger.info("Training configuration received", 
                       extra={"context": {
                           "client_id": client_id,
                           "dataset": dataset,
                           "epochs": training_config.get("epochs"),
                           "batch_size": training_config.get("batchSize"),
                           "optimizer": training_config.get("optimizer"),
                           "loss_function": training_config.get("lossFunction")
                       }})
            latest_training_config = data
            
            # Map loss function
            loss_function = LOSS_FUNCTION_MAPPING.get(training_config["lossFunction"])
            if not loss_function:
                logger.error(f"Invalid loss function", 
                           extra={"context": {
                               "client_id": client_id,
                               "loss_function": training_config["lossFunction"]
                           }})
                emit("training_error", {"error": f"Invalid loss function: {training_config['lossFunction']}"})
                return

            # Load and preprocess the dataset using load_dataset
            logger.info(f"Loading dataset: {dataset}", 
                       extra={"context": {"client_id": client_id}})
            (x_train, y_train), (x_test, y_test) = load_dataset(dataset)
            logger.debug(f"Dataset loaded successfully: {dataset}",
                        extra={"context": {
                            "client_id": client_id,
                            "x_train_shape": x_train.shape,
                            "y_train_shape": y_train.shape,
                            "x_test_shape": x_test.shape,
                            "y_test_shape": y_test.shape
                        }})

            # Use tf.data.Dataset for efficient batching
            batch_size = training_config["batchSize"]
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size, drop_remainder=False)
            val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size, drop_remainder=False)

            # Clear previous TensorFlow session to reset the model
            tf.keras.backend.clear_session()
            logger.debug("TensorFlow session cleared", extra={"context": {"client_id": client_id}})

            # Build the model
            logger.info("Building model", extra={"context": {"client_id": client_id}})
            model = build_model_from_architecture(model_architecture, x_train.shape[1:], dataset)
            x_train_shape = x_train.shape[1:]
            
            # Compile the model
            logger.info("Compiling model", 
                       extra={"context": {
                           "client_id": client_id,
                           "optimizer": training_config["optimizer"].lower(),
                           "loss": loss_function
                       }})
            model.compile(
                optimizer=training_config["optimizer"].lower(),
                loss=loss_function,
                metrics=["accuracy"]
            )

            # Initialize Weights & Biases
            wandb_config = {
                "dataset": dataset,
                "optimizer": training_config["optimizer"].lower(),
                "loss_function": loss_function,
                "batch_size": batch_size,
                "epochs": training_config["epochs"],
                "model_architecture": model_architecture.get("architecture", [])
            }
            run_id = wandb_logger.init(model=model, config=wandb_config)
            logger.info(f"W&B run initialized with ID: {run_id}",
                       extra={"context": {"client_id": client_id}})
            
            # Emit a message that training is starting
            emit("training_start", {
                "message": "Training has started!",
                "wandb_run_url": wandb_logger.get_run_url()
            })
            logger.info("Training started", extra={"context": {"client_id": client_id}})
            
            # Define total epochs and stage size
            total_epochs = training_config["epochs"]
            if dataset in ["Iris", "Breast Cancer"]:
                stage_size = 5
            elif dataset in ["MNIST"]:
                stage_size = 3
            elif dataset == "CIFAR-10":
                stage_size = 1
            else:
                stage_size = 2  # Default
                
            logger.debug(f"Training configuration: {total_epochs} epochs, {stage_size} epochs per stage",
                        extra={"context": {"client_id": client_id}})

            # Train the model in stages
            for stage_start in range(0, total_epochs, stage_size):
                if get_stop_flag(client_id):
                    logger.info(f"Training stopped by client", 
                               extra={"context": {"client_id": client_id}})
                    emit("training_stopped", {"message": "Training was stopped by the client."})
                    wandb_logger.finish()  # Finish W&B run on stop
                    return
                
                # Calculate the number of epochs for this stage
                current_stage_size = min(stage_size, total_epochs - stage_start)
                
                logger.info(f"Starting training stage: epochs {stage_start+1}-{stage_start+current_stage_size} of {total_epochs}",
                           extra={"context": {"client_id": client_id}})

                # Define the callback for real-time updates
                callback = RealTimeUpdateCallback(socketio, client_id, total_epochs)

                # Train the model for the current stage
                history = model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=stage_start + current_stage_size,
                    initial_epoch=stage_start,
                    verbose=0,  # Suppress built-in progress bar
                    callbacks=[callback]  # Add the custom callback for real-time updates
                )

                # Emit staged progress
                for epoch_offset in range(current_stage_size):
                    epoch = stage_start + epoch_offset + 1
                    logger.debug(f"Emitting progress for epoch {epoch}",
                                extra={"context": {"client_id": client_id}})
                    
                    # Log metrics to W&B
                    wandb_metrics = {
                        "loss": history.history["loss"][epoch_offset],
                        "accuracy": history.history["accuracy"][epoch_offset],
                        "val_loss": history.history["val_loss"][epoch_offset],
                        "val_accuracy": history.history["val_accuracy"][epoch_offset],
                    }
                    wandb_logger.log_metrics(wandb_metrics, step=epoch)
                    
                    emit("training_progress_stage", {
                        "epoch": epoch,
                        "total_epochs": total_epochs,
                        "loss": history.history["loss"][epoch_offset],
                        "accuracy": history.history["accuracy"][epoch_offset],
                        "val_loss": history.history["val_loss"][epoch_offset],
                        "val_accuracy": history.history["val_accuracy"][epoch_offset],
                    })
                    socketio.sleep(0)

            final_metrics = {}
            
            logger.info("Training completed, calculating final metrics", 
                       extra={"context": {"client_id": client_id, "dataset": dataset}})

            # Additional metrics for classification datasets
            if dataset in ["Iris", "MNIST", "CIFAR-10", "Breast Cancer"]:
                logger.debug("Calculating classification metrics", 
                           extra={"context": {"client_id": client_id}})
                predictions = model.predict(x_test)
                if dataset == "Breast Cancer":
                    # Binary classification: Apply threshold for class prediction
                    y_pred = (predictions > 0.5).astype(int) # Convert to 0 or 1
                    y_true = y_test  # Ensure y_test is also flat
                else:
                    # Multi-class classification
                    y_pred = np.argmax(predictions, axis=1)
                    y_true = np.argmax(y_test, axis=1)
                conf_matrix = confusion_matrix(y_true, y_pred).tolist()  # Convert to list for JSON serialization
                final_metrics["confusion_matrix"] = conf_matrix
                logger.debug("Confusion matrix calculated", 
                           extra={"context": {"client_id": client_id}})
                
                # Log confusion matrix to W&B
                class_labels = []
                if dataset == "Iris":
                    class_labels = ["Setosa", "Versicolor", "Virginica"]
                elif dataset == "MNIST" or dataset == "CIFAR-10":
                    if dataset == "MNIST":
                        class_labels = [str(i) for i in range(10)]
                    else:  # CIFAR-10
                        class_labels = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
                
                wandb_logger.log_confusion_matrix(y_true, y_pred, labels=class_labels)
                
            elif dataset == "California Housing":
                logger.debug("Calculating regression metrics", 
                           extra={"context": {"client_id": client_id}})
                predictions = model.predict(x_test)

                # Ensure predictions and y_test are NumPy arrays
                predictions = predictions if isinstance(predictions, np.ndarray) else predictions.numpy()
                y_test = y_test if isinstance(y_test, np.ndarray) else y_test.numpy()

                # Calculate residuals
                residuals = (y_test - predictions).tolist()

                # Compute Regression Metrics
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                r2 = r2_score(y_test, predictions)
                
                logger.debug(f"Regression metrics calculated: RMSE={rmse:.4f}, R²={r2:.4f}", 
                           extra={"context": {"client_id": client_id}})

                # Save metrics
                final_metrics["rmse"] = rmse
                final_metrics["r2"] = r2

                # Send raw data for residual plot
                final_metrics["residuals_plot"] = {
                    "predicted_values": predictions.tolist(),
                    "residuals": residuals
                }

                logger.debug("Residual plot data prepared", 
                           extra={"context": {"client_id": client_id}})

                # ------------------ Multicollinearity Heatmap ------------------
                # Convert feature set to DataFrame for easy correlation calculation
                feature_names = [
                    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
                    "Population", "AveOccup", "Latitude", "Longitude"
                ]  # Replace with actual feature names if different
                
                # Convert Tensor to NumPy array if needed
                x_test_np = x_test if isinstance(x_test, np.ndarray) else x_test.numpy()

                x_test_df = pd.DataFrame(x_test_np, columns=feature_names)

                # Compute Pearson Correlation Matrix
                correlation_matrix = x_test_df.corr().values  # Get correlation values
                
                logger.debug("Generating multicollinearity heatmap", 
                           extra={"context": {"client_id": client_id}})

                # Generate heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    correlation_matrix,
                    annot=True,
                    cmap="coolwarm",
                    center=0,
                    linewidths=0.5,
                    fmt=".2f",
                    annot_kws={"size": 10}
                )
                plt.title("Multicollinearity Heatmap")

                # Convert heatmap to base64
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                heatmap_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                buf.close()
                plt.close()

                # Add heatmap to metrics
                final_metrics["multicollinearity_heatmap"] = heatmap_base64
                logger.debug("Multicollinearity heatmap generated", 
                           extra={"context": {"client_id": client_id}})

            # Emit final training results
            logger.info("Sending final training results to client", 
                       extra={"context": {"client_id": client_id}})
            emit("training_complete", {
                "message": "Training completed successfully!",
                "metrics": final_metrics,
                "loss_over_time": history.history["loss"],
                "val_loss_over_time": history.history.get("val_loss", []),
                "success": True
            })
            
            # Save the model with custom_objects for Lambda layers
            try:
                logger.info("Saving trained model", 
                           extra={"context": {"client_id": client_id, "path": TRAINED_MODEL_PATH}})
                # First try saving with include_optimizer=False which often helps with Lambda layers
                model.save(TRAINED_MODEL_PATH, include_optimizer=False)
                logger.info("Model saved successfully", 
                           extra={"context": {"client_id": client_id}})
            except Exception as save_error:
                logger.warning(f"Could not save model with standard method: {str(save_error)}",
                             extra={"context": {"client_id": client_id}})
                
                # Alternative approach: Save only the weights
                try:
                    weights_path = os.path.join(EXPORT_FOLDER, "model_weights.h5")
                    model.save_weights(weights_path)
                    logger.info(f"Saved model weights to {weights_path}",
                               extra={"context": {"client_id": client_id}})
                    
                    # Save the architecture separately
                    arch_path = os.path.join(EXPORT_FOLDER, "model_architecture.json")
                    with open(arch_path, "w") as f:
                        f.write(model.to_json())
                    logger.info("Saved model architecture separately",
                               extra={"context": {"client_id": client_id}})
                except Exception as weights_error:
                    logger.error(f"Could not save model weights: {str(weights_error)}",
                                extra={"context": {"client_id": client_id}})
            
            # Finish W&B run
            wandb_logger.finish()
            logger.info("W&B run completed", extra={"context": {"client_id": client_id}})
                
        except Exception as e:
            logger.error(f"Training error: {str(e)}", 
                        extra={"context": {"client_id": client_id}}, 
                        exc_info=True)
            emit("training_error", {"error": str(e)})
            
            # Ensure W&B run is properly closed even on error
            try:
                wandb_logger.finish()
            except:
                pass 