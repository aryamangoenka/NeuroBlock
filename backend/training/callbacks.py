import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from backend.utils.logging import get_logger
from backend.utils.wandb_integration import wandb_logger

# Initialize logger
logger = get_logger(__name__)

class RealTimeUpdateCallback(Callback):
    """
    Callback to emit training progress via WebSockets.
    
    This callback sends real-time updates to the client as the model trains.
    """
    
    def __init__(self, socketio, client_id, total_epochs):
        """
        Initialize the callback.
        
        Args:
            socketio: The Flask-SocketIO instance
            client_id: The client ID to emit updates to
            total_epochs: The total number of epochs for training
        """
        super().__init__()
        self.socketio = socketio
        self.client_id = client_id
        self.total_epochs = total_epochs
        logger.debug(f"RealTimeUpdateCallback initialized", 
                    extra={"context": {"client_id": client_id, "total_epochs": total_epochs}})

    def on_epoch_end(self, epoch, logs=None):
        """
        Emit training progress after each epoch.
        
        Args:
            epoch: The current epoch
            logs: Dict of metrics from training
        """
        if logs is not None:
            logger.debug(f"Emitting training progress for epoch {epoch + 1}/{self.total_epochs}", 
                        extra={"context": {
                            "client_id": self.client_id, 
                            "epoch": epoch + 1,
                            "loss": logs.get("loss"),
                            "accuracy": logs.get("accuracy")
                        }})
            
            # Also log to W&B for each epoch
            try:
                wandb_metrics = {
                    "epoch": epoch + 1,
                    "loss": logs.get("loss"),
                    "accuracy": logs.get("accuracy"),
                    "val_loss": logs.get("val_loss"),
                    "val_accuracy": logs.get("val_accuracy"),
                }
                wandb_logger.log_metrics(wandb_metrics, step=epoch + 1)
            except Exception as e:
                logger.warning(f"Error logging to W&B in callback: {str(e)}")
            
            self.socketio.emit("training_progress", {
                "epoch": epoch + 1,
                "total_epochs": self.total_epochs,
                "loss": logs.get("loss"),
                "accuracy": logs.get("accuracy"),
                "val_loss": logs.get("val_loss"),
                "val_accuracy": logs.get("val_accuracy"),
            }, to=self.client_id)
        
        # Check for stop flag
        from backend.api.sockets import get_stop_flag
        if get_stop_flag(self.client_id):
            logger.info(f"Training interrupted by client disconnection", 
                      extra={"context": {"client_id": self.client_id, "epoch": epoch + 1}})
            raise KeyboardInterrupt("Training stopped by the client disconnecting.") 