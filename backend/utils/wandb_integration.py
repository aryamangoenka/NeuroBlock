import os
try:
    import wandb
    # hasattr guards against a stray wandb/ directory being picked up as an
    # empty namespace package instead of the real library
    WANDB_AVAILABLE = hasattr(wandb, "init")
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to prevent threading issues on macOS
import matplotlib.pyplot as plt
import io
from PIL import Image
import tensorflow as tf
from backend.utils.logging import get_logger

logger = get_logger(__name__)

class WandBLogger:
    """
    Utility class for Weights & Biases integration.
    Handles initializing runs, logging metrics, and managing the W&B lifecycle.
    """
    
    def __init__(self, config=None, project_name="dnd-neural-network"):
        """
        Initialize the W&B logger.
        
        Args:
            config (dict): Configuration dictionary for the W&B run
            project_name (str): The W&B project name
        """
        self.initialized = False
        self.project_name = project_name
        self.config = config or {}
        self.run = None
        self.run_id = None
    
    def init(self, model=None, config=None):
        """
        Initialize a new W&B run.
        
        Args:
            model: The Keras model to log
            config: Additional config to update the initial config
        """
        if not WANDB_AVAILABLE:
            logger.info("wandb is not installed; skipping experiment tracking")
            return None
        try:
            if config:
                self.config.update(config)
            
            # Initialize wandb
            self.run = wandb.init(project=self.project_name, config=self.config)
            self.run_id = self.run.id
            self.initialized = True
            
            # If model is provided, log its architecture
            if model:
                self.log_model_summary(model)
            
            logger.info(f"W&B run initialized with ID: {self.run_id}")
            return self.run_id
            
        except Exception as e:
            logger.error(f"Error initializing W&B: {str(e)}")
            return None
    
    def log_metrics(self, metrics, step=None):
        """
        Log metrics to W&B.
        
        Args:
            metrics (dict): Dictionary of metrics to log
            step (int): Optional step number
        """
        if not self.initialized:
            logger.warning("W&B not initialized, skipping metric logging")
            return
        
        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            logger.error(f"Error logging metrics to W&B: {str(e)}")
    
    def log_model_summary(self, model):
        """
        Log model architecture summary to W&B.
        
        Args:
            model: The Keras model to log
        """
        if not self.initialized:
            logger.warning("W&B not initialized, skipping model summary logging")
            return
            
        try:
            # Log model architecture as text
            stringlist = []
            model.summary(print_fn=lambda x: stringlist.append(x))
            model_summary = "\n".join(stringlist)
            wandb.run.summary["model_summary"] = model_summary
            
            # Log model graph (requires TensorFlow 2.x)
            try:
                wandb.watch(model, log="all")
            except Exception as watch_error:
                logger.warning(f"Could not watch model with W&B: {str(watch_error)}")
                
        except Exception as e:
            logger.error(f"Error logging model summary to W&B: {str(e)}")
    
    def log_confusion_matrix(self, y_true, y_pred, labels=None, title="Confusion Matrix"):
        """
        Log a confusion matrix visualization to W&B.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: List of label names
            title: Title for the confusion matrix
        """
        if not self.initialized:
            logger.warning("W&B not initialized, skipping confusion matrix logging")
            return
            
        try:
            # Convert one-hot encoded targets to class indices if needed
            if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                y_true = np.argmax(y_true, axis=1)
            
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                y_pred = np.argmax(y_pred, axis=1)
            
            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Create a figure and plot the confusion matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            cax = ax.matshow(cm, cmap=plt.cm.Blues)
            fig.colorbar(cax)
            
            # Set up the axes
            if labels:
                ax.set_xticklabels([''] + labels)
                ax.set_yticklabels([''] + labels)
            
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(title)
            
            # Add text annotations for each cell
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", 
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
            
            # Convert plot to image and log to W&B
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            wandb.log({title: wandb.Image(img)})
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error logging confusion matrix to W&B: {str(e)}")
    
    def log_image(self, image_array, caption="Image"):
        """
        Log an image to W&B.
        
        Args:
            image_array: Numpy array of the image
            caption: Caption for the image
        """
        if not self.initialized:
            logger.warning("W&B not initialized, skipping image logging")
            return
            
        try:
            wandb.log({caption: wandb.Image(image_array)})
        except Exception as e:
            logger.error(f"Error logging image to W&B: {str(e)}")
    
    def finish(self):
        """
        Finish the W&B run.
        """
        if not self.initialized:
            logger.warning("W&B not initialized, nothing to finish")
            return
            
        try:
            wandb.finish()
            logger.info(f"W&B run {self.run_id} finished")
            self.initialized = False
            self.run = None
        except Exception as e:
            logger.error(f"Error finishing W&B run: {str(e)}")

    def get_run_url(self):
        """
        Get the URL for the current W&B run.
        
        Returns:
            str: URL to the current run or None if not initialized
        """
        if not self.initialized or not self.run:
            return None
        
        return self.run.get_url()

# Create a singleton instance
wandb_logger = WandBLogger() 