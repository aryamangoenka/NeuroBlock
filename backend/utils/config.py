import os
from dotenv import load_dotenv
from backend.utils.logging import get_logger

logger = get_logger(__name__)

# Try to load .env file if it exists
load_dotenv()

class Config:
    """Configuration manager for environment variables"""
    
    @staticmethod
    def get_wandb_api_key():
        """
        Get the Weights & Biases API key from environment variables.
        Returns:
            str: The W&B API key or None if not set
        """
        api_key = os.environ.get('WANDB_API_KEY')
        if not api_key:
            logger.warning("WANDB_API_KEY not found in environment variables")
        return api_key
    
    @staticmethod
    def set_wandb_api_key(api_key):
        """
        Set the Weights & Biases API key in environment variables.
        Args:
            api_key (str): The W&B API key
        """
        os.environ['WANDB_API_KEY'] = api_key
        logger.info("WANDB_API_KEY set in environment variables")

# Initialize configuration
config = Config() 