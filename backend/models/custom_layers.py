import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention
from backend.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# Define custom attention function for Lambda layer
@tf.keras.utils.register_keras_serializable(package='custom_layers')
def apply_attention(x, num_heads=8, key_dim=64, dropout=0.0):
    logger.debug(f"Applying attention with Lambda layer", 
                extra={"context": {"num_heads": num_heads, "key_dim": key_dim, "dropout": dropout}})
    attention_layer = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout
    )
    return attention_layer(x, x)

# Create a custom attention layer class instead of using Lambda
@tf.keras.utils.register_keras_serializable(package='custom_layers')
class CustomAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads=8, key_dim=64, dropout=0.0, **kwargs):
        super(CustomAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout = dropout
        self.attention = None  # Will be initialized in build()
        logger.debug(f"CustomAttentionLayer initialized", 
                    extra={"context": {
                        "num_heads": num_heads, 
                        "key_dim": key_dim, 
                        "dropout": dropout,
                        "name": kwargs.get("name", "unnamed")
                    }})
        
    def build(self, input_shape):
        logger.debug(f"Building CustomAttentionLayer with input shape: {input_shape}", 
                    extra={"context": {"name": self.name}})
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout
        )
        super(CustomAttentionLayer, self).build(input_shape)
        
    def call(self, inputs, training=None):
        return self.attention(inputs, inputs, training=training)
        
    def get_config(self):
        config = super(CustomAttentionLayer, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout': self.dropout
        })
        return config

# Enable unsafe deserialization for Lambda layers
tf.keras.utils.register_keras_serializable(package='custom_layers')(apply_attention)
tf.keras.config.enable_unsafe_deserialization()
logger.info("Custom attention layers registered and unsafe deserialization enabled") 