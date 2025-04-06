import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, AveragePooling2D
from backend.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

def create_resnet_block(inputs, block_type="Basic", in_channels=64, out_channels=64, 
                        stride=[1, 1], activation="relu", use_skip_connection=True,
                        downsample_type="None"):
    """
    Create a ResNet block (Basic or Bottleneck)
    
    Args:
        inputs: Input tensor
        block_type: "Basic" or "Bottleneck"
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for convolution
        activation: Activation function to use
        use_skip_connection: Whether to use skip connection
        downsample_type: Type of downsampling for skip connection ("None", "Conv1x1", "AvgPool")
    
    Returns:
        Output tensor
    """
    logger.debug(f"Creating ResNet block", 
               extra={"context": {
                   "block_type": block_type,
                   "in_channels": in_channels,
                   "out_channels": out_channels,
                   "stride": stride,
                   "activation": activation,
                   "use_skip_connection": use_skip_connection,
                   "downsample_type": downsample_type
               }})
               
    # Convert activation name to lowercase for consistency
    activation = activation.lower()
    
    # Handle Basic block (2 conv layers with skip connection)
    if block_type == "Basic":
        # First convolution
        x = Conv2D(filters=out_channels, 
                  kernel_size=(3, 3),
                  strides=tuple(stride),
                  padding='same')(inputs)
        x = BatchNormalization()(x)
        
        if activation == "relu":
            x = Activation('relu')(x)
        elif activation == "leakyrelu":
            x = tf.keras.layers.LeakyReLU()(x)
        
        # Second convolution
        x = Conv2D(filters=out_channels,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  padding='same')(x)
        x = BatchNormalization()(x)
        
        # Skip connection
        if use_skip_connection:
            # If dimensions change (different number of channels or stride > 1),
            # we need to project the input to match the output dimensions
            if in_channels != out_channels or stride[0] > 1 or stride[1] > 1:
                logger.debug(f"Creating projection shortcut for Basic block", 
                           extra={"context": {
                               "method": downsample_type,
                               "in_channels": in_channels, 
                               "out_channels": out_channels,
                               "stride": stride
                           }})
                if downsample_type == "Conv1x1":
                    # 1x1 convolution to match dimensions
                    shortcut = Conv2D(filters=out_channels,
                                     kernel_size=(1, 1),
                                     strides=tuple(stride),
                                     padding='same')(inputs)
                    shortcut = BatchNormalization()(shortcut)
                elif downsample_type == "AvgPool":
                    # Average pooling followed by 1x1 conv if needed
                    shortcut = AveragePooling2D(pool_size=tuple(stride),
                                               strides=tuple(stride),
                                               padding='same')(inputs)
                    if in_channels != out_channels:
                        shortcut = Conv2D(filters=out_channels,
                                         kernel_size=(1, 1),
                                         strides=(1, 1),
                                         padding='same')(shortcut)
                        shortcut = BatchNormalization()(shortcut)
                else:
                    # Default to 1x1 conv
                    shortcut = Conv2D(filters=out_channels,
                                     kernel_size=(1, 1),
                                     strides=tuple(stride),
                                     padding='same')(inputs)
                    shortcut = BatchNormalization()(shortcut)
            else:
                shortcut = inputs
                
            # Add the skip connection
            x = Add()([x, shortcut])
        
        # Final activation
        if activation == "relu":
            x = Activation('relu')(x)
        elif activation == "leakyrelu":
            x = tf.keras.layers.LeakyReLU()(x)
            
    # Handle Bottleneck block (1x1 -> 3x3 -> 1x1 with skip connection)
    elif block_type == "Bottleneck":
        # For bottleneck blocks, out_channels is the final output channels
        # Typically, the internal channels are lower
        bottleneck_channels = out_channels // 4
        logger.debug(f"Creating Bottleneck block with internal channels: {bottleneck_channels}")
        
        # First 1x1 convolution to reduce dimensions
        x = Conv2D(filters=bottleneck_channels,
                  kernel_size=(1, 1),
                  strides=(1, 1),
                  padding='same')(inputs)
        x = BatchNormalization()(x)
        
        if activation == "relu":
            x = Activation('relu')(x)
        elif activation == "leakyrelu":
            x = tf.keras.layers.LeakyReLU()(x)
        
        # 3x3 convolution
        x = Conv2D(filters=bottleneck_channels,
                  kernel_size=(3, 3),
                  strides=tuple(stride),
                  padding='same')(x)
        x = BatchNormalization()(x)
        
        if activation == "relu":
            x = Activation('relu')(x)
        elif activation == "leakyrelu":
            x = tf.keras.layers.LeakyReLU()(x)
        
        # Final 1x1 convolution to restore dimensions
        x = Conv2D(filters=out_channels,
                  kernel_size=(1, 1),
                  strides=(1, 1),
                  padding='same')(x)
        x = BatchNormalization()(x)
        
        # Skip connection
        if use_skip_connection:
            # If dimensions change (different number of channels or stride > 1),
            # we need to project the input to match the output dimensions
            if in_channels != out_channels or stride[0] > 1 or stride[1] > 1:
                logger.debug(f"Creating projection shortcut for Bottleneck block", 
                           extra={"context": {
                               "method": downsample_type,
                               "in_channels": in_channels, 
                               "out_channels": out_channels,
                               "stride": stride
                           }})
                if downsample_type == "Conv1x1":
                    # 1x1 convolution to match dimensions
                    shortcut = Conv2D(filters=out_channels,
                                     kernel_size=(1, 1),
                                     strides=tuple(stride),
                                     padding='same')(inputs)
                    shortcut = BatchNormalization()(shortcut)
                elif downsample_type == "AvgPool":
                    # Average pooling followed by 1x1 conv if needed
                    shortcut = AveragePooling2D(pool_size=tuple(stride),
                                               strides=tuple(stride),
                                               padding='same')(inputs)
                    if in_channels != out_channels:
                        shortcut = Conv2D(filters=out_channels,
                                         kernel_size=(1, 1),
                                         strides=(1, 1),
                                         padding='same')(shortcut)
                        shortcut = BatchNormalization()(shortcut)
                else:
                    # Default to 1x1 conv
                    shortcut = Conv2D(filters=out_channels,
                                     kernel_size=(1, 1),
                                     strides=tuple(stride),
                                     padding='same')(inputs)
                    shortcut = BatchNormalization()(shortcut)
            else:
                shortcut = inputs
                
            # Add the skip connection
            x = Add()([x, shortcut])
        
        # Final activation
        if activation == "relu":
            x = Activation('relu')(x)
        elif activation == "leakyrelu":
            x = tf.keras.layers.LeakyReLU()(x)
    
    return x 