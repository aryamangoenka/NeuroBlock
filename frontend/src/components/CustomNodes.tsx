
import { Handle, Position } from "reactflow";
import "../styles/components/CustomNodes.scss";

// Enhanced handle style for better usability
const enhancedHandleStyle = {
  background: "#fff",
  border: "3px solid #333",
  width: "16px",
  height: "16px",
  borderRadius: "50%",
  cursor: "pointer",
  boxShadow: "0 2px 8px rgba(0, 0, 0, 0.2)",
  zIndex: 10,
};

// Source handle (right side) with distinct color
const sourceHandleStyle = {
  ...enhancedHandleStyle,
  background: "#4CAF50",
  border: "3px solid #2E7D32",
};

// Target handle (left side) with distinct color
const targetHandleStyle = {
  ...enhancedHandleStyle,
  background: "#2196F3",
  border: "3px solid #1565C0",
};

// Export all node components
export {
  DenseNode,
  ConvolutionNode,
  MaxPoolingNode,
  GlobalAveragePoolNode,
  FlattenNode,
  DropoutNode,
  BatchNormalizationNode,
  AttentionNode,
  InputNode,
  ResNetBlockNode,
  OutputNode,
  AddLayerNode,
  ActivationNode,
  CustomBlockNode,
};

// Dense Node
const DenseNode = ({ data }: { data: any }) => (
  <div className="custom-node dense-node">
    <Handle
      type="target"
      position={Position.Left}
      style={targetHandleStyle}
      isConnectable={true}
    />
    <h4>{data.label || "Dense Layer"}</h4>
    <p>Neurons: {data.neurons}</p>
    <Handle
      type="source"
      position={Position.Right}
      style={sourceHandleStyle}
      isConnectable={true}
    />
  </div>
);

// Convolutional Node
const ConvolutionNode = ({ data }: { data: any }) => (
  <div className="custom-node convolution-node">
    <Handle
      type="target"
      position={Position.Left}
      style={targetHandleStyle}
      isConnectable={true}
    />
    <h4>{data.label || "Convolution Layer"}</h4>
    <p>Filters: {data.filters}</p>
    <p>Kernel Size: {data.kernelSize?.join("x")}</p>
    <p>Stride: {data.stride?.join("x")}</p>
    <p>Padding: {data.padding || "valid"}</p>
    <Handle
      type="source"
      position={Position.Right}
      style={sourceHandleStyle}
      isConnectable={true}
    />
  </div>
);

// Max Pooling Node
const MaxPoolingNode = ({ data }: { data: any }) => (
  <div className="custom-node maxpooling-node">
    <Handle
      type="target"
      position={Position.Left}
      style={targetHandleStyle}
      isConnectable={true}
    />
    <h4>{data.label || "Max Pooling Layer"}</h4>
    <p>Pool Size: {data.poolSize?.join("x")}</p>
    <p>Stride: {data.stride?.join("x")}</p>
    <p>Padding: {data.padding || "valid"}</p>
    <Handle
      type="source"
      position={Position.Right}
      style={sourceHandleStyle}
      isConnectable={true}
    />
  </div>
);

// Global Average Pooling Node
const GlobalAveragePoolNode = ({ data }: { data: any }) => (
  <div className="custom-node globalaveragepool-node">
    <Handle
      type="target"
      position={Position.Left}
      style={targetHandleStyle}
      isConnectable={true}
    />
    <h4>{data.label || "Global Average Pooling Layer"}</h4>
    <Handle
      type="source"
      position={Position.Right}
      style={sourceHandleStyle}
      isConnectable={true}
    />
  </div>
);

// Flatten Node
const FlattenNode = ({ data }: { data: any }) => (
  <div className="custom-node flatten-node">
    <Handle
      type="target"
      position={Position.Left}
      style={targetHandleStyle}
      isConnectable={true}
    />
    <h4>{data.label || "Flatten Layer"}</h4>
    <Handle
      type="source"
      position={Position.Right}
      style={sourceHandleStyle}
      isConnectable={true}
    />
  </div>
);

// Dropout Node
const DropoutNode = ({ data }: { data: any }) => (
  <div className="custom-node dropout-node">
    <Handle
      type="target"
      position={Position.Left}
      style={targetHandleStyle}
      isConnectable={true}
    />
    <h4>{data.label || "Dropout Layer"}</h4>
    <p>Rate: {data.rate}</p>
    <Handle
      type="source"
      position={Position.Right}
      style={sourceHandleStyle}
      isConnectable={true}
    />
  </div>
);

// Batch Normalization Node
const BatchNormalizationNode = ({ data }: { data: any }) => (
  <div className="custom-node batchnormalization-node">
    <Handle
      type="target"
      position={Position.Left}
      style={targetHandleStyle}
      isConnectable={true}
    />
    <h4>{data.label || "Batch Normalization Layer"}</h4>
    <p>Momentum: {data.momentum}</p>
    <p>Epsilon: {data.epsilon}</p>
    <Handle
      type="source"
      position={Position.Right}
      style={sourceHandleStyle}
      isConnectable={true}
    />
  </div>
);

// Attention Node
const AttentionNode = ({ data }: { data: any }) => (
  <div className="custom-node attention-node">
    <Handle
      type="target"
      position={Position.Left}
      style={targetHandleStyle}
      isConnectable={true}
    />
    <h4>{data.label || "Attention Layer"}</h4>
    <p>Heads: {data.heads}</p>
    <p>Key Dim: {data.keyDim}</p>
    <p>Dropout: {data.dropout}</p>
    <Handle
      type="source"
      position={Position.Right}
      style={sourceHandleStyle}
      isConnectable={true}
    />
  </div>
);

// Input Node - only has source handle
const InputNode = ({ data }: { data: any }) => (
  <div className="custom-node input-node">
    <h4>{data.label || "Input Layer"}</h4>
    <Handle
      type="source"
      position={Position.Right}
      style={{
        ...sourceHandleStyle,
        background: "#FF9800", // Special color for input
        border: "3px solid #F57C00",
      }}
      isConnectable={true}
    />
  </div>
);

// ResNetBlock Node
const ResNetBlockNode = ({ data }: { data: any }) => (
  <div
    className={`custom-node resnetblock-node ${
      data.blockType?.toLowerCase() || "basic"
    }-block`}
  >
    <Handle
      type="target"
      position={Position.Left}
      style={targetHandleStyle}
      id="target"
      isConnectable={true}
    />
    <div className="resnet-block-content">
      <h4>{data.label || "ResNet Block"}</h4>
      <div className="resnet-block-info">
        <div className="resnet-block-type">
          {data.blockType || "Basic"} Block
        </div>
        <p>
          <strong>Filters:</strong> {data.filters || 64}
        </p>
        <p>
          <strong>Stride:</strong>{" "}
          {Array.isArray(data.stride)
            ? data.stride.join("x")
            : data.stride || "1x1"}
        </p>
      </div>
      <div className="resnet-structure">
        {data.blockType === "Bottleneck" ? (
          <div className="bottleneck-structure">
            <div className="layer">Conv 1×1</div>
            <div className="layer">Conv 3×3</div>
            <div className="layer">Conv 1×1</div>
          </div>
        ) : (
          <div className="basic-structure">
            <div className="layer">Conv 3×3</div>
            <div className="layer">Conv 3×3</div>
          </div>
        )}
      </div>
      <div className="resnet-skip-connection">
        <div className="skip-line"></div>
        {data.stride &&
          Array.isArray(data.stride) &&
          (data.stride[0] > 1 || data.stride[1] > 1) && (
            <div className="projection-shortcut">1×1</div>
          )}
      </div>
    </div>
    <Handle
      type="source"
      position={Position.Right}
      style={sourceHandleStyle}
      id="source"
      isConnectable={true}
    />
  </div>
);

// Output Node - enhanced with special styling
const OutputNode = ({ data }: { data: any }) => {
  const outputSourceHandleStyle = {
    ...enhancedHandleStyle,
    background: "#E91E63", // Pink for output
    border: "3px solid #C2185B",
    zIndex: 10,
  };

  return (
    <div className="custom-node output-node">
      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Left}
        style={targetHandleStyle}
        id="input"
        isConnectable={true}
      />

      <div className="output-node-content">
        <h4>{data.label || "Output Layer"}</h4>
      </div>

      {/* Output handle with enhanced visibility */}
      <Handle
        type="source"
        position={Position.Right}
        style={outputSourceHandleStyle}
        id="output"
        isConnectable={true}
      />
    </div>
  );
};

// Add Layer Node
const AddLayerNode = ({ data }: { data: any }) => (
  <div className="custom-node add-layer-node">
    <Handle
      type="target"
      position={Position.Left}
      style={targetHandleStyle}
      isConnectable={true}
    />
    <h4>{data.label || "Add Layer"}</h4>
    <div className="add-layer-icon">
      <i className="fas fa-plus-circle"></i>
    </div>
    <Handle
      type="source"
      position={Position.Right}
      style={sourceHandleStyle}
      isConnectable={true}
    />
  </div>
);

// Activation Node
const ActivationNode = ({ data }: { data: any }) => {
  // Convert function name to lowercase and hyphenated for CSS
  const functionClass = (data.function || "relu")
    .toLowerCase()
    .replace(/\s+/g, "-");

  return (
    <div
      className={`custom-node activation-node ${functionClass}-node`}
      data-activation-type={data.function || "relu"}
    >
      <Handle
        type="target"
        position={Position.Left}
        style={targetHandleStyle}
        isConnectable={true}
      />
      <div className="simple-activation-content">{data.function || "relu"}</div>
      <Handle
        type="source"
        position={Position.Right}
        style={sourceHandleStyle}
        isConnectable={true}
      />
    </div>
  );
};

// Custom Block Node
const CustomBlockNode = ({ data }: { data: any }) => {
  const layers = data.layers || [];
  const blockName = data.blockName || "Custom Block";

  return (
    <div className="custom-node custom-block-node">
      <Handle
        type="target"
        position={Position.Left}
        style={targetHandleStyle}
        isConnectable={true}
      />
      <div className="custom-block-content">
        <h4>{blockName}</h4>
        <div className="custom-block-layers">
          <p>
            {layers.length} layer{layers.length !== 1 ? "s" : ""}
          </p>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Right}
        style={sourceHandleStyle}
        isConnectable={true}
      />
    </div>
  );
};
