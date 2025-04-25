import { Handle, Position } from "reactflow";
import "../styles/components/CustomNodes.scss";

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
};

// Dense Node
const DenseNode = ({ data }: { data: any }) => (
  <div className="custom-node dense-node">
    <Handle
      type="target"
      position={Position.Left}
      style={{ background: "#555" }}
    />
    <h4>{data.label || "Dense Layer"}</h4>
    <p>Neurons: {data.neurons}</p>
    <p>Activation: {data.activation}</p>
    <Handle
      type="source"
      position={Position.Right}
      style={{ background: "#555" }}
    />
  </div>
);

// Convolutional Node
const ConvolutionNode = ({ data }: { data: any }) => (
  <div className="custom-node convolution-node">
    <Handle
      type="target"
      position={Position.Left}
      style={{ background: "#555" }}
    />
    <h4>{data.label || "Convolution Layer"}</h4>
    <p>Filters: {data.filters}</p>
    <p>Kernel Size: {data.kernelSize?.join("x")}</p>
    <p>Stride: {data.stride?.join("x")}</p>
    <p>Activation: {data.activation}</p>
    <Handle
      type="source"
      position={Position.Right}
      style={{ background: "#555" }}
    />
  </div>
);

// MaxPooling Node
const MaxPoolingNode = ({ data }: { data: any }) => (
  <div className="custom-node maxpooling-node">
    <Handle
      type="target"
      position={Position.Left}
      style={{ background: "#555" }}
    />
    <h4>{data.label || "MaxPooling Layer"}</h4>
    <p>Pool Size: {data.poolSize?.join("x")}</p>
    <p>Stride: {data.stride?.join("x")}</p>
    <p>Padding: {data.padding}</p>
    <Handle
      type="source"
      position={Position.Right}
      style={{ background: "#555" }}
    />
  </div>
);

// GlobalAveragePooling Node
const GlobalAveragePoolNode = ({ data }: { data: any }) => (
  <div className="custom-node globalaveragepool-node">
    <Handle
      type="target"
      position={Position.Left}
      style={{ background: "#555" }}
    />
    <h4>{data.label || "GlobalAveragePooling Layer"}</h4>
    <Handle
      type="source"
      position={Position.Right}
      style={{ background: "#555" }}
    />
  </div>
);

// Flatten Node
const FlattenNode = ({ data }: { data: any }) => (
  <div className="custom-node flatten-node">
    <Handle
      type="target"
      position={Position.Left}
      style={{ background: "#555" }}
    />
    <h4>{data.label || "Flatten Layer"}</h4>
    <Handle
      type="source"
      position={Position.Right}
      style={{ background: "#555" }}
    />
  </div>
);

// Dropout Node
const DropoutNode = ({ data }: { data: any }) => (
  <div className="custom-node dropout-node">
    <Handle
      type="target"
      position={Position.Left}
      style={{ background: "#555" }}
    />
    <h4>{data.label || "Dropout Layer"}</h4>
    <p>Rate: {data.rate}</p>
    <Handle
      type="source"
      position={Position.Right}
      style={{ background: "#555" }}
    />
  </div>
);

// Batch Normalization Node
const BatchNormalizationNode = ({ data }: { data: any }) => (
  <div className="custom-node batchnormalization-node">
    <Handle
      type="target"
      position={Position.Left}
      style={{ background: "#555" }}
    />
    <h4>{data.label || "Batch Normalization Layer"}</h4>
    <p>Momentum: {data.momentum}</p>
    <p>Epsilon: {data.epsilon}</p>
    <Handle
      type="source"
      position={Position.Right}
      style={{ background: "#555" }}
    />
  </div>
);

// Attention Node
const AttentionNode = ({ data }: { data: any }) => (
  <div className="custom-node attention-node">
    <Handle
      type="target"
      position={Position.Left}
      style={{ background: "#555" }}
    />
    <h4>{data.label || "Attention Layer"}</h4>
    <p>Heads: {data.heads}</p>
    <p>Key Dim: {data.keyDim}</p>
    <p>Dropout: {data.dropout}</p>
    <Handle
      type="source"
      position={Position.Right}
      style={{ background: "#555" }}
    />
  </div>
);

// Input Node
const InputNode = ({ data }: { data: any }) => (
  <div className="custom-node input-node">
    <h4>{data.label || "Input Layer"}</h4>
    <Handle
      type="source"
      position={Position.Right}
      style={{ background: "#555" }}
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
      style={{ background: "#555" }}
      id="target"
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
        <p>
          <strong>Activation:</strong> {data.activation || "ReLU"}
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
      style={{ background: "#555" }}
      id="source"
    />
  </div>
);

// Output Node
const OutputNode = ({ data }: { data: any }) => (
  <div className="custom-node output-node">
    <Handle
      type="target"
      position={Position.Left}
      style={{ background: "#555" }}
    />
    <h4>{data.label || "Output Layer"}</h4>
    <p>Activation: {data.activation}</p>
  </div>
);

// Add Layer Node
const AddLayerNode = ({ data }: { data: any }) => (
  <div className="custom-node add-layer-node">
    <Handle
      type="target"
      position={Position.Left}
      style={{ background: "#555" }}
    />
    <h4>{data.label || "Add Layer"}</h4>
    <div className="add-layer-icon">
      <i className="fas fa-plus-circle"></i>
    </div>
    <Handle
      type="source"
      position={Position.Right}
      style={{ background: "#555" }}
    />
  </div>
);
