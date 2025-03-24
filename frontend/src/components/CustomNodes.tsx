import { Handle, Position } from "reactflow";
import "../styles/components/CustomNodes.scss";

// Dense Node
export const DenseNode = ({ data }: { data: any }) => (
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
export const ConvolutionNode = ({ data }: { data: any }) => (
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
export const MaxPoolingNode = ({ data }: { data: any }) => (
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

// Flatten Node
export const FlattenNode = ({ data }: { data: any }) => (
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
export const DropoutNode = ({ data }: { data: any }) => (
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
export const BatchNormalizationNode = ({ data }: { data: any }) => (
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
export const AttentionNode = ({ data }: { data: any }) => (
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
export const InputNode = ({ data }: { data: any }) => (
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
export const ResNetBlockNode = ({ data }: { data: any }) => (
  <div
    className={`custom-node resnetblock-node ${
      data.blockType?.toLowerCase() || "basic"
    }-block ${data.useSkipConnection === false ? "no-skip-connection" : ""}`}
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
          <strong>Channels:</strong> {data.inChannels || "?"} →{" "}
          {data.outChannels || "?"}
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
        {data.downsampleType && data.downsampleType !== "None" && (
          <p>
            <strong>Downsample:</strong> {data.downsampleType}
          </p>
        )}
      </div>
      <div className="resnet-skip-connection">
        <div className="skip-line"></div>
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
export const OutputNode = ({ data }: { data: any }) => (
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
