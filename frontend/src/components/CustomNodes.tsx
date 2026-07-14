import { Handle, Position } from "reactflow";
import "../styles/components/CustomNodes.scss";

/**
 * Paper Lab node system (see DESIGN.md).
 * Every node is a white module with a colored left rail:
 *   filled rail  = layer with trainable parameters ("it learns")
 *   hollow rail  = transform-only layer
 */

type NodeVisual = {
  hue: string; // CSS var for the layer hue
  learns: boolean; // filled vs hollow rail
};

const VISUALS: Record<string, NodeVisual> = {
  input: { hue: "var(--n-input)", learns: false },
  conv: { hue: "var(--n-conv)", learns: true },
  pool: { hue: "var(--n-pool)", learns: false },
  flatten: { hue: "var(--n-flat)", learns: false },
  dense: { hue: "var(--n-dense)", learns: true },
  dropout: { hue: "var(--n-drop)", learns: false },
  activation: { hue: "var(--n-act)", learns: false },
  batchnorm: { hue: "var(--n-bn)", learns: true },
  attention: { hue: "var(--n-attn)", learns: true },
  output: { hue: "var(--n-out)", learns: true },
  custom: { hue: "var(--n-input)", learns: false },
};

const handleClass = "nb-handle";

const NodeShell = ({
  visual,
  title,
  params,
  hasInput = true,
  hasOutput = true,
  inputId,
  outputId,
  extraClass = "",
  children,
}: {
  visual: NodeVisual;
  title: string;
  params?: (string | undefined | false)[];
  hasInput?: boolean;
  hasOutput?: boolean;
  inputId?: string;
  outputId?: string;
  extraClass?: string;
  children?: React.ReactNode;
}) => (
  <div
    className={`nb-node ${visual.learns ? "learns" : "transforms"} ${extraClass}`}
    style={{ "--rc": visual.hue } as React.CSSProperties}
  >
    {hasInput && (
      <Handle
        type="target"
        position={Position.Left}
        className={handleClass}
        id={inputId}
        isConnectable={true}
      />
    )}
    <div className="nb-node-body">
      <h4>{title}</h4>
      {params && params.filter(Boolean).length > 0 && (
        <p className="nb-params">{params.filter(Boolean).join(" · ")}</p>
      )}
      {children}
    </div>
    {visual.learns && <span className="nb-learns-tag">learns</span>}
    {hasOutput && (
      <Handle
        type="source"
        position={Position.Right}
        className={handleClass}
        id={outputId}
        isConnectable={true}
      />
    )}
  </div>
);

// ===== Node components (names, data fields, and handle ids preserved) =====

const InputNode = ({ data }: { data: any }) => (
  <NodeShell
    visual={VISUALS.input}
    title={data.label || "Input"}
    hasInput={false}
  />
);

const DenseNode = ({ data }: { data: any }) => (
  <NodeShell
    visual={VISUALS.dense}
    title={data.label || "Dense"}
    params={[data.neurons != null && `units ${data.neurons}`]}
  />
);

const ConvolutionNode = ({ data }: { data: any }) => (
  <NodeShell
    visual={VISUALS.conv}
    title={data.label || "Conv2D"}
    params={[
      data.filters != null && `${data.filters} filters`,
      data.kernelSize && `${data.kernelSize.join("×")}`,
      data.padding && `${data.padding}`,
    ]}
  />
);

const MaxPoolingNode = ({ data }: { data: any }) => (
  <NodeShell
    visual={VISUALS.pool}
    title={data.label || "MaxPooling"}
    params={[
      data.poolSize && `pool ${data.poolSize.join("×")}`,
      data.stride && `stride ${data.stride.join("×")}`,
    ]}
  />
);

const GlobalAveragePoolNode = ({ data }: { data: any }) => (
  <NodeShell
    visual={VISUALS.pool}
    title={data.label || "Global Avg Pool"}
    params={["spatial → 1×1"]}
  />
);

const FlattenNode = ({ data }: { data: any }) => (
  <NodeShell
    visual={VISUALS.flatten}
    title={data.label || "Flatten"}
    params={["2D → 1D"]}
  />
);

const DropoutNode = ({ data }: { data: any }) => (
  <NodeShell
    visual={VISUALS.dropout}
    title={data.label || "Dropout"}
    params={[data.rate != null && `rate ${data.rate}`]}
  />
);

const BatchNormalizationNode = ({ data }: { data: any }) => (
  <NodeShell
    visual={VISUALS.batchnorm}
    title={data.label || "BatchNorm"}
    params={[
      data.momentum != null && `momentum ${data.momentum}`,
      data.epsilon != null && `ε ${data.epsilon}`,
    ]}
  />
);

const AttentionNode = ({ data }: { data: any }) => (
  <NodeShell
    visual={VISUALS.attention}
    title={data.label || "Attention"}
    params={[
      data.heads != null && `${data.heads} heads`,
      data.keyDim != null && `key ${data.keyDim}`,
      data.dropout != null && `drop ${data.dropout}`,
    ]}
  />
);

const OutputNode = ({ data }: { data: any }) => (
  <NodeShell
    visual={VISUALS.output}
    title={data.label || "Output"}
    inputId="input"
    outputId="output"
  />
);

const ActivationNode = ({ data }: { data: any }) => {
  const fn = data.function || "relu";
  return (
    <NodeShell
      visual={VISUALS.activation}
      title={fn}
      extraClass="nb-activation"
    />
  );
};

const AddLayerNode = ({ data }: { data: any }) => (
  <NodeShell
    visual={VISUALS.custom}
    title={data.label || "Add"}
    params={["merge branches"]}
  />
);

const ResNetBlockNode = ({ data }: { data: any }) => (
  <NodeShell
    visual={VISUALS.conv}
    title={data.label || "ResNet Block"}
    params={[
      `${data.blockType || "Basic"}`,
      data.filters != null && `${data.filters || 64} filters`,
      Array.isArray(data.stride) && `stride ${data.stride.join("×")}`,
    ]}
    inputId="target"
    outputId="source"
    extraClass="nb-resnet"
  >
    <div className="nb-resnet-structure">
      {data.blockType === "Bottleneck" ? (
        <>
          <span>1×1</span>
          <span>3×3</span>
          <span>1×1</span>
        </>
      ) : (
        <>
          <span>3×3</span>
          <span>3×3</span>
        </>
      )}
      <i className="nb-skip" title="skip connection" />
    </div>
  </NodeShell>
);

const CustomBlockNode = ({ data }: { data: any }) => {
  const layers = data.layers || [];
  return (
    <NodeShell
      visual={VISUALS.custom}
      title={data.blockName || "Custom Block"}
      params={[`${layers.length} layer${layers.length !== 1 ? "s" : ""}`]}
      extraClass="nb-customblock"
    />
  );
};

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
