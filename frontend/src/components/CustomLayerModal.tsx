import React, { useState } from "react";
import "../styles/components/CustomLayerModal.scss";
import ToastNotification, { ToastType } from "./ToastNotification";

interface LayerType {
  id: string;
  name: string;
  icon: string;
  type: string;
}

interface SelectedLayer {
  id: string;
  name: string;
  icon: string;
  type: string;
  order: number;
}

interface CustomLayerModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (blockName: string, layers: SelectedLayer[]) => void;
}

const CustomLayerModal: React.FC<CustomLayerModalProps> = ({
  isOpen,
  onClose,
  onSave,
}) => {
  const [blockName, setBlockName] = useState<string>("");
  const [selectedLayers, setSelectedLayers] = useState<SelectedLayer[]>([]);
  const [toast, setToast] = useState<{
    message: string;
    type?: ToastType;
  } | null>(null);
  const showToast = (message: string, type: ToastType = "info") =>
    setToast({ message, type });

  // Available layer types (excluding Input and Output as they're special)
  const availableLayers: LayerType[] = [
    { id: "dense", name: "Dense", icon: "fas fa-network-wired", type: "Dense" },
    {
      id: "convolution",
      name: "Convolution",
      icon: "fas fa-border-all",
      type: "Convolution",
    },
    {
      id: "maxpooling",
      name: "MaxPooling",
      icon: "fas fa-filter",
      type: "MaxPooling",
    },
    {
      id: "globalaveragepool",
      name: "Global Avg Pool",
      icon: "fas fa-compress",
      type: "GlobalAveragePool",
    },
    {
      id: "flatten",
      name: "Flatten",
      icon: "fas fa-compress-arrows-alt",
      type: "Flatten",
    },
    { id: "dropout", name: "Dropout", icon: "fas fa-random", type: "Dropout" },
    {
      id: "batchnormalization",
      name: "BatchNormalization",
      icon: "fas fa-balance-scale",
      type: "BatchNormalization",
    },
    {
      id: "attention",
      name: "Attention",
      icon: "fas fa-brain",
      type: "Attention",
    },
    {
      id: "activation",
      name: "Activation",
      icon: "fas fa-bolt",
      type: "Activation",
    },
    {
      id: "addlayer",
      name: "Add Layer",
      icon: "fas fa-plus-circle",
      type: "AddLayer",
    },
  ];

  const handleAddLayer = (layer: LayerType) => {
    const newLayer: SelectedLayer = {
      ...layer,
      order: selectedLayers.length,
    };
    setSelectedLayers([...selectedLayers, newLayer]);
  };

  const handleRemoveLayer = (index: number) => {
    const updatedLayers = selectedLayers.filter((_, i) => i !== index);
    // Update order numbers
    const reorderedLayers = updatedLayers.map((layer, i) => ({
      ...layer,
      order: i,
    }));
    setSelectedLayers(reorderedLayers);
  };

  const handleMoveLayer = (fromIndex: number, toIndex: number) => {
    if (toIndex < 0 || toIndex >= selectedLayers.length) return;

    const updatedLayers = [...selectedLayers];
    const [movedLayer] = updatedLayers.splice(fromIndex, 1);
    updatedLayers.splice(toIndex, 0, movedLayer);

    // Update order numbers
    const reorderedLayers = updatedLayers.map((layer, i) => ({
      ...layer,
      order: i,
    }));
    setSelectedLayers(reorderedLayers);
  };

  const handleSave = () => {
    if (!blockName.trim()) {
      showToast("Please enter a name for your custom block.", "error");
      return;
    }
    if (selectedLayers.length === 0) {
      showToast("Please add at least one layer to your custom block.", "error");
      return;
    }
    onSave(blockName.trim(), selectedLayers);
    handleReset();
  };

  const handleCancel = () => {
    handleReset();
    onClose();
  };

  const handleReset = () => {
    setBlockName("");
    setSelectedLayers([]);
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      {toast && (
        <ToastNotification
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
      <div className="modal-container custom-layer-modal">
        <div className="modal-header">
          <h3>Create Custom Layer</h3>
          <button className="close-modal-btn" onClick={handleCancel}>
            <i className="fas fa-times"></i>
          </button>
        </div>

        <div className="modal-body">
          <div className="custom-layer-content">
            {/* Left Column - Available Layers */}
            <div className="available-layers-column">
              <h4>Available Layers</h4>
              <div className="layer-list">
                {availableLayers.map((layer) => (
                  <div
                    key={layer.id}
                    className={`layer-item ${layer.id}-layer`}
                  >
                    <span>
                      <i className={layer.icon}></i>
                      {layer.name}
                    </span>
                    <button
                      className="add-button"
                      onClick={() => handleAddLayer(layer)}
                      title={`Add ${layer.name} layer`}
                    >
                      <i className="fas fa-plus"></i>
                    </button>
                  </div>
                ))}
              </div>
            </div>

            {/* Right Column - Custom Stack Builder */}
            <div className="custom-stack-column">
              <h4>Custom Layer Stack</h4>

              {/* Block Name Input */}
              <div className="block-name-input">
                <label htmlFor="blockName">Block Name:</label>
                <input
                  id="blockName"
                  type="text"
                  value={blockName}
                  onChange={(e) => setBlockName(e.target.value)}
                  placeholder="Enter custom block name"
                  maxLength={50}
                />
              </div>

              {/* Selected Layers */}
              <div className="selected-layers">
                {selectedLayers.length === 0 ? (
                  <div className="empty-stack">
                    <i className="fas fa-layer-group"></i>
                    <p>Add layers from the left to build your custom stack</p>
                  </div>
                ) : (
                  <div className="layers-stack">
                    {selectedLayers.map((layer, index) => (
                      <div
                        key={`${layer.id}-${index}`}
                        className="selected-layer-item"
                      >
                        <div className="layer-order">{index + 1}</div>
                        <div className="layer-content">
                          <i className={layer.icon}></i>
                          <span>{layer.name}</span>
                        </div>
                        <div className="layer-controls">
                          <button
                            className="move-btn"
                            onClick={() => handleMoveLayer(index, index - 1)}
                            disabled={index === 0}
                            title="Move up"
                          >
                            <i className="fas fa-chevron-up"></i>
                          </button>
                          <button
                            className="move-btn"
                            onClick={() => handleMoveLayer(index, index + 1)}
                            disabled={index === selectedLayers.length - 1}
                            title="Move down"
                          >
                            <i className="fas fa-chevron-down"></i>
                          </button>
                          <button
                            className="remove-btn"
                            onClick={() => handleRemoveLayer(index)}
                            title="Remove layer"
                          >
                            <i className="fas fa-trash"></i>
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="modal-footer">
          <button className="cancel-btn" onClick={handleCancel}>
            Cancel
          </button>
          <button className="save-btn" onClick={handleSave}>
            Save
          </button>
        </div>
      </div>
    </div>
  );
};

export default CustomLayerModal;
