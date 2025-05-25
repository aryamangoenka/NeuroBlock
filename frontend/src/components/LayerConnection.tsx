import React from "react";
import TutorialLayerCard from "./TutorialLayerCard";

interface LayerConnectionProps {
  inputLayer: {
    type: string;
    name: string;
    color: string;
    properties?: { name: string; value: string }[];
  };
  outputLayer: {
    type: string;
    name: string;
    color: string;
    properties?: { name: string; value: string }[];
  };
}

const LayerConnection: React.FC<LayerConnectionProps> = ({
  inputLayer,
  outputLayer,
}) => {
  return (
    <div className="layer-connection">
      <div className="connection-node">
        <TutorialLayerCard
          type={inputLayer.type}
          name={inputLayer.name}
          color={inputLayer.color}
          properties={inputLayer.properties}
        />
      </div>
      <div className="connection-line"></div>
      <div className="connection-node">
        <TutorialLayerCard
          type={outputLayer.type}
          name={outputLayer.name}
          color={outputLayer.color}
          properties={outputLayer.properties}
        />
      </div>
    </div>
  );
};

export default LayerConnection;
