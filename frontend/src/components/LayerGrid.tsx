import React from "react";
import TutorialLayerCard from "./TutorialLayerCard";

interface LayerData {
  type: string;
  name: string;
  color: string;
  description: string;
  properties?: { name: string; value: string }[];
}

interface LayerGridProps {
  layers: LayerData[];
}

const LayerGrid: React.FC<LayerGridProps> = ({ layers }) => {
  return (
    <div className="layer-grid">
      {layers.map((layer, index) => (
        <TutorialLayerCard
          key={index}
          type={layer.type}
          name={layer.name}
          color={layer.color}
          description={layer.description}
          properties={layer.properties}
        />
      ))}
    </div>
  );
};

export default LayerGrid;

// Predefined layer colors and data for easy use
export const layerColors = {
  input: "#00BCD4", // Cyan
  dense: "#4CAF50", // Green
  conv2d: "#2196F3", // Blue
  maxPooling: "#9C27B0", // Purple
  flatten: "#FF9800", // Orange
  dropout: "#F44336", // Red
  batchNorm: "#188cc6", // Gray-Blue
  attention: "#673AB7", // Deep Purple
  output: "#FF5722", // Deep Orange
  lstm: "#009688", // Teal
  embedding: "#795548", // Brown
};

export const commonLayers: LayerData[] = [
  {
    type: "input",
    name: "Input Layer",
    color: layerColors.input,
    description:
      "The entry point for data into your neural network. Defines the shape of your input data.",
    properties: [{ name: "Shape", value: "(28, 28, 1)" }],
  },
  {
    type: "dense",
    name: "Dense Layer",
    color: layerColors.dense,
    description:
      "Fully connected layer where each neuron is connected to every neuron in the previous layer.",
    properties: [
      { name: "Units", value: "128" },
      { name: "Activation", value: "relu" },
    ],
  },
  {
    type: "conv2d",
    name: "Conv2D Layer",
    color: layerColors.conv2d,
    description:
      "Applies filters to detect spatial patterns in input data. Essential for image processing tasks.",
    properties: [
      { name: "Filters", value: "32" },
      { name: "Kernel", value: "3×3" },
    ],
  },
  {
    type: "maxPooling",
    name: "MaxPooling2D",
    color: layerColors.maxPooling,
    description:
      "Reduces the spatial dimensions by taking the maximum value in each window.",
    properties: [{ name: "Pool Size", value: "2×2" }],
  },
  {
    type: "dropout",
    name: "Dropout Layer",
    color: layerColors.dropout,
    description:
      "Randomly sets a fraction of inputs to zero during training to prevent overfitting.",
    properties: [{ name: "Rate", value: "0.25" }],
  },
  {
    type: "output",
    name: "Output Layer",
    color: layerColors.output,
    description: "The final layer that produces the model's predictions.",
    properties: [
      { name: "Units", value: "10" },
      { name: "Activation", value: "softmax" },
    ],
  },
];
