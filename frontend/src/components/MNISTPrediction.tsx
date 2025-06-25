import React, { useState } from "react";
import DrawingCanvas from "./DrawingCanvas";
import API_BASE_URL from "../utils/apiConfig";

interface MNISTPredictionProps {
  onPredictionStart?: () => void;
  onPredictionComplete?: (result: {
    prediction: number;
    confidence: number;
  }) => void;
  isTrainingComplete: boolean;
}

const MNISTPrediction: React.FC<MNISTPredictionProps> = ({
  onPredictionStart,
  onPredictionComplete,
  isTrainingComplete,
}) => {
  const [prediction, setPrediction] = useState<number | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [clearCanvas, setClearCanvas] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDrawingComplete = async (imageData: Blob) => {
    if (!isTrainingComplete) {
      setError(
        "Please wait for training to complete before making predictions."
      );
      return;
    }

    setIsLoading(true);
    setError(null);
    onPredictionStart?.();

    try {
      const formData = new FormData();
      formData.append("image", imageData, "drawing.png");
      formData.append("dataset_name", "MNIST");

      const response = await fetch(`${API_BASE_URL}/api/predict/image`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Prediction failed: ${response.statusText}`);
      }

      const result = await response.json();

      if (result.error) {
        throw new Error(result.error);
      }

      setPrediction(result.prediction);
      setConfidence(result.confidence);
      onPredictionComplete?.(result);
    } catch (err) {
      console.error("Error making prediction:", err);
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setPrediction(null);
    setConfidence(null);
    setError(null);
    setClearCanvas(true);
    // Reset clearCanvas after a short delay to allow for next drawing
    setTimeout(() => setClearCanvas(false), 100);
  };

  return (
    <div className="mnist-prediction-container">
      <div className="drawing-section">
        <h4>Draw a digit (0-9)</h4>
        <DrawingCanvas
          onDrawingComplete={handleDrawingComplete}
          width={280}
          height={200}
          lineWidth={20}
          clearCanvas={clearCanvas}
        />
        <div className="drawing-controls">
          <button
            className="clear-button"
            onClick={handleClear}
            disabled={isLoading}
          >
            <i className="fas fa-eraser"></i>
            Clear
          </button>
        </div>
      </div>

      <div className="prediction-section">
        {isLoading ? (
          <div className="loading-state">
            <i className="fas fa-spinner fa-spin"></i>
            Predicting...
          </div>
        ) : error ? (
          <div className="error-state">{error}</div>
        ) : prediction !== null && confidence !== null ? (
          <div className="prediction-result">
            <div className="predicted-digit">
              <span>Predicted Digit:</span>
              <strong>{prediction}</strong>
            </div>
            <div className="confidence">
              <span>Confidence:</span>
              <strong>{Math.round(confidence * 100)}%</strong>
            </div>
          </div>
        ) : (
          <div className="empty-state">
            Draw a digit above to get a prediction
          </div>
        )}
      </div>
    </div>
  );
};

export default MNISTPrediction;
