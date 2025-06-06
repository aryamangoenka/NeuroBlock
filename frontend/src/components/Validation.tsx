import React, { useState, useEffect } from "react";
import axios from "axios";
import API_BASE_URL from "../utils/apiConfig";
import "../styles/components/Validation.scss";

interface ValidationProps {
  datasetName: string;
  isTrainingComplete: boolean;
}

interface FeatureInfo {
  feature_names: string[];
  feature_types: string[];
  class_labels: string[] | null;
}

const Validation: React.FC<ValidationProps> = ({
  datasetName,
  isTrainingComplete,
}) => {
  const [showModal, setShowModal] = useState(false);
  const [inputValues, setInputValues] = useState<{ [key: string]: string }>({});
  const [prediction, setPrediction] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [featureInfo, setFeatureInfo] = useState<FeatureInfo | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);

  const isImageDataset = datasetName === "MNIST" || datasetName === "CIFAR-10";

  // Fetch feature info from backend when modal opens or dataset changes
  useEffect(() => {
    if (showModal && datasetName) {
      fetchFeatureInfo();
    }
    // eslint-disable-next-line
  }, [showModal, datasetName]);

  const fetchFeatureInfo = async () => {
    try {
      const response = await axios.get(
        `${API_BASE_URL}/api/predict/features?dataset_name=${encodeURIComponent(
          datasetName
        )}`
      );
      setFeatureInfo(response.data);
      // Initialize input values
      const initialValues: Record<string, string> = {};
      response.data.feature_names.forEach((feature: string) => {
        initialValues[feature] = "";
      });
      setInputValues(initialValues);
    } catch (err) {
      setError("Failed to fetch feature information");
      setFeatureInfo(null);
      console.error("Error fetching feature info:", err);
    }
  };

  const handleInputChange = (key: string, value: string) => {
    setInputValues((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    setError(null);
    setPrediction(null);

    try {
      // Prepare input as numbers
      const numericInputs: { [key: string]: number } = {};
      (featureInfo?.feature_names || []).forEach((key: string) => {
        numericInputs[key] = parseFloat(inputValues[key]);
      });
      const response = await axios.post(`${API_BASE_URL}/api/predict`, {
        input_data: numericInputs,
        dataset_name: datasetName,
      });
      setPrediction(response.data);
    } catch (err: any) {
      setError(
        err.response?.data?.error ||
          "Failed to get prediction. Please try again."
      );
      console.error("Prediction error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setImageFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleImageSubmit = async () => {
    if (!imageFile) {
      setError("Please upload an image file.");
      return;
    }
    setIsLoading(true);
    setError(null);
    setPrediction(null);
    try {
      const formData = new FormData();
      formData.append("image", imageFile);
      formData.append("dataset_name", datasetName);
      const response = await axios.post(
        `${API_BASE_URL}/api/predict/image`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      setPrediction(response.data);
    } catch (err: any) {
      setError(
        err.response?.data?.error ||
          "Failed to get prediction. Please try again."
      );
      console.error("Prediction error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  // Helper to get label for prediction
  const getPredictionLabel = () => {
    const labels = featureInfo?.class_labels;
    if (
      labels &&
      typeof prediction?.prediction === "number" &&
      labels[prediction.prediction] !== undefined
    ) {
      return `${labels[prediction.prediction]} (${prediction.prediction})`;
    }
    return prediction?.prediction;
  };

  if (!isTrainingComplete) {
    return null;
  }

  return (
    <div className="validation-container">
      <button className="validate-button" onClick={() => setShowModal(true)}>
        <i className="fas fa-check-circle"></i>
        WANT TO TRY IT YOURSELF?
      </button>

      {showModal && (
        <div className="modal-overlay" onClick={() => setShowModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Custom Input Validation</h2>
              <button
                className="close-button"
                onClick={() => setShowModal(false)}
              >
                <i className="fas fa-times"></i>
              </button>
            </div>

            {isImageDataset ? (
              <>
                <div className="input-fields">
                  <label htmlFor="image-upload">Upload Image</label>
                  <input
                    type="file"
                    id="image-upload"
                    accept="image/*"
                    onChange={handleImageChange}
                  />
                  {imageFile && (
                    <div className="image-preview">
                      <img
                        src={URL.createObjectURL(imageFile)}
                        alt="Preview"
                        style={{ maxWidth: 200, maxHeight: 200, marginTop: 10 }}
                      />
                    </div>
                  )}
                </div>
                <div className="modal-actions">
                  <button
                    className="submit-button"
                    onClick={handleImageSubmit}
                    disabled={isLoading}
                  >
                    {isLoading ? (
                      <>
                        <i className="fas fa-spinner fa-spin"></i>
                        Predicting...
                      </>
                    ) : (
                      <>
                        <i className="fas fa-magic"></i>
                        Get Prediction
                      </>
                    )}
                  </button>
                </div>
              </>
            ) : featureInfo ? (
              <>
                <div className="input-fields">
                  {featureInfo.feature_names.map((key, idx) => (
                    <div key={key} className="input-group">
                      <label htmlFor={key}>
                        {key
                          .replace(/_/g, " ")
                          .replace(/\b\w/g, (l) => l.toUpperCase())}
                      </label>
                      <input
                        type={
                          featureInfo.feature_types[idx] === "int" ||
                          featureInfo.feature_types[idx] === "float"
                            ? "number"
                            : "text"
                        }
                        id={key}
                        value={inputValues[key] || ""}
                        onChange={(e) => handleInputChange(key, e.target.value)}
                        placeholder={`Enter ${key.replace(/_/g, " ")}`}
                        step="any"
                      />
                    </div>
                  ))}
                </div>

                <div className="modal-actions">
                  <button
                    className="submit-button"
                    onClick={handleSubmit}
                    disabled={isLoading}
                  >
                    {isLoading ? (
                      <>
                        <i className="fas fa-spinner fa-spin"></i>
                        Predicting...
                      </>
                    ) : (
                      <>
                        <i className="fas fa-magic"></i>
                        Get Prediction
                      </>
                    )}
                  </button>
                </div>
              </>
            ) : (
              <div className="error-message">
                <i className="fas fa-exclamation-circle"></i>
                {error || "Feature information not available."}
              </div>
            )}

            {error && featureInfo && (
              <div className="error-message">
                <i className="fas fa-exclamation-circle"></i>
                {error}
              </div>
            )}

            {prediction && (
              <div className="prediction-result">
                <h3>Prediction Results</h3>
                <div className="prediction-content">
                  <p className="prediction-value">
                    <strong>Prediction:</strong> {getPredictionLabel()}
                  </p>
                  {prediction.confidence !== undefined &&
                    prediction.confidence !== null && (
                      <p className="confidence-value">
                        <strong>Confidence:</strong>{" "}
                        {(prediction.confidence * 100).toFixed(2)}%
                      </p>
                    )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Validation;
