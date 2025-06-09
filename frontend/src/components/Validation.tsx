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
  categorical_options?: { [key: string]: string[] | number[] };
  quality_warning?: {
    message: string;
    issues: string[];
    recommendations: string[];
  };
  data_quality?: {
    quality_score: number;
    quality_level: string;
    warnings: string[];
    recommendations: string[];
    analysis: {
      n_samples: number;
      n_features: number;
      task_type: string;
    };
  };
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
    const fetchFeatureInfo = async (dataset: string) => {
      try {
        const encodedDatasetName = encodeURIComponent(dataset);
        const url = `${API_BASE_URL}/api/predict/features?dataset_name=${encodedDatasetName}`;

        const response = await axios.get(url, {
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          withCredentials: true,
        });

        setFeatureInfo(response.data);
        // Initialize input values
        const initialValues: Record<string, string> = {};
        response.data.feature_names.forEach((feature: string) => {
          initialValues[feature] = "";
        });
        setInputValues(initialValues);
      } catch (err: any) {
        setError("Failed to fetch feature information");
        setFeatureInfo(null);
        console.error("Error fetching feature info:", err);
      }
    };

    if (showModal && datasetName) {
      if (isImageDataset) {
        // For image datasets, skip API call and set empty feature info
        setFeatureInfo({
          feature_names: [],
          feature_types: [],
          class_labels: null,
        });
        setInputValues({});
      } else {
        fetchFeatureInfo(datasetName);
      }
    }
  }, [showModal, datasetName, isImageDataset]);

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
      // Prepare input data based on feature types
      const processedInputs: { [key: string]: number | string } = {};
      (featureInfo?.feature_names || []).forEach((key: string, idx: number) => {
        const featureType = featureInfo?.feature_types[idx];
        if (featureType === "int" || featureType === "float") {
          processedInputs[key] = parseFloat(inputValues[key]);
        } else {
          // For categorical or other types, keep as string
          processedInputs[key] = inputValues[key];
        }
      });
      const response = await axios.post(
        `${API_BASE_URL}/api/predict`,
        {
          input_data: processedInputs,
          dataset_name: datasetName,
        },
        {
          withCredentials: true,
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
          withCredentials: true,
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
    // Use prediction_label if available (from new generic system)
    if (prediction?.prediction_label) {
      return `${prediction.prediction_label} (${prediction.prediction})`;
    }

    // Fallback to class_labels lookup for built-in datasets
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
                {/* Enhanced Data Quality Display */}
                {featureInfo.data_quality && (
                  <div
                    className="data-quality-assessment"
                    style={{
                      backgroundColor:
                        featureInfo.data_quality.quality_level === "excellent"
                          ? "#d4edda"
                          : featureInfo.data_quality.quality_level === "good"
                          ? "#d1ecf1"
                          : featureInfo.data_quality.quality_level === "fair"
                          ? "#fff3cd"
                          : featureInfo.data_quality.quality_level === "poor"
                          ? "#f8d7da"
                          : "#f8d7da",
                      border: `1px solid ${
                        featureInfo.data_quality.quality_level === "excellent"
                          ? "#c3e6cb"
                          : featureInfo.data_quality.quality_level === "good"
                          ? "#bee5eb"
                          : featureInfo.data_quality.quality_level === "fair"
                          ? "#ffeaa7"
                          : featureInfo.data_quality.quality_level === "poor"
                          ? "#f5c6cb"
                          : "#f5c6cb"
                      }`,
                      borderRadius: "4px",
                      padding: "12px",
                      margin: "10px 0",
                      fontSize: "14px",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        marginBottom: "8px",
                      }}
                    >
                      <i
                        className={`fas ${
                          featureInfo.data_quality.quality_level === "excellent"
                            ? "fa-check-circle"
                            : featureInfo.data_quality.quality_level === "good"
                            ? "fa-thumbs-up"
                            : featureInfo.data_quality.quality_level === "fair"
                            ? "fa-exclamation-triangle"
                            : "fa-exclamation-circle"
                        }`}
                        style={{
                          color:
                            featureInfo.data_quality.quality_level ===
                            "excellent"
                              ? "#155724"
                              : featureInfo.data_quality.quality_level ===
                                "good"
                              ? "#0c5460"
                              : featureInfo.data_quality.quality_level ===
                                "fair"
                              ? "#856404"
                              : "#721c24",
                          marginRight: "8px",
                        }}
                      ></i>
                      <strong
                        style={{
                          color:
                            featureInfo.data_quality.quality_level ===
                            "excellent"
                              ? "#155724"
                              : featureInfo.data_quality.quality_level ===
                                "good"
                              ? "#0c5460"
                              : featureInfo.data_quality.quality_level ===
                                "fair"
                              ? "#856404"
                              : "#721c24",
                        }}
                      >
                        Data Quality:{" "}
                        {featureInfo.data_quality.quality_level.toUpperCase()}(
                        {featureInfo.data_quality.quality_score}/100)
                      </strong>
                    </div>
                    <div
                      style={{
                        marginLeft: "24px",
                        fontSize: "12px",
                        opacity: 0.8,
                      }}
                    >
                      {featureInfo.data_quality.analysis.n_samples} samples,{" "}
                      {featureInfo.data_quality.analysis.n_features} features
                    </div>
                    {featureInfo.data_quality.warnings.length > 0 && (
                      <div style={{ marginTop: "8px" }}>
                        <strong style={{ fontSize: "12px" }}>Warnings:</strong>
                        {featureInfo.data_quality.warnings.map(
                          (warning, idx) => (
                            <div
                              key={idx}
                              style={{
                                fontSize: "11px",
                                marginLeft: "16px",
                                marginTop: "2px",
                              }}
                            >
                              • {warning}
                            </div>
                          )
                        )}
                      </div>
                    )}
                    {featureInfo.data_quality.recommendations.length > 0 && (
                      <div style={{ marginTop: "8px" }}>
                        <strong style={{ fontSize: "12px" }}>
                          Recommendations:
                        </strong>
                        {featureInfo.data_quality.recommendations.map(
                          (rec, idx) => (
                            <div
                              key={idx}
                              style={{
                                fontSize: "11px",
                                marginLeft: "16px",
                                marginTop: "2px",
                              }}
                            >
                              • {rec}
                            </div>
                          )
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* Legacy Quality Warning Display (for backward compatibility) */}
                {featureInfo.quality_warning && !featureInfo.data_quality && (
                  <div
                    className="quality-warning"
                    style={{
                      backgroundColor: "#fff3cd",
                      border: "1px solid #ffeaa7",
                      borderRadius: "4px",
                      padding: "12px",
                      margin: "10px 0",
                      fontSize: "14px",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        marginBottom: "8px",
                      }}
                    >
                      <i
                        className="fas fa-exclamation-triangle"
                        style={{ color: "#856404", marginRight: "8px" }}
                      ></i>
                      <strong style={{ color: "#856404" }}>
                        {featureInfo.quality_warning.message}
                      </strong>
                    </div>
                    {featureInfo.quality_warning.recommendations.map(
                      (rec, idx) => (
                        <div
                          key={idx}
                          style={{ color: "#856404", marginLeft: "24px" }}
                        >
                          • {rec}
                        </div>
                      )
                    )}
                  </div>
                )}
                <div className="input-fields">
                  {featureInfo.feature_names.map((key, idx) => (
                    <div key={key} className="input-group">
                      <label htmlFor={key}>
                        {key
                          .replace(/_/g, " ")
                          .replace(/\b\w/g, (l) => l.toUpperCase())}
                      </label>
                      {featureInfo.feature_types[idx] === "categorical" &&
                      (featureInfo.categorical_options?.[key] ||
                        featureInfo.class_labels) ? (
                        <select
                          id={key}
                          value={inputValues[key] || ""}
                          onChange={(e) =>
                            handleInputChange(key, e.target.value)
                          }
                        >
                          <option value="">
                            Select {key.replace(/_/g, " ")}
                          </option>
                          {/* Use feature-specific categorical options if available, otherwise fall back to class_labels */}
                          {(
                            featureInfo.categorical_options?.[key] ||
                            featureInfo.class_labels ||
                            []
                          ).map((option) => (
                            <option key={option} value={option}>
                              {option}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <input
                          type={
                            featureInfo.feature_types[idx] === "int" ||
                            featureInfo.feature_types[idx] === "float"
                              ? "number"
                              : "text"
                          }
                          id={key}
                          value={inputValues[key] || ""}
                          onChange={(e) =>
                            handleInputChange(key, e.target.value)
                          }
                          placeholder={`Enter ${key.replace(/_/g, " ")}`}
                          step="any"
                        />
                      )}
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

                  {/* Prediction explanation */}
                  <div
                    className="prediction-explanation"
                    style={{
                      backgroundColor: "#f8f9fa",
                      border: "1px solid #e9ecef",
                      borderRadius: "4px",
                      padding: "10px",
                      margin: "10px 0",
                      fontSize: "13px",
                      color: "#495057",
                    }}
                  >
                    <i
                      className="fas fa-info-circle"
                      style={{ marginRight: "6px", color: "#17a2b8" }}
                    ></i>
                    {featureInfo?.data_quality?.analysis?.task_type ===
                    "regression" ? (
                      <>
                        <strong>Regression Prediction:</strong> This model
                        predicts a numerical value. The prediction represents
                        the estimated continuous output based on the input
                        features you provided.
                      </>
                    ) : (
                      <>
                        <strong>Classification Prediction:</strong> This model
                        predicts which category your input belongs to.
                        {featureInfo?.class_labels
                          ? `The model chose from ${featureInfo.class_labels.length} possible classes.`
                          : "The model determines the most likely class for your input."}
                      </>
                    )}
                  </div>

                  {prediction.confidence !== undefined &&
                    prediction.confidence !== null && (
                      <p className="confidence-value">
                        <strong>Confidence:</strong>{" "}
                        {(prediction.confidence * 100).toFixed(2)}%
                      </p>
                    )}
                  {prediction.class_probabilities && (
                    <div className="class-probabilities">
                      <strong>Class Probabilities:</strong>
                      <ul>
                        {prediction.class_probabilities.map(
                          (prob: number, idx: number) => (
                            <li key={idx}>
                              {featureInfo?.class_labels &&
                              featureInfo.class_labels[idx]
                                ? `${featureInfo.class_labels[idx]}: ${(
                                    prob * 100
                                  ).toFixed(2)}%`
                                : `Class ${idx}: ${(prob * 100).toFixed(2)}%`}
                            </li>
                          )
                        )}
                      </ul>
                    </div>
                  )}
                  {prediction.error && (
                    <div className="error-message">
                      <i className="fas fa-exclamation-triangle"></i>
                      {prediction.error}
                    </div>
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
