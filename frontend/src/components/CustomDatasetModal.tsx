import React, { useState, useRef, useCallback } from "react";
import {
  previewDataset,
  validateDatasetConfig,
  createCustomDataset,
  CustomDatasetApiError,
  DatasetPreview,
  ValidationResult,
  DatasetConfig,
  formatFileSize,
  getDataTypeColor,
} from "../utils/customDatasetApi";
import "../styles/components/CustomDatasetModal.scss";

interface CustomDatasetModalProps {
  isOpen: boolean;
  onClose: () => void;
  onDatasetCreated: (datasetName: string) => void;
}

type WizardStep = 1 | 2 | 3;

interface FormState {
  file: File | null;
  preview: DatasetPreview | null;
  config: Partial<DatasetConfig>;
  validation: ValidationResult | null;
  datasetName: string;
}

const CustomDatasetModal: React.FC<CustomDatasetModalProps> = ({
  isOpen,
  onClose,
  onDatasetCreated,
}) => {
  const [currentStep, setCurrentStep] = useState<WizardStep>(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [formState, setFormState] = useState<FormState>({
    file: null,
    preview: null,
    config: {
      missing_value_strategy: "drop",
    },
    validation: null,
    datasetName: "",
  });

  const resetForm = useCallback(() => {
    setCurrentStep(1);
    setIsLoading(false);
    setError(null);
    setIsDragOver(false);
    setFormState({
      file: null,
      preview: null,
      config: { missing_value_strategy: "drop" },
      validation: null,
      datasetName: "",
    });
  }, []);

  const handleClose = useCallback(() => {
    resetForm();
    onClose();
  }, [resetForm, onClose]);

  // Step 1: File Upload and Preview
  const handleFileSelect = async (file: File) => {
    if (!file) return;

    // Validate file type
    const validTypes = [".csv", ".xlsx", ".xls"];
    const fileExtension = file.name
      .toLowerCase()
      .substring(file.name.lastIndexOf("."));
    if (!validTypes.includes(fileExtension)) {
      setError("Please select a CSV or Excel file (.csv, .xlsx, .xls)");
      return;
    }

    // Validate file size (50MB limit)
    if (file.size > 50 * 1024 * 1024) {
      setError("File size must be less than 50MB");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const preview = await previewDataset(file);
      setFormState((prev) => ({
        ...prev,
        file,
        preview,
        datasetName: file.name.replace(/\.[^/.]+$/, ""), // Remove extension
      }));
    } catch (err) {
      if (err instanceof CustomDatasetApiError) {
        setError(err.message);
      } else {
        setError("Failed to preview dataset. Please try again.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  // Step 2: Configuration
  const handleConfigChange = async (updates: Partial<DatasetConfig>) => {
    const newConfig = { ...formState.config, ...updates };
    setFormState((prev) => ({ ...prev, config: newConfig }));

    // Validate configuration if we have enough info
    if (
      newConfig.target_column &&
      newConfig.feature_columns &&
      newConfig.task_type &&
      formState.preview
    ) {
      try {
        const validation = await validateDatasetConfig(
          newConfig as DatasetConfig,
          formState.preview
        );
        setFormState((prev) => ({ ...prev, validation }));
      } catch (err) {
        console.error("Validation error:", err);
      }
    }
  };

  const handleFeatureColumnToggle = (column: string) => {
    const currentFeatures = formState.config.feature_columns || [];
    const newFeatures = currentFeatures.includes(column)
      ? currentFeatures.filter((c) => c !== column)
      : [...currentFeatures, column];

    handleConfigChange({ feature_columns: newFeatures });
  };

  // Step 3: Review and Create
  const handleCreateDataset = async () => {
    if (!formState.file || !formState.config || !formState.datasetName) {
      setError("Missing required information");
      return;
    }

    const config: DatasetConfig = {
      dataset_name: formState.datasetName,
      target_column: formState.config.target_column!,
      feature_columns: formState.config.feature_columns!,
      task_type: formState.config.task_type!,
      missing_value_strategy: formState.config.missing_value_strategy,
    };

    setIsLoading(true);
    setError(null);

    try {
      const result = await createCustomDataset(formState.file, config);
      if (result.success) {
        onDatasetCreated(result.dataset.name);
        handleClose();
      } else {
        setError("Failed to create dataset");
      }
    } catch (err) {
      if (err instanceof CustomDatasetApiError) {
        setError(err.message);
      } else {
        setError("Failed to create dataset. Please try again.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const canProceedToStep2 = !!(formState.file && formState.preview);
  const canProceedToStep3 = !!(
    canProceedToStep2 &&
    formState.config.target_column &&
    formState.config.feature_columns &&
    formState.config.feature_columns.length > 0 &&
    formState.config.task_type
  );

  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      <div className="modal-container custom-dataset-modal">
        <div className="modal-header">
          <h3>Create Custom Dataset</h3>
          <div className="step-indicator">
            <div className={`step ${currentStep >= 1 ? "active" : ""}`}>1</div>
            <div className={`step ${currentStep >= 2 ? "active" : ""}`}>2</div>
            <div className={`step ${currentStep >= 3 ? "active" : ""}`}>3</div>
          </div>
          <button className="close-modal-btn" onClick={handleClose}>
            <i className="fas fa-times"></i>
          </button>
        </div>

        <div className="modal-body">
          {/* Step 1: File Upload and Preview */}
          {currentStep === 1 && (
            <div className="step-content">
              <h4>
                <i className="fas fa-upload"></i>
                Upload Dataset File
              </h4>

              <div
                className={`file-upload-area ${isDragOver ? "drag-over" : ""}`}
                onDragEnter={handleDragEnter}
                onDragLeave={handleDragLeave}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <i className="fas fa-cloud-upload-alt"></i>
                <p>Drag and drop your dataset file here, or click to browse</p>
                <p className="file-requirements">
                  Supports CSV and Excel files (max 50MB, 100K rows)
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,.xlsx,.xls"
                  onChange={handleFileInputChange}
                  style={{ display: "none" }}
                />
              </div>

              {formState.file && (
                <div className="file-info">
                  <div className="file-details">
                    <i className="fas fa-file-alt"></i>
                    <span>{formState.file.name}</span>
                    <span className="file-size">
                      {formatFileSize(formState.file.size)}
                    </span>
                  </div>
                </div>
              )}

              {formState.preview && (
                <div className="data-preview">
                  <h5>
                    <i className="fas fa-eye"></i>
                    Data Preview
                  </h5>

                  <div className="preview-stats">
                    <div className="stat-item">
                      <span className="label">Shape:</span>
                      <span className="value">
                        {formState.preview.shape[0]} rows ×{" "}
                        {formState.preview.shape[1]} columns
                      </span>
                    </div>
                    <div className="stat-item">
                      <span className="label">Missing values:</span>
                      <span className="value">
                        {Object.values(formState.preview.missing_values).reduce(
                          (a, b) => a + b,
                          0
                        )}{" "}
                        total
                      </span>
                    </div>
                  </div>

                  <div className="preview-table-wrapper">
                    <table className="preview-table">
                      <thead>
                        <tr>
                          {formState.preview.columns.map((col, index) => (
                            <th key={index}>
                              <div className="column-header">
                                <span className="column-name">{col}</span>
                                <span
                                  className="column-type"
                                  style={{
                                    backgroundColor: getDataTypeColor(
                                      formState.preview!.data_types[col]
                                    ),
                                  }}
                                >
                                  {formState.preview!.data_types[col]}
                                </span>
                              </div>
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {formState.preview.sample_data &&
                        formState.preview.sample_data.length > 0 ? (
                          formState.preview.sample_data.map(
                            (record, rowIndex) => (
                              <tr key={rowIndex}>
                                {formState.preview!.columns.map(
                                  (col, cellIndex) => (
                                    <td key={cellIndex}>
                                      {record[col] === null ||
                                      record[col] === undefined ? (
                                        <span className="missing-value">—</span>
                                      ) : (
                                        String(record[col])
                                      )}
                                    </td>
                                  )
                                )}
                              </tr>
                            )
                          )
                        ) : (
                          <tr>
                            <td
                              colSpan={formState.preview!.columns.length}
                              style={{ textAlign: "center", padding: "20px" }}
                            >
                              No data available
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Step 2: Configuration */}
          {currentStep === 2 && formState.preview && (
            <div className="step-content">
              <h4>
                <i className="fas fa-cog"></i>
                Configure Dataset
              </h4>

              <div className="config-section">
                <div className="config-row">
                  <label htmlFor="target-column">Target Column:</label>
                  <select
                    id="target-column"
                    value={formState.config.target_column || ""}
                    onChange={(e) =>
                      handleConfigChange({ target_column: e.target.value })
                    }
                  >
                    <option value="">Select target column</option>
                    {formState.preview.columns.map((col) => (
                      <option key={col} value={col}>
                        {col} ({formState.preview!.data_types[col]})
                      </option>
                    ))}
                  </select>
                </div>

                <div className="config-row">
                  <label>Task Type:</label>
                  <div className="radio-group">
                    <label className="radio-item">
                      <input
                        type="radio"
                        name="task_type"
                        value="classification"
                        checked={
                          formState.config.task_type === "classification"
                        }
                        onChange={(e) =>
                          handleConfigChange({
                            task_type: e.target.value as "classification",
                          })
                        }
                      />
                      <span>Classification</span>
                    </label>
                    <label className="radio-item">
                      <input
                        type="radio"
                        name="task_type"
                        value="regression"
                        checked={formState.config.task_type === "regression"}
                        onChange={(e) =>
                          handleConfigChange({
                            task_type: e.target.value as "regression",
                          })
                        }
                      />
                      <span>Regression</span>
                    </label>
                  </div>
                </div>

                <div className="config-row">
                  <label>Feature Columns:</label>
                  <div className="feature-selection">
                    {formState.preview.columns
                      .filter((col) => col !== formState.config.target_column)
                      .map((col) => (
                        <label key={col} className="checkbox-item">
                          <input
                            type="checkbox"
                            checked={
                              formState.config.feature_columns?.includes(col) ||
                              false
                            }
                            onChange={() => handleFeatureColumnToggle(col)}
                          />
                          <span className="checkbox-label">
                            {col}
                            <span className="column-type-badge">
                              {formState.preview!.data_types[col]}
                            </span>
                          </span>
                        </label>
                      ))}
                  </div>
                </div>

                <div className="config-row">
                  <label htmlFor="missing-strategy">
                    Missing Value Strategy:
                  </label>
                  <select
                    id="missing-strategy"
                    value={formState.config.missing_value_strategy || "drop"}
                    onChange={(e) =>
                      handleConfigChange({
                        missing_value_strategy: e.target.value as
                          | "drop"
                          | "fill_mean"
                          | "fill_mode",
                      })
                    }
                  >
                    <option value="drop">Drop rows with missing values</option>
                    <option value="fill_mean">Fill with mean (numeric)</option>
                    <option value="fill_mode">
                      Fill with mode (most common)
                    </option>
                  </select>
                </div>
              </div>

              {formState.validation && (
                <div className="validation-results">
                  <h5>
                    <i className="fas fa-check-circle"></i>
                    Validation Results
                  </h5>

                  {formState.validation.errors.length > 0 && (
                    <div className="validation-group errors">
                      <h6>
                        <i className="fas fa-exclamation-triangle"></i>
                        Errors
                      </h6>
                      <ul>
                        {formState.validation.errors.map((error, index) => (
                          <li key={index}>{error}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {formState.validation.warnings.length > 0 && (
                    <div className="validation-group warnings">
                      <h6>
                        <i className="fas fa-exclamation-circle"></i>
                        Warnings
                      </h6>
                      <ul>
                        {formState.validation.warnings.map((warning, index) => (
                          <li key={index}>{warning}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {formState.validation.recommendations.length > 0 && (
                    <div className="validation-group recommendations">
                      <h6>
                        <i className="fas fa-lightbulb"></i>
                        Recommendations
                      </h6>
                      <ul>
                        {formState.validation.recommendations.map(
                          (rec, index) => (
                            <li key={index}>{rec}</li>
                          )
                        )}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Step 3: Review and Create */}
          {currentStep === 3 && formState.preview && (
            <div className="step-content">
              <h4>
                <i className="fas fa-eye"></i>
                Review and Create
              </h4>

              <div className="review-section">
                <div className="config-row">
                  <label htmlFor="dataset-name">Dataset Name:</label>
                  <input
                    id="dataset-name"
                    type="text"
                    value={formState.datasetName}
                    onChange={(e) =>
                      setFormState((prev) => ({
                        ...prev,
                        datasetName: e.target.value,
                      }))
                    }
                    placeholder="Enter dataset name"
                    maxLength={50}
                  />
                </div>

                <div className="review-summary">
                  <h5>Configuration Summary</h5>
                  <div className="summary-item">
                    <span className="label">File:</span>
                    <span className="value">{formState.file?.name}</span>
                  </div>
                  <div className="summary-item">
                    <span className="label">Shape:</span>
                    <span className="value">
                      {formState.preview.shape[0]} rows ×{" "}
                      {formState.preview.shape[1]} columns
                    </span>
                  </div>
                  <div className="summary-item">
                    <span className="label">Target Column:</span>
                    <span className="value">
                      {formState.config.target_column}
                    </span>
                  </div>
                  <div className="summary-item">
                    <span className="label">Task Type:</span>
                    <span className="value">{formState.config.task_type}</span>
                  </div>
                  <div className="summary-item">
                    <span className="label">Feature Columns:</span>
                    <span className="value">
                      {formState.config.feature_columns?.length} selected
                    </span>
                  </div>
                  <div className="summary-item">
                    <span className="label">Missing Value Strategy:</span>
                    <span className="value">
                      {formState.config.missing_value_strategy}
                    </span>
                  </div>
                </div>

                {formState.validation && !formState.validation.valid && (
                  <div className="validation-warning">
                    <i className="fas fa-exclamation-triangle"></i>
                    <span>
                      There are validation issues with your configuration.
                      Please review and fix them before creating the dataset.
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}

          {error && (
            <div className="error-message">
              <i className="fas fa-exclamation-circle"></i>
              {error}
            </div>
          )}

          {isLoading && (
            <div className="loading-overlay">
              <div className="loading-spinner">
                <i className="fas fa-spinner fa-spin"></i>
                <span>
                  {currentStep === 1 && "Processing file..."}
                  {currentStep === 2 && "Validating configuration..."}
                  {currentStep === 3 && "Creating dataset..."}
                </span>
              </div>
            </div>
          )}
        </div>

        <div className="modal-footer">
          <button className="cancel-btn" onClick={handleClose}>
            Cancel
          </button>

          <div className="nav-buttons">
            {currentStep > 1 && (
              <button
                className="back-btn"
                onClick={() =>
                  setCurrentStep((prev) => (prev - 1) as WizardStep)
                }
                disabled={isLoading}
              >
                <i className="fas fa-chevron-left"></i>
                Back
              </button>
            )}

            {currentStep < 3 ? (
              <button
                className="next-btn"
                onClick={() =>
                  setCurrentStep((prev) => (prev + 1) as WizardStep)
                }
                disabled={
                  isLoading ||
                  (currentStep === 1 && !canProceedToStep2) ||
                  (currentStep === 2 && !canProceedToStep3)
                }
              >
                Next
                <i className="fas fa-chevron-right"></i>
              </button>
            ) : (
              <button
                className="create-btn"
                onClick={handleCreateDataset}
                disabled={
                  isLoading ||
                  !formState.datasetName.trim() ||
                  !!(formState.validation && !formState.validation.valid)
                }
              >
                <i className="fas fa-plus"></i>
                Create Dataset
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CustomDatasetModal;
