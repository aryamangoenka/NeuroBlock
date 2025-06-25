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
  isImageArchive,
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
  folderFiles: File[] | null;
  preview: DatasetPreview | null;
  config: Partial<DatasetConfig>;
  validation: ValidationResult | null;
  datasetName: string;
  isImageDataset: boolean;
  isFolderUpload: boolean;
  imageConfig: {
    target_size: string;
    channels: number;
  };
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
  const folderInputRef = useRef<HTMLInputElement>(null);

  const [formState, setFormState] = useState<FormState>({
    file: null,
    folderFiles: null,
    preview: null,
    config: {
      missing_value_strategy: "drop",
    },
    validation: null,
    datasetName: "",
    isImageDataset: false,
    isFolderUpload: false,
    imageConfig: {
      target_size: "224,224",
      channels: 3,
    },
  });

  const resetForm = useCallback(() => {
    setCurrentStep(1);
    setIsLoading(false);
    setError(null);
    setIsDragOver(false);
    setFormState({
      file: null,
      folderFiles: null,
      preview: null,
      config: { missing_value_strategy: "drop" },
      validation: null,
      datasetName: "",
      isImageDataset: false,
      isFolderUpload: false,
      imageConfig: {
        target_size: "224,224",
        channels: 3,
      },
    });
  }, []);

  const handleClose = useCallback(() => {
    resetForm();
    onClose();
  }, [resetForm, onClose]);

  const processFolderFiles = (files: File[]): FormData => {
    const formData = new FormData();

    files.forEach((file, index) => {
      formData.append(`files`, file);
      formData.append(
        `file_paths[${index}]`,
        file.webkitRelativePath || file.name
      );
    });

    return formData;
  };

  const handleFolderSelect = async (files: File[]) => {
    if (!files || files.length === 0) return;

    const imageFiles = files.filter((file) => {
      const extension = file.name
        .toLowerCase()
        .substring(file.name.lastIndexOf("."));
      return [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"].includes(
        extension
      );
    });

    if (imageFiles.length === 0) {
      setError(
        "No image files found in the folder. Please ensure your folder contains JPG, PNG, or other supported image formats."
      );
      return;
    }

    const totalSize = imageFiles.reduce((sum, file) => sum + file.size, 0);
    const maxSize = 100 * 1024 * 1024;
    if (totalSize > maxSize) {
      setError(
        `Total folder size (${(totalSize / (1024 * 1024)).toFixed(
          1
        )}MB) exceeds maximum allowed size (${maxSize / (1024 * 1024)}MB)`
      );
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const folderStructure = analyzeFolderStructure(imageFiles);

      if (folderStructure.classes.length === 0) {
        setError(
          "No class folders detected. Please organize your images in folders named after each class."
        );
        return;
      }

      const preview: DatasetPreview = {
        filename: "folder_upload",
        dataset_type: "image",
        shape: [folderStructure.totalImages, folderStructure.classes.length],
        image_shape: [224, 224, 3],
        columns: ["image_data", "class"],
        data_types: { image_data: "image", class: "categorical" },
        missing_values: { image_data: 0, class: 0 },
        missing_percentages: { image_data: 0.0, class: 0.0 },
        sample_data: folderStructure.sampleImages,
        numeric_statistics: {},
        categorical_info: {
          class: {
            unique_count: folderStructure.classes.length,
            top_values: folderStructure.classDistribution,
          },
        },
        class_distribution: folderStructure.classDistribution,
        total_images: folderStructure.totalImages,
        num_classes: folderStructure.classes.length,
        class_names: folderStructure.classes,
        target_size: [224, 224],
        channels: 3,
        row_limit_exceeded: false,
        analysis_timestamp: new Date().toISOString(),
      };

      setFormState((prev) => ({
        ...prev,
        file: null,
        folderFiles: imageFiles,
        preview,
        datasetName: "folder_dataset_" + Date.now(),
        isImageDataset: true,
        isFolderUpload: true,
        config: {
          ...prev.config,
          task_type: "classification",
        },
      }));
    } catch (err) {
      setError("Failed to process folder. Please try again.");
      console.error("Folder processing error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const analyzeFolderStructure = (files: File[]) => {
    const classDistribution: Record<string, number> = {};
    const sampleImages: any[] = [];
    const classes: string[] = [];

    files.forEach((file) => {
      const path = file.webkitRelativePath || file.name;
      const pathParts = path.split("/");

      let className = "default";
      if (pathParts.length > 1) {
        className = pathParts[pathParts.length - 2];
      }

      if (!classDistribution[className]) {
        classDistribution[className] = 0;
        classes.push(className);

        if (sampleImages.length < 5) {
          sampleImages.push({
            class: className,
            filename: file.name,
            size: [0, 0],
            mode: "RGB",
          });
        }
      }

      classDistribution[className]++;
    });

    return {
      classes: classes.sort(),
      classDistribution,
      sampleImages,
      totalImages: files.length,
    };
  };

  const handleFileSelect = async (file: File) => {
    if (!file) return;

    const isImage = isImageArchive(file);

    const tabularTypes = [".csv", ".xlsx", ".xls"];
    const imageTypes = [".zip", ".tar", ".tar.gz", ".tgz"];
    const fileExtension = file.name
      .toLowerCase()
      .substring(file.name.lastIndexOf("."));

    const isValidTabular = tabularTypes.includes(fileExtension);
    const isValidImage = imageTypes.some((ext) =>
      file.name.toLowerCase().endsWith(ext)
    );

    if (!isValidTabular && !isValidImage) {
      setError(
        "Please select a CSV, Excel file (.csv, .xlsx, .xls) or image archive (.zip, .tar, .tar.gz, .tgz)"
      );
      return;
    }

    const maxSize = isImage ? 100 * 1024 * 1024 : 50 * 1024 * 1024;
    if (file.size > maxSize) {
      setError(`File size must be less than ${maxSize / (1024 * 1024)}MB`);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const imageConfig = isImage
        ? {
            target_size: formState.imageConfig.target_size,
            channels: formState.imageConfig.channels,
          }
        : undefined;

      const preview = await previewDataset(file, imageConfig);
      setFormState((prev) => ({
        ...prev,
        file,
        folderFiles: null,
        preview,
        datasetName: file.name.replace(/\.[^/.]+$/, ""),
        isImageDataset: isImage,
        isFolderUpload: false,
        config: {
          ...prev.config,
          task_type: isImage ? "classification" : prev.config.task_type,
        },
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

    const items = Array.from(e.dataTransfer.items);
    const files = Array.from(e.dataTransfer.files);

    const hasDirectories = items.some((item) => {
      const entry = item.webkitGetAsEntry?.();
      return entry?.isDirectory;
    });

    if (hasDirectories) {
      handleFolderDrop(items);
    } else if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleFolderDrop = async (items: DataTransferItem[]) => {
    const allFiles: File[] = [];

    const processEntry = async (entry: any): Promise<void> => {
      return new Promise((resolve) => {
        if (entry.isFile) {
          entry.file((file: File) => {
            const newFile = new File([file], file.name, { type: file.type });
            (newFile as any).webkitRelativePath = entry.fullPath.substring(1);
            allFiles.push(newFile);
            resolve();
          });
        } else if (entry.isDirectory) {
          const reader = entry.createReader();
          reader.readEntries(async (entries: any[]) => {
            await Promise.all(entries.map(processEntry));
            resolve();
          });
        } else {
          resolve();
        }
      });
    };

    const entries = items
      .map((item) => item.webkitGetAsEntry())
      .filter(Boolean);
    await Promise.all(entries.map(processEntry));

    if (allFiles.length > 0) {
      handleFolderSelect(allFiles);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleFolderInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFolderSelect(Array.from(files));
    }
  };

  const handleConfigChange = async (updates: Partial<DatasetConfig>) => {
    const newConfig = { ...formState.config, ...updates };
    setFormState((prev) => ({ ...prev, config: newConfig }));

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

  const handleCreateDataset = async () => {
    if (
      (!formState.file && !formState.folderFiles) ||
      !formState.config ||
      !formState.datasetName
    ) {
      setError("Missing required information");
      return;
    }

    const config: DatasetConfig = {
      dataset_name: formState.datasetName,
      task_type: formState.config.task_type!,
      missing_value_strategy: formState.config.missing_value_strategy,
    };

    if (formState.isImageDataset) {
      config.target_size = formState.imageConfig.target_size;
      config.channels = formState.imageConfig.channels;
    } else {
      config.target_column = formState.config.target_column!;
      config.feature_columns = formState.config.feature_columns!;
    }

    setIsLoading(true);
    setError(null);

    try {
      let result;
      if (formState.isFolderUpload) {
        const formData = processFolderFiles(formState.folderFiles!);
        result = await createCustomDataset(formData, config);
      } else {
        result = await createCustomDataset(formState.file!, config);
      }
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

  const canProceedToStep2 = !!(formState.file || formState.folderFiles);
  const canProceedToStep3 = !!(
    canProceedToStep2 &&
    formState.config.task_type &&
    (formState.isImageDataset ||
      (formState.config.target_column &&
        formState.config.feature_columns &&
        formState.config.feature_columns.length > 0))
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
              >
                <i className="fas fa-cloud-upload-alt"></i>
                <p>Drag and drop your dataset file or image folder here</p>
                <p className="file-requirements">
                  Supports CSV, Excel files (max 50MB), image archives (ZIP,
                  TAR, max 100MB), or drag image folders directly
                </p>

                <div className="upload-options">
                  <button
                    type="button"
                    className="upload-btn file-btn"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <i className="fas fa-file"></i>
                    Browse Files
                  </button>

                  <button
                    type="button"
                    className="upload-btn folder-btn"
                    onClick={() => folderInputRef.current?.click()}
                  >
                    <i className="fas fa-folder"></i>
                    Browse Folders
                  </button>
                </div>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,.xlsx,.xls,.zip,.tar,.tar.gz,.tgz"
                  onChange={handleFileInputChange}
                  style={{ display: "none" }}
                />

                <input
                  ref={folderInputRef}
                  type="file"
                  webkitdirectory=""
                  multiple
                  onChange={handleFolderInputChange}
                  style={{ display: "none" }}
                />
              </div>

              {(formState.file || formState.folderFiles) && (
                <div className="file-info">
                  <div className="file-details">
                    {formState.isFolderUpload ? (
                      <>
                        <i className="fas fa-folder"></i>
                        <span>
                          Folder Upload: {formState.folderFiles?.length} images
                        </span>
                        <span className="file-size">
                          {formatFileSize(
                            formState.folderFiles?.reduce(
                              (sum, file) => sum + file.size,
                              0
                            ) || 0
                          )}
                        </span>
                      </>
                    ) : (
                      <>
                        <i className="fas fa-file-alt"></i>
                        <span>{formState.file?.name}</span>
                        <span className="file-size">
                          {formatFileSize(formState.file?.size || 0)}
                        </span>
                      </>
                    )}
                  </div>
                </div>
              )}

              {formState.preview && (
                <div className="data-preview">
                  <h5>
                    <i className="fas fa-eye"></i>
                    {formState.isImageDataset
                      ? "Image Dataset Preview"
                      : "Data Preview"}
                  </h5>

                  {formState.isImageDataset && (
                    <div className="development-notice">
                      <div className="notice-header">
                        <i className="fas fa-exclamation-triangle"></i>
                        <span>Development Notice</span>
                      </div>
                      <p>
                        Image dataset support is currently under active
                        development. While the basic functionality is working,
                        some features may be experimental or subject to change.
                        Please report any issues you encounter.
                      </p>
                    </div>
                  )}

                  <div className="preview-stats">
                    {formState.isImageDataset ? (
                      <>
                        <div className="stat-item">
                          <span className="label">Total Images:</span>
                          <span className="value">
                            {formState.preview.total_images ||
                              formState.preview.shape[0]}
                          </span>
                        </div>
                        <div className="stat-item">
                          <span className="label">Classes:</span>
                          <span className="value">
                            {formState.preview.num_classes ||
                              formState.preview.shape[1]}
                          </span>
                        </div>
                        <div className="stat-item">
                          <span className="label">Image Size:</span>
                          <span className="value">
                            {formState.preview.target_size
                              ? `${formState.preview.target_size[0]}×${formState.preview.target_size[1]}`
                              : `${formState.preview.image_shape?.[0] || 224}×${
                                  formState.preview.image_shape?.[1] || 224
                                }`}
                          </span>
                        </div>
                        <div className="stat-item">
                          <span className="label">Channels:</span>
                          <span className="value">
                            {formState.preview.channels ||
                              formState.preview.image_shape?.[2] ||
                              3}
                            (
                            {(formState.preview.channels ||
                              formState.preview.image_shape?.[2] ||
                              3) === 1
                              ? "Grayscale"
                              : "RGB"}
                            )
                          </span>
                        </div>
                      </>
                    ) : (
                      <>
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
                            {Object.values(
                              formState.preview.missing_values
                            ).reduce((a, b) => a + b, 0)}{" "}
                            total
                          </span>
                        </div>
                      </>
                    )}
                  </div>

                  {formState.isImageDataset ? (
                    <div className="image-preview-section">
                      <h6>
                        <i className="fas fa-chart-bar"></i>
                        Class Distribution
                      </h6>
                      {formState.preview.class_distribution && (
                        <div className="class-distribution">
                          {Object.entries(
                            formState.preview.class_distribution
                          ).map(([className, count]) => {
                            const maxCount = Math.max(
                              ...Object.values(
                                formState.preview?.class_distribution || {}
                              )
                            );
                            return (
                              <div key={className} className="class-item">
                                <div className="class-info">
                                  <span className="class-name">
                                    {className}
                                  </span>
                                  <span className="class-count">
                                    {count} images
                                  </span>
                                </div>
                                <div className="class-bar">
                                  <div
                                    className="class-bar-fill"
                                    style={{
                                      width: `${(count / maxCount) * 100}%`,
                                    }}
                                  ></div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}

                      {formState.preview.sample_data &&
                        formState.preview.sample_data.length > 0 && (
                          <div className="sample-images">
                            <h6>
                              <i className="fas fa-images"></i>
                              Sample Images Found
                            </h6>
                            <div className="sample-list">
                              {formState.preview.sample_data
                                .slice(0, 5)
                                .map((sample: any, index: number) => (
                                  <div key={index} className="sample-item">
                                    <i className="fas fa-image"></i>
                                    <span className="sample-class">
                                      {sample.class}
                                    </span>
                                    <span className="sample-filename">
                                      {sample.filename}
                                    </span>
                                    <span className="sample-size">
                                      {sample.size?.[0]}×{sample.size?.[1]}
                                    </span>
                                  </div>
                                ))}
                            </div>
                          </div>
                        )}
                    </div>
                  ) : (
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
                                          <span className="missing-value">
                                            —
                                          </span>
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
                  )}
                </div>
              )}
            </div>
          )}

          {currentStep === 2 && formState.preview && (
            <div className="step-content">
              <h4>
                <i className="fas fa-cog"></i>
                Configure Dataset
              </h4>

              <div className="config-section">
                {formState.isImageDataset ? (
                  <>
                    <div className="config-info">
                      <div className="info-item">
                        <i className="fas fa-info-circle"></i>
                        <span>
                          Image datasets are automatically configured for
                          classification tasks. Classes are determined by folder
                          names in your archive.
                        </span>
                      </div>
                    </div>

                    <div className="development-notice">
                      <div className="notice-header">
                        <i className="fas fa-exclamation-triangle"></i>
                        <span>Development Notice</span>
                      </div>
                      <p>
                        Image dataset support is currently under active
                        development. Some advanced features may be limited or
                        experimental. We recommend testing with smaller datasets
                        first.
                      </p>
                    </div>

                    <div className="config-row">
                      <label htmlFor="image-size">Target Image Size:</label>
                      <select
                        id="image-size"
                        value={formState.imageConfig.target_size}
                        onChange={(e) =>
                          setFormState((prev) => ({
                            ...prev,
                            imageConfig: {
                              ...prev.imageConfig,
                              target_size: e.target.value,
                            },
                            config: {
                              ...prev.config,
                              target_size: e.target.value,
                            },
                          }))
                        }
                      >
                        <option value="224,224">224×224 (Recommended)</option>
                        <option value="128,128">128×128 (Faster)</option>
                        <option value="256,256">
                          256×256 (Higher Quality)
                        </option>
                        <option value="512,512">
                          512×512 (Very High Quality)
                        </option>
                      </select>
                    </div>

                    <div className="config-row">
                      <label htmlFor="channels">Color Channels:</label>
                      <select
                        id="channels"
                        value={formState.imageConfig.channels}
                        onChange={(e) =>
                          setFormState((prev) => ({
                            ...prev,
                            imageConfig: {
                              ...prev.imageConfig,
                              channels: parseInt(e.target.value),
                            },
                            config: {
                              ...prev.config,
                              channels: parseInt(e.target.value),
                            },
                          }))
                        }
                      >
                        <option value={3}>RGB (3 channels)</option>
                        <option value={1}>Grayscale (1 channel)</option>
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
                            checked={true}
                            disabled={true}
                          />
                          <span>
                            Classification (Only supported for images)
                          </span>
                        </label>
                      </div>
                    </div>

                    <div className="detected-classes">
                      <h6>
                        <i className="fas fa-tags"></i>
                        Detected Classes ({formState.preview.num_classes || 0})
                      </h6>
                      <div className="class-tags">
                        {formState.preview.class_names?.map((className) => (
                          <span key={className} className="class-tag">
                            {className}
                          </span>
                        ))}
                      </div>
                    </div>
                  </>
                ) : (
                  <>
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
                            checked={
                              formState.config.task_type === "regression"
                            }
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
                          .filter(
                            (col) => col !== formState.config.target_column
                          )
                          .map((col) => (
                            <label key={col} className="checkbox-item">
                              <input
                                type="checkbox"
                                checked={
                                  formState.config.feature_columns?.includes(
                                    col
                                  ) || false
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
                        value={
                          formState.config.missing_value_strategy || "drop"
                        }
                        onChange={(e) =>
                          handleConfigChange({
                            missing_value_strategy: e.target.value as
                              | "drop"
                              | "fill_mean"
                              | "fill_mode",
                          })
                        }
                      >
                        <option value="drop">
                          Drop rows with missing values
                        </option>
                        <option value="fill_mean">
                          Fill with mean (numeric)
                        </option>
                        <option value="fill_mode">
                          Fill with mode (most common)
                        </option>
                      </select>
                    </div>
                  </>
                )}
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
                    <span className="label">Dataset Type:</span>
                    <span className="value">
                      {formState.isImageDataset
                        ? "Image Dataset"
                        : "Tabular Dataset"}
                    </span>
                  </div>
                  {formState.isImageDataset ? (
                    <>
                      <div className="summary-item">
                        <span className="label">Total Images:</span>
                        <span className="value">
                          {formState.preview.total_images ||
                            formState.preview.shape[0]}
                        </span>
                      </div>
                      <div className="summary-item">
                        <span className="label">Classes:</span>
                        <span className="value">
                          {formState.preview.num_classes ||
                            formState.preview.shape[1]}
                        </span>
                      </div>
                      <div className="summary-item">
                        <span className="label">Image Size:</span>
                        <span className="value">
                          {formState.imageConfig.target_size.replace(",", "×")}
                        </span>
                      </div>
                      <div className="summary-item">
                        <span className="label">Channels:</span>
                        <span className="value">
                          {formState.imageConfig.channels} (
                          {formState.imageConfig.channels === 1
                            ? "Grayscale"
                            : "RGB"}
                          )
                        </span>
                      </div>
                      <div className="summary-item">
                        <span className="label">Task Type:</span>
                        <span className="value">Classification</span>
                      </div>
                    </>
                  ) : (
                    <>
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
                        <span className="value">
                          {formState.config.task_type}
                        </span>
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
                    </>
                  )}
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
