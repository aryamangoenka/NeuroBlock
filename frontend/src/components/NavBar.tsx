import React, { useState, useEffect, useRef } from "react";
import { NavLink } from "react-router-dom";
import "../styles/components/NavBar.scss";

// Define a custom event for saving the model - copied from NewBuildPage for direct access
const triggerSaveModel = () => {
  const event = new CustomEvent("saveModel");
  window.dispatchEvent(event);
};
void triggerSaveModel;
// Define a custom event for training the model
const triggerTrainModel = () => {
  const event = new CustomEvent("trainModel");
  window.dispatchEvent(event);
};

// Define a custom event for stopping the training
const triggerStopTraining = () => {
  const event = new CustomEvent("stopTraining");
  window.dispatchEvent(event);
};

// Define a custom event for exporting the model
const triggerExportModel = (format: string) => {
  const event = new CustomEvent("exportModel", {
    detail: { format },
  });
  window.dispatchEvent(event);
};

// Function to check if dataset is properly selected
const checkDatasetSelected = () => {
  // This will return a promise that resolves with true/false
  return new Promise((resolve) => {
    console.log("Checking if dataset is properly selected...");

    // Create and dispatch an event to check dataset
    const checkEvent = new CustomEvent("checkDatasetSelected", {
      detail: {
        callback: (hasDataset: boolean, datasetName: string) => {
          console.log(
            `Dataset check result: ${hasDataset}, dataset: ${datasetName}`
          );
          resolve(hasDataset);
        },
      },
    });

    console.log("Dispatching checkDatasetSelected event");
    window.dispatchEvent(checkEvent);

    // Set a timeout in case the event isn't handled
    setTimeout(() => {
      console.warn("Dataset check timed out");
      resolve(false);
    }, 500);
  });
};

const NavBar: React.FC = () => {
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [hasSelectedDataset, setHasSelectedDataset] = useState<boolean>(false);
  const [exportDropdownOpen, setExportDropdownOpen] = useState<boolean>(false);
  const exportDropdownRef = useRef<HTMLDivElement>(null);

  // Listen for training state changes
  useEffect(() => {
    const handleTrainingStateChange = (event: CustomEvent) => {
      setIsTraining(event.detail.isTraining);
    };

    const handleDatasetChange = (event: CustomEvent) => {
      const datasetName = event.detail.dataset;
      console.log("Dataset change event received in NavBar:", datasetName);

      // Only set true if the dataset name is not empty
      const hasDataset = !!datasetName && datasetName.trim() !== "";
      console.log("Setting hasSelectedDataset to:", hasDataset);

      setHasSelectedDataset(hasDataset);
    };

    window.addEventListener(
      "trainingStateChange",
      handleTrainingStateChange as EventListener
    );
    window.addEventListener(
      "datasetChange",
      handleDatasetChange as EventListener
    );

    return () => {
      window.removeEventListener(
        "trainingStateChange",
        handleTrainingStateChange as EventListener
      );
      window.removeEventListener(
        "datasetChange",
        handleDatasetChange as EventListener
      );
    };
  }, []);

  // Listen for export model event
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        exportDropdownRef.current &&
        !exportDropdownRef.current.contains(event.target as Node)
      ) {
        setExportDropdownOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const handleSaveModelClick = async () => {
    console.log(
      "Save button clicked. Current hasSelectedDataset state:",
      hasSelectedDataset
    );

    // Double-check dataset selection from the context directly
    const datasetConfirmed = await checkDatasetSelected();
    console.log("Dataset confirmed from NewBuildPage:", datasetConfirmed);

    if (!datasetConfirmed) {
      console.warn("Dataset not detected when trying to save");
      alert(
        "No dataset selected! Please select a dataset in the Settings tab before saving."
      );
      return;
    }

    // Dispatch getTrainingConfig event to ensure latest config is saved to localStorage
    const getConfigEvent = new CustomEvent("getTrainingConfig");
    window.dispatchEvent(getConfigEvent);

    // At this point we're confident the model has a dataset selected
    console.log("Dataset is selected, triggering save model event");

    // Use a custom event that includes diagnostic info
    const event = new CustomEvent("saveModel", {
      detail: {
        source: "NavBar",
        timestamp: new Date().toISOString(),
      },
    });
    window.dispatchEvent(event);
  };

  const handleTrainModelClick = () => {
    if (!hasSelectedDataset) {
      alert("No dataset selected! Please select a dataset before training.");
      return;
    }

    if (!isTraining) {
      triggerTrainModel();
    }
  };

  const handleStopTrainingClick = () => {
    if (isTraining) {
      triggerStopTraining();
    }
  };

  const handleExportClick = (format: string) => {
    if (!hasSelectedDataset) {
      alert("No dataset selected! Please select a dataset before exporting.");
      return;
    }

    triggerExportModel(format);
    setExportDropdownOpen(false);
  };

  const toggleExportDropdown = () => {
    setExportDropdownOpen(!exportDropdownOpen);
  };

  return (
    <nav className="navbar">
      <div className="container d-flex justify-content-between align-items-center">
        <div className="navbar-brand">
          <i className="fas fa-brain brand-icon"></i>
          <span className="brand-text">NeuroBlock</span>
        </div>

        <div className="navbar-actions">
          <NavLink
            to="/tutorial"
            className="tutorial-button"
            title="Learn how to use DND Neural Network"
          >
            <i className="fas fa-question-circle"></i>
            <span>Tutorial</span>
          </NavLink>

          <button
            className="save-model-button"
            title="Save your current model"
            onClick={handleSaveModelClick}
            disabled={isTraining}
          >
            <i className="fas fa-save"></i>
            <span>Save Model</span>
          </button>

          <button
            className={`train-button-red ${isTraining ? "training" : ""}`}
            title={
              !hasSelectedDataset
                ? "Please select a dataset first"
                : isTraining
                ? "Training in progress"
                : "Train your model"
            }
            onClick={handleTrainModelClick}
            disabled={isTraining}
          >
            {isTraining ? (
              <>
                <i className="fas fa-spinner fa-spin"></i>
                <span>Training...</span>
              </>
            ) : (
              <>
                <i className="fas fa-play-circle"></i>
                <span>Train</span>
              </>
            )}
          </button>

          {isTraining && (
            <button
              className="stop-button-red"
              title="Stop the current training session"
              onClick={handleStopTrainingClick}
            >
              <i className="fas fa-stop-circle"></i>
              <span>Stop</span>
            </button>
          )}

          <div className="export-dropdown-container" ref={exportDropdownRef}>
            <button
              className="export-button-black"
              title="Export your model"
              onClick={toggleExportDropdown}
              disabled={isTraining}
            >
              <i className="fas fa-file-export"></i>
              <span>Export</span>
              <i
                className={`fas fa-caret-down ms-1 ${
                  exportDropdownOpen ? "dropdown-open" : ""
                }`}
              ></i>
            </button>

            {exportDropdownOpen && (
              <div className="export-dropdown">
                <button
                  onClick={() => handleExportClick("py")}
                  className="export-option"
                >
                  <i className="fab fa-python"></i> Python
                </button>
                <button
                  onClick={() => handleExportClick("keras")}
                  className="export-option"
                >
                  <i className="fas fa-cube"></i> Keras
                </button>
                <button
                  onClick={() => handleExportClick("pytorch")}
                  className="export-option"
                >
                  <i className="fas fa-fire"></i> PyTorch
                </button>
                <button
                  onClick={() => handleExportClick("savedmodel")}
                  className="export-option"
                >
                  <i className="fas fa-save"></i> SavedModel
                </button>
                <button
                  onClick={() => handleExportClick("ipynb")}
                  className="export-option"
                >
                  <i className="fas fa-book"></i> Notebook
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default NavBar;
