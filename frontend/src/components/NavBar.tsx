import React, { useState, useEffect } from "react";
import { NavLink } from "react-router-dom";
import "../styles/components/NavBar.scss";

// Define a custom event for saving the model - copied from NewBuildPage for direct access
const triggerSaveModel = () => {
  const event = new CustomEvent("saveModel");
  window.dispatchEvent(event);
};

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

const NavBar: React.FC = () => {
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [hasSelectedDataset, setHasSelectedDataset] = useState<boolean>(false);

  // Listen for training state changes
  useEffect(() => {
    const handleTrainingStateChange = (event: CustomEvent) => {
      setIsTraining(event.detail.isTraining);
    };

    const handleDatasetChange = (event: CustomEvent) => {
      setHasSelectedDataset(!!event.detail.dataset);
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

  const handleSaveModelClick = () => {
    triggerSaveModel();
  };

  const handleTrainModelClick = () => {
    if (hasSelectedDataset && !isTraining) {
      triggerTrainModel();
    }
  };

  const handleStopTrainingClick = () => {
    if (isTraining) {
      triggerStopTraining();
    }
  };

  return (
    <nav className="navbar">
      <div className="container d-flex justify-content-between align-items-center">
        <div className="navbar-brand">
          <i className="fas fa-brain brand-icon"></i>
          <span className="brand-text">NeuroBlock</span>
        </div>

        <div className="navbar-actions d-flex align-items-center">
          <NavLink
            to="/tutorial"
            className="tutorial-button me-3"
            title="Learn how to use DND Neural Network"
          >
            <i className="fas fa-question-circle"></i>
            <span>Tutorial</span>
          </NavLink>

          <button
            className="save-model-button me-3"
            title="Save your current model"
            onClick={handleSaveModelClick}
            disabled={isTraining}
          >
            <i className="fas fa-save"></i>
            <span>Save Model</span>
          </button>

          <button
            className={`train-button-red me-2 ${isTraining ? "training" : ""}`}
            title={
              !hasSelectedDataset
                ? "Please select a dataset first"
                : isTraining
                ? "Training in progress"
                : "Train your model"
            }
            onClick={handleTrainModelClick}
            disabled={!hasSelectedDataset || isTraining}
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

          <button
            className={`stop-button-red ${!isTraining ? "disabled" : ""}`}
            title={
              isTraining
                ? "Stop the current training session"
                : "Training is not in progress"
            }
            onClick={handleStopTrainingClick}
            disabled={!isTraining}
          >
            <i className="fas fa-stop-circle"></i>
            <span>Stop</span>
          </button>
        </div>
      </div>
    </nav>
  );
};

export default NavBar;
