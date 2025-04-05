import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import NavBar from "../components/NavBar";
import NetworkVisualization3D from "../components/NetworkVisualization3D";
import "../styles/components/Network3DPage.scss";

const Network3DPage: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [hasData, setHasData] = useState(true);

  // Get data from state passed through navigation
  const { nodes, edges, trainingProgress, isTraining } = location.state || {
    nodes: [],
    edges: [],
    trainingProgress: {
      currentEpoch: 0,
      totalEpochs: 0,
      accuracy: 0,
      loss: 0,
      valAccuracy: 0,
      valLoss: 0,
    },
    isTraining: false,
  };

  // Check if we have valid data
  useEffect(() => {
    if (!location.state || !nodes || nodes.length === 0) {
      console.warn("No network data found in navigation state");
      setHasData(false);
    }
  }, [location.state, nodes]);

  // Handle keyboard events (ESC to exit fullscreen)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isFullscreen) {
        setIsFullscreen(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isFullscreen]);

  // Back to editor button handler
  const handleBackToEditor = () => {
    navigate("/newbuild");
  };

  // Toggle fullscreen
  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  // If no data, show an error message
  if (!hasData) {
    return (
      <div className="network-3d-page">
        <NavBar />
        <div className="network-3d-page-content">
          <div className="no-data-container">
            <h2>No Neural Network Data</h2>
            <p>
              No network data was found. Please create a network first and then
              view it in 3D.
            </p>
            <button className="back-button" onClick={handleBackToEditor}>
              <i className="fas fa-arrow-left"></i> Back to Editor
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`network-3d-page ${isFullscreen ? "fullscreen" : ""}`}>
      {!isFullscreen && <NavBar />}

      <div className="network-3d-page-content">
        <div className="network-3d-header">
          <h2>3D Neural Network Visualization</h2>
          <div className="network-3d-controls">
            <button className="back-button" onClick={handleBackToEditor}>
              <i className="fas fa-arrow-left"></i> Back to Editor
            </button>
            <button className="fullscreen-button" onClick={toggleFullscreen}>
              {isFullscreen ? (
                <>
                  <i className="fas fa-compress"></i> Exit Fullscreen
                </>
              ) : (
                <>
                  <i className="fas fa-expand"></i> Fullscreen
                </>
              )}
            </button>
          </div>
        </div>

        <div className="network-3d-container-wrapper">
          <NetworkVisualization3D
            isActive={true}
            isTraining={isTraining}
            trainingProgress={trainingProgress}
            nodes={nodes}
            edges={edges}
          />

          {isTraining && (
            <div className="training-overlay">
              <div className="training-stats">
                <div className="stat-item">
                  <span>Epoch:</span>
                  <strong>
                    {trainingProgress.currentEpoch}/
                    {trainingProgress.totalEpochs}
                  </strong>
                </div>
                <div className="stat-item">
                  <span>Accuracy:</span>
                  <strong>
                    {(trainingProgress.accuracy * 100).toFixed(2)}%
                  </strong>
                </div>
                <div className="stat-item">
                  <span>Loss:</span>
                  <strong>{trainingProgress.loss.toFixed(4)}</strong>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="network-3d-info">
          <h3>Network Information</h3>
          <div className="info-stats">
            <div className="info-stat">
              <span>Nodes:</span>
              <strong>{nodes.length}</strong>
            </div>
            <div className="info-stat">
              <span>Connections:</span>
              <strong>{edges.length}</strong>
            </div>
            <div className="info-stat">
              <span>Layer Types:</span>
              <strong>
                {Array.from(new Set(nodes.map((node) => node.type))).length}
              </strong>
            </div>
          </div>

          <div className="visualization-help">
            <h4>Navigation Controls</h4>
            <ul>
              <li>
                <strong>Rotate:</strong> Click and drag
              </li>
              <li>
                <strong>Zoom:</strong> Scroll wheel
              </li>
              <li>
                <strong>Pan:</strong> Right-click and drag
              </li>
              <li>
                <strong>Reset View:</strong> Double-click
              </li>
              <li>
                <strong>Fullscreen:</strong> Press the fullscreen button
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Network3DPage;
