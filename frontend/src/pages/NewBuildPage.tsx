import React, { useState, useRef, useEffect, JSX, useCallback } from "react";
import "../styles/components/NewBuildPage.scss";
import { useNewBuildPageContext } from "../context/NewBuildPageContext";
import ReactFlow, {
  addEdge,
  Background,
  Controls,
  applyNodeChanges,
  applyEdgeChanges,
  Node,
  Connection,
  NodeChange,
  EdgeChange,
  OnNodesChange,
  OnEdgesChange,
  Edge,
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
} from "reactflow";
import "reactflow/dist/style.css";
import { io, Socket } from "socket.io-client";
import { Line, Scatter } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";
import {
  DenseNode,
  ConvolutionNode,
  MaxPoolingNode,
  GlobalAveragePoolNode,
  FlattenNode,
  DropoutNode,
  BatchNormalizationNode,
  InputNode,
  OutputNode,
  AttentionNode,
  ResNetBlockNode,
  AddLayerNode,
  ActivationNode,
  CustomBlockNode,
} from "../components/CustomNodes";

import axios from "axios";
import API_BASE_URL from "../utils/apiConfig";
import { getSocketUrl } from "../utils/customDatasetApi";
import CustomLayerModal from "../components/CustomLayerModal";
import CustomDatasetModal from "../components/CustomDatasetModal";
import {
  getCustomDatasets,
  getAvailableDatasets,
} from "../utils/customDatasetApi";
import ToastNotification, { ToastType } from "../components/ToastNotification";
import Validation from "../components/Validation";
void getCustomDatasets;
// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// Define the node types for ReactFlow
const nodeTypes = {
  input: InputNode,
  output: OutputNode,
  dense: DenseNode,
  convolution: ConvolutionNode,
  maxpooling: MaxPoolingNode,
  globalaveragepool: GlobalAveragePoolNode,
  flatten: FlattenNode,
  dropout: DropoutNode,
  batchnormalization: BatchNormalizationNode,
  attention: AttentionNode,
  resnetblock: ResNetBlockNode,
  addlayer: AddLayerNode,
  activation: ActivationNode,
  customblock: CustomBlockNode,
};

// Define the sidebar navigation options
type SidebarOption = "layers" | "templates" | "activations";
type RightSidebarTab = "dataset" | "parameters" | "train";

// Add these near the top of the file with other type definitions
type ValidationErrors = string[];

// Custom Edge with Delete Button
const CustomEdge = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  style = {},
  markerEnd,
  data,
}: any) => {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
  });
  return (
    <>
      <BaseEdge path={edgePath} markerEnd={markerEnd} style={style} />
      <EdgeLabelRenderer>
        <div
          style={{
            position: "absolute",
            transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            pointerEvents: "all",
            zIndex: 1,
          }}
          className="custom-edge-label"
        >
          <button
            className="edge-delete-btn"
            onClick={() => data?.onDelete?.(id)}
            title="Delete edge"
          >
            ×
          </button>
        </div>
      </EdgeLabelRenderer>
    </>
  );
};

const edgeTypes = { custom: CustomEdge };

const NewBuildPage = (): JSX.Element => {
  const {
    nodes,
    setNodes,
    edges,
    setEdges,

    setValidationErrors,
    selectedDataset,
    setSelectedDataset,
    trainingConfig,
    setTrainingConfig,
    isTraining,
    setIsTraining,
    trainingProgress,
    setTrainingProgress,
  } = useNewBuildPageContext();

  // Add trainingComplete state
  const [trainingComplete, setTrainingComplete] = useState<boolean>(false);

  // Socket.io reference
  const socketRef = useRef<Socket | null>(null);

  // Reference to track if visualization was manually selected by user
  const visualizationUserSelected = useRef<boolean>(false);

  // State to track which sidebar option is active
  const [activeSidebarOption, setActiveSidebarOption] =
    useState<SidebarOption>("layers");

  // State to track the selected node for parameter editing
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);

  // State to track the selected visualization
  const [selectedVisualization, setSelectedVisualization] =
    useState<string>("loss"); // Default to loss view instead of accuracy

  // State to control custom layer modal visibility
  const [showCustomLayerModal, setShowCustomLayerModal] =
    useState<boolean>(false);

  // State for custom layers (session-only, not persisted)
  const [customLayers, setCustomLayers] = useState<any[]>([]);

  // State to control custom dataset modal visibility
  const [showCustomDatasetModal, setShowCustomDatasetModal] = useState(false);

  // State for custom datasets
  const [customDatasets, setCustomDatasets] = useState<string[]>([]);

  // State for socket connection
  const [socketConnected, setSocketConnected] = useState(false);
  void socketConnected;
  // Clear custom layers on component unmount (when user navigates away or closes tab)
  useEffect(() => {
    return () => {
      // Cleanup function runs when component unmounts
      setCustomLayers([]);
    };
  }, []);

  // Clear custom layers on page refresh/reload/navigation
  useEffect(() => {
    const handleBeforeUnload = () => {
      // Clear any stored custom layers if they exist
      localStorage.removeItem("neuroblock-custom-layers");
    };

    const handleVisibilityChange = () => {
      // Clear custom layers when page becomes hidden (user switches tabs, minimizes, etc.)
      if (document.visibilityState === "hidden") {
        localStorage.removeItem("neuroblock-custom-layers");
      }
    };

    // Add event listeners
    window.addEventListener("beforeunload", handleBeforeUnload);
    window.addEventListener("pagehide", handleBeforeUnload);
    document.addEventListener("visibilitychange", handleVisibilityChange);

    // Cleanup on component unmount
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
      window.removeEventListener("pagehide", handleBeforeUnload);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      // Final cleanup - remove any custom layers from localStorage
      localStorage.removeItem("neuroblock-custom-layers");
    };
  }, []);

  // Add these state variables with the other useState declarations
  const [showValidationErrors, setShowValidationErrors] =
    useState<boolean>(false);
  void showValidationErrors;
  // Chart data state variables
  const [labels, setLabels] = useState<string[]>([]);
  const [lossData, setLossData] = useState<number[]>([]);
  const [accuracyData, setAccuracyData] = useState<number[]>([]);
  const [valLossData, setValLossData] = useState<number[]>([]);
  const [valAccuracyData, setValAccuracyData] = useState<number[]>([]);

  // Confusion matrix state
  const [confusionMatrix, setConfusionMatrix] = useState<number[][]>([]);
  const [chartKey, setChartKey] = useState<number>(0); // For forcing chart re-renders

  // California Housing specific visualizations
  const [heatmapImage, setHeatmapImage] = useState<string | null>(null);
  const [residuals, setResiduals] = useState<number[]>([]);
  const [predictedValues, setPredictedValues] = useState<number[]>([]);
  const [rmse, setRmse] = useState<number | null>(null);
  const [r2, setR2] = useState<number | null>(null);

  // Chart data objects
  const lossChartData = {
    labels,
    datasets: [
      {
        label: "Loss",
        data: lossData,
        borderColor: "#FF7043",
        backgroundColor: "rgba(255, 99, 132, 0.2)",
        fill: true,
        tension: 0.4,
      },
      {
        label: "Validation Loss",
        data: valLossData,
        borderColor: "#29B6F6",
        backgroundColor: "rgba(54, 162, 235, 0.2)",
        fill: true,
        tension: 0.4,
      },
    ],
  };

  const accuracyChartData = {
    labels,
    datasets: [
      {
        label: "Accuracy",
        data: accuracyData,
        borderColor: "rgba(54, 162, 235, 1)",
        backgroundColor: "rgba(54, 162, 235, 0.2)",
        fill: true,
        tension: 0.4,
      },
      {
        label: "Validation Accuracy",
        data: valAccuracyData,
        borderColor: "rgba(153, 102, 255, 1)",
        backgroundColor: "rgba(153, 102, 255, 0.2)",
        fill: true,
        tension: 0.4,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: true },
      tooltip: { enabled: true },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  // State to track if the model has been saved
  const [isModelSaved, setIsModelSaved] = useState<boolean>(false);

  // State to track layer sizes (input/output dimensions)
  const [layerSizes, setLayerSizes] = useState<
    Record<string, { inputSize: string; outputSize: string }>
  >({});

  // Ref to track if we're in the middle of a save operation
  const isSavingRef = useRef(false);

  // Export functionality state - the export section has been removed from the right sidebar,
  // but these state variables are still needed for export functionality from the navbar
  const [exportStatusMessage, setExportStatusMessage] = useState<string>("");

  // First, let's add a state variable for the W&B URL
  const [wandbUrl, setWandbUrl] = useState<string | null>(null);

  // Reset isModelSaved when nodes or edges change, but only if not currently saving
  useEffect(() => {
    if (!isSavingRef.current) {
      console.log(
        "🔄 Resetting isModelSaved to false due to nodes/edges change"
      );
      setIsModelSaved(false);
    } else {
      console.log(
        "💾 Skipping isModelSaved reset - save operation in progress"
      );
    }
  }, [nodes, edges]);

  // Fetch custom datasets on component mount
  useEffect(() => {
    const fetchCustomDatasets = async () => {
      try {
        console.log("🔍 Fetching available datasets from:", API_BASE_URL);
        // Use getAvailableDatasets to get both built-in and custom datasets
        const availableDatasets = await getAvailableDatasets();
        console.log("📦 Full API response:", availableDatasets);
        console.log("🔧 Built-in datasets:", availableDatasets.built_in);
        console.log("⭐ Custom datasets:", availableDatasets.custom);

        // Extract only custom dataset names for the dropdown
        const customDatasetNames = availableDatasets.custom.map(
          (dataset) => dataset.name
        );
        setCustomDatasets(customDatasetNames);
        console.log("✅ Fetched custom datasets:", customDatasetNames);
      } catch (error) {
        console.error("❌ Failed to fetch custom datasets:", error);
        setCustomDatasets([]);
      }
    };

    fetchCustomDatasets();
  }, []);

  // Function to refresh custom datasets list
  const refreshCustomDatasets = async () => {
    try {
      console.log("🔄 Refreshing available datasets from:", API_BASE_URL);
      const availableDatasets = await getAvailableDatasets();
      console.log("📦 Refresh API response:", availableDatasets);
      console.log(
        "⭐ Custom datasets after refresh:",
        availableDatasets.custom
      );

      const customDatasetNames = availableDatasets.custom.map(
        (dataset) => dataset.name
      );
      setCustomDatasets(customDatasetNames);
      console.log("✅ Refreshed custom datasets:", customDatasetNames);
    } catch (error) {
      console.error("❌ Failed to refresh custom datasets:", error);
    }
  };

  // Function to get visualization options based on dataset
  const getVisualizationOptions = () => {
    // Default options available for all datasets
    const defaultOptions = [{ value: "loss", label: "Loss" }];

    // Classification datasets show accuracy, regression datasets don't
    const isClassificationDataset = [
      "MNIST",
      "CIFAR-10",
      "Iris",
      "Breast Cancer",
    ].includes(selectedDataset || "");

    if (isClassificationDataset) {
      // Add accuracy at the beginning for classification datasets
      defaultOptions.unshift({ value: "accuracy", label: "Accuracy" });

      // If no visualization is explicitly set, default to accuracy for classification
      if (
        selectedVisualization === "loss" &&
        !visualizationUserSelected.current
      ) {
        setTimeout(() => setSelectedVisualization("accuracy"), 0);
      }
    } else {
      // For regression datasets, make sure we're not showing accuracy
      if (selectedVisualization === "accuracy") {
        setTimeout(() => setSelectedVisualization("loss"), 0);
      }
    }

    // Dataset-specific options that are currently supported
    const datasetOptions: Record<string, { value: string; label: string }[]> = {
      MNIST: [{ value: "confusion_matrix", label: "Confusion Matrix" }],
      "CIFAR-10": [{ value: "confusion_matrix", label: "Confusion Matrix" }],
      Iris: [{ value: "confusion_matrix", label: "Confusion Matrix" }],
      "Breast Cancer": [
        { value: "confusion_matrix", label: "Confusion Matrix" },
      ],
      "California Housing": [
        { value: "heatmap", label: "Multicollinearity Heatmap" },
        { value: "residual_plot", label: "Residual Plot" },
      ],
    };

    // Combine default options with dataset-specific options
    const options = selectedDataset
      ? [...defaultOptions, ...(datasetOptions[selectedDataset] || [])]
      : defaultOptions;

    console.log("Visualization options for dataset:", selectedDataset, options);
    return options;
  };

  // Update the socket.io event handling for training progress
  useEffect(() => {
    // Initialize socket connection
    const socketUrl = getSocketUrl();
    console.log("Connecting to socket at:", socketUrl);

    // Detect production (GCP/Cloud Run) environment
    const isProd =
      window.location.hostname.includes("neuroblock.co") ||
      window.location.hostname.includes("run.app");

    socketRef.current = io(socketUrl, {
      transports: isProd ? ["polling"] : ["websocket", "polling"],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
      secure: true,
      rejectUnauthorized: false,
      path: "/socket.io",
      withCredentials: true,
    });

    // Socket event listeners
    socketRef.current.on("connect", () => {
      console.log("Socket connected");
      setSocketConnected(true);
    });

    socketRef.current.on("connect_error", (error) => {
      console.error("Socket connection error:", error);
      setSocketConnected(false);
    });

    socketRef.current.on("disconnect", () => {
      console.log("Socket disconnected");
      setSocketConnected(false);
    });

    // Listen for training start
    socketRef.current.on("training_start", (data) => {
      console.log("Training started:", data.message);
      // Check if we have a W&B URL
      if (data.wandb_run_url) {
        console.log("W&B run URL:", data.wandb_run_url);
        setWandbUrl(data.wandb_run_url);
      }
      // Reset chart data when training starts
      setLabels([]);
      setLossData([]);
      setAccuracyData([]);
      setValLossData([]);
      setValAccuracyData([]);
      setConfusionMatrix([]);

      // Reset training progress
      setTrainingProgress({
        currentEpoch: 0,
        totalEpochs: parseInt(trainingConfig.epochs.toString()),
        accuracy: 0,
        loss: 0,
        valAccuracy: 0,
        valLoss: 0,
      });
    });

    // Listen for real-time training progress
    socketRef.current.on("training_progress", (data) => {
      console.log("Training progress received:", data);

      // Parse numeric values to ensure they're numbers, not strings
      const accuracy =
        typeof data.accuracy === "string"
          ? parseFloat(data.accuracy)
          : data.accuracy;
      const loss =
        typeof data.loss === "string" ? parseFloat(data.loss) : data.loss;
      const valAccuracy =
        typeof data.val_accuracy === "string"
          ? parseFloat(data.val_accuracy)
          : data.val_accuracy;
      const valLoss =
        typeof data.val_loss === "string"
          ? parseFloat(data.val_loss)
          : data.val_loss;

      // Update training progress state with properly parsed values
      setTrainingProgress({
        currentEpoch: data.epoch,
        totalEpochs: data.total_epochs,
        accuracy: accuracy,
        loss: loss,
        valAccuracy: valAccuracy,
        valLoss: valLoss,
      });

      // Update chart data with properly parsed values
      setLabels((prev) => [...prev, `Epoch ${data.epoch}`]);
      setLossData((prev) => [...prev, loss]);
      setAccuracyData((prev) => [...prev, accuracy]);
      setValLossData((prev) => [...prev, valLoss]);
      setValAccuracyData((prev) => [...prev, valAccuracy]);

      // Force chart re-render
      setChartKey((prev) => prev + 1);
    });

    // Listen for training complete
    socketRef.current.on("training_complete", (data) => {
      console.log("Training complete:", data.message);
      console.log(
        "FULL TRAINING COMPLETE DATA:",
        JSON.stringify(data, null, 2)
      );
      setIsTraining(false);
      setTrainingComplete(true); // Set training as complete

      // Check for confusion matrix data
      if (data.metrics && data.metrics.confusion_matrix) {
        console.log(
          "Confusion Matrix received:",
          data.metrics.confusion_matrix
        );
        setConfusionMatrix(data.metrics.confusion_matrix);
      }

      // California Housing specific metrics
      if (selectedDataset === "California Housing") {
        console.log("Processing California Housing specific metrics");

        // Set RMSE and R2 score if available
        if (data.metrics?.rmse) {
          console.log("RMSE received:", data.metrics.rmse);
          setRmse(data.metrics.rmse);
        }

        if (data.metrics?.r2) {
          console.log("R² Score received:", data.metrics.r2);
          setR2(data.metrics.r2);
        }

        // Set residuals and predicted values for residual plot
        if (data.metrics?.residuals_plot) {
          const predictedValues =
            data.metrics.residuals_plot.predicted_values || [];
          const residualsData = data.metrics.residuals_plot.residuals || [];

          console.log(
            "Residuals plot data received - length:",
            `predicted_values: ${predictedValues.length}`,
            `residuals: ${residualsData.length}`
          );

          setPredictedValues(predictedValues);
          setResiduals(residualsData);
        }

        // Set heatmap image if available
        if (data.metrics?.multicollinearity_heatmap) {
          console.log(
            "Multicollinearity heatmap received - data length:",
            data.metrics.multicollinearity_heatmap.length
          );

          const heatmapData = `data:image/png;base64,${data.metrics.multicollinearity_heatmap}`;
          setHeatmapImage(heatmapData);
        }

        // Force visualization update
        if (data.metrics?.multicollinearity_heatmap) {
          setSelectedVisualization("heatmap");
        } else if (data.metrics?.residuals_plot) {
          setSelectedVisualization("residual_plot");
        }

        // Force re-render
        setChartKey((prev) => prev + 1);
      }
    });

    // Listen for training error
    socketRef.current.on("training_error", (data) => {
      console.error("Training error:", data.error);
      console.error("Full error data:", data);
      setIsTraining(false);

      // Show more detailed error information
      const errorMessage = data.error || "Unknown training error";
      console.error("Detailed error for manual architecture:", {
        error: errorMessage,
        currentNodes: nodes.length,
        currentEdges: edges.length,
        activationLayers: nodes
          .filter((n) => n.type === "activation")
          .map((n) => ({
            id: n.id,
            function: n.data.function,
            data: n.data,
          })),
      });

      showToast(`Training error: ${errorMessage}`, "error");
    });

    // Cleanup Socket.IO connection on component unmount
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, [trainingConfig, nodes, edges]);

  // Debug log for selectedDataset changes
  useEffect(() => {
    console.log("Selected dataset changed to:", selectedDataset || "none");
  }, [selectedDataset]);

  // Listen for changes to isTraining and selectedDataset and emit events
  useEffect(() => {
    const event = new CustomEvent("trainingStateChange", {
      detail: { isTraining },
    });
    window.dispatchEvent(event);
  }, [isTraining]);

  useEffect(() => {
    console.log("Dataset changed in NewBuildPage:", selectedDataset);

    const event = new CustomEvent("datasetChange", {
      detail: { dataset: selectedDataset },
    });

    console.log("Dispatching datasetChange event with data:", event.detail);
    window.dispatchEvent(event);
  }, [selectedDataset]);

  // Add event listener for dataset selection check
  useEffect(() => {
    const handleCheckDataset = (event: CustomEvent) => {
      console.log("Received checkDatasetSelected event");
      const callback = event.detail.callback;

      // Make a comprehensive check
      const hasDataset = !!selectedDataset && selectedDataset.trim() !== "";
      console.log("Current dataset state:", {
        selectedDataset,
        hasDataset,
        numberOfNodes: nodes.length,
        numberOfEdges: edges.length,
        modelIsReady: hasDataset && nodes.length >= 2 && edges.length > 0,
      });

      if (typeof callback === "function") {
        // Pass back both the boolean result and the dataset name
        callback(hasDataset, selectedDataset || "none");
      }
    };

    window.addEventListener(
      "checkDatasetSelected",
      handleCheckDataset as EventListener
    );

    return () => {
      window.removeEventListener(
        "checkDatasetSelected",
        handleCheckDataset as EventListener
      );
    };
  }, [selectedDataset, nodes, edges]);

  // Add an event listener for the save model event
  useEffect(() => {
    const saveModelHandler = (event: CustomEvent) => {
      // Log detailed diagnostic information
      const eventDetails = event.detail || {
        source: "unknown",
        timestamp: new Date().toISOString(),
      };
      console.log(
        `Save model event received in NewBuildPage at ${new Date().toISOString()}.`,
        {
          eventSource: eventDetails.source,
          eventTimestamp: eventDetails.timestamp,
          timeDifference: eventDetails.timestamp
            ? new Date().getTime() - new Date(eventDetails.timestamp).getTime()
            : "unknown",
          currentDataset: selectedDataset,
          hasDataset: !!selectedDataset && selectedDataset.trim() !== "",
          modelState: {
            nodes: nodes.length,
            edges: edges.length,
            hasInputLayer: nodes.some((node) => node.type === "input"),
            hasOutputLayer: nodes.some((node) => node.type === "output"),
          },
        }
      );

      if (!selectedDataset || selectedDataset === "") {
        console.error("Error: No dataset selected when trying to save");
        showToast(
          "No dataset selected! Please select a dataset before saving.",
          "error"
        );
        return;
      }

      // Pre-validate to check if there are any issues
      const validationIssues = preValidateModel();
      if (validationIssues) {
        console.warn("Pre-validation failed:", validationIssues);
        showToast(`Cannot save model: ${validationIssues}`, "error");
        return;
      }

      console.log("Dataset is selected, proceeding with save");
      handleSaveModel();
    };

    window.addEventListener("saveModel", saveModelHandler as EventListener);

    // Cleanup function
    return () => {
      window.removeEventListener(
        "saveModel",
        saveModelHandler as EventListener
      );
    };
  }, [selectedDataset, nodes, edges]); // Include all relevant dependencies

  // Pre-validation to check basic requirements before attempting to save or train
  const preValidateModel = (): string | null => {
    console.log("Pre-validating model");

    // Check if dataset is selected
    if (!selectedDataset) {
      return "Please select a dataset first";
    }

    // Check if we have nodes
    if (nodes.length < 2) {
      return "Your model needs at least an input and output layer";
    }

    // Check if input and output layers exist
    const inputLayer = nodes.find((node) => node.type === "input");
    const outputLayer = nodes.find((node) => node.type === "output");

    if (!inputLayer) {
      return "Your model needs an input layer";
    }

    if (!outputLayer) {
      return "Your model needs an output layer";
    }

    // Check connections
    if (edges.length === 0) {
      return "Your model needs connections between layers";
    }

    // Check if input is connected
    const isInputConnected = edges.some(
      (edge) => edge.source === inputLayer.id
    );
    if (!isInputConnected) {
      return "Input layer must be connected to other layers";
    }

    // Check if output is connected
    const isOutputConnected = edges.some(
      (edge) => edge.target === outputLayer.id
    );
    if (!isOutputConnected) {
      return "Output layer must be connected to other layers";
    }

    // Check for disconnected nodes
    const connectedNodeIds = new Set<string>();
    edges.forEach((edge) => {
      connectedNodeIds.add(edge.source);
      connectedNodeIds.add(edge.target);
    });

    const disconnectedNodes = nodes.filter(
      (node) => !connectedNodeIds.has(node.id)
    );
    if (disconnectedNodes.length > 0) {
      return `Some layers are disconnected: ${disconnectedNodes
        .map((n) => n.id)
        .join(", ")}`;
    }

    // For classification datasets, check that a Softmax activation is connected to the output
    if (
      (selectedDataset === "MNIST" ||
        selectedDataset === "CIFAR-10" ||
        selectedDataset === "Iris" ||
        selectedDataset === "Breast Cancer") &&
      outputLayer
    ) {
      // Find all Softmax activation layers
      const softmaxLayers = nodes.filter(
        (node) => node.type === "activation" && node.data.function === "softmax"
      );

      // Build a connectivity graph to check for both direct and indirect connections
      const graph = new Map<string, Set<string>>();

      // Initialize graph with all nodes
      nodes.forEach((node) => {
        graph.set(node.id, new Set<string>());
      });

      // Add connections to the graph (in both directions for undirected traversal)
      edges.forEach((edge) => {
        if (graph.has(edge.source)) {
          graph.get(edge.source)!.add(edge.target);
        }
        if (graph.has(edge.target)) {
          graph.get(edge.target)!.add(edge.source);
        }
      });

      // Check if there's any path between output layer and any Softmax activation layer
      let isSoftmaxConnected = false;

      // Breadth-first search to find connections
      const bfs = (startNodeId: string) => {
        const queue: string[] = [startNodeId];
        const visited = new Set<string>([startNodeId]);

        while (queue.length > 0) {
          const currentNodeId = queue.shift()!;

          // Check if this is a Softmax activation layer
          if (softmaxLayers.some((node) => node.id === currentNodeId)) {
            isSoftmaxConnected = true;
            break;
          }

          // Add connected nodes to the queue
          const neighbors = graph.get(currentNodeId) || new Set<string>();
          neighbors.forEach((neighborId) => {
            if (!visited.has(neighborId)) {
              visited.add(neighborId);
              queue.push(neighborId);
            }
          });
        }
      };

      // Start BFS from the output layer
      bfs(outputLayer.id);

      // Also check the reverse: if any Softmax layer is connected to the output
      if (!isSoftmaxConnected && softmaxLayers.length > 0) {
        for (const layer of softmaxLayers) {
          if (!isSoftmaxConnected) {
            const visited = new Set<string>();

            // Depth-first search to see if this Softmax layer connects to output
            const dfs = (nodeId: string) => {
              if (visited.has(nodeId)) return;
              visited.add(nodeId);

              if (nodeId === outputLayer.id) {
                isSoftmaxConnected = true;
                return;
              }

              const neighbors = graph.get(nodeId) || new Set<string>();
              neighbors.forEach((neighborId) => {
                if (!isSoftmaxConnected) {
                  dfs(neighborId);
                }
              });
            };

            dfs(layer.id);
          }
        }
      }
    }

    // For regression datasets, ensure no Softmax activation is connected to output
    if (selectedDataset === "California Housing" && outputLayer) {
      // Find all Softmax activation layers
      const softmaxLayers = nodes.filter(
        (node) => node.type === "activation" && node.data.function === "softmax"
      );

      if (softmaxLayers.length > 0) {
        // Build a connectivity graph
        const graph = new Map<string, Set<string>>();

        // Initialize graph
        nodes.forEach((node) => {
          graph.set(node.id, new Set<string>());
        });

        // Add connections
        edges.forEach((edge) => {
          if (graph.has(edge.source)) {
            graph.get(edge.source)!.add(edge.target);
          }
          if (graph.has(edge.target)) {
            graph.get(edge.target)!.add(edge.source);
          }
        });

        // Check for connection between output and Softmax
        let hasSoftmaxConnection = false;

        // BFS from output
        const queue: string[] = [outputLayer.id];
        const visited = new Set<string>([outputLayer.id]);

        while (queue.length > 0 && !hasSoftmaxConnection) {
          const currentNodeId = queue.shift()!;

          if (softmaxLayers.some((node) => node.id === currentNodeId)) {
            hasSoftmaxConnection = true;
            break;
          }

          const neighbors = graph.get(currentNodeId) || new Set<string>();
          neighbors.forEach((neighborId) => {
            if (!visited.has(neighborId)) {
              visited.add(neighborId);
              queue.push(neighborId);
            }
          });
        }

        if (hasSoftmaxConnection) {
          return "Regression tasks should not use Softmax activation in the output layer";
        }
      }
    }

    console.log("Basic pre-validation passed");
    return null;
  };

  // Function to normalize dataset name for the backend
  const normalizeDatasetName = (datasetName: string): string => {
    // Convert dataset names to lowercase and replace spaces with underscores
    const normalized = datasetName.toLowerCase().replace(/\s+/g, "_");

    // Map specific dataset names to their expected backend format
    const datasetMapping: Record<string, string> = {
      mnist: "MNIST",
      "cifar-10": "CIFAR-10",
      iris: "Iris",
      breast_cancer: "Breast Cancer",
      california_housing: "California Housing",
    };

    return datasetMapping[normalized] || normalized;
  };

  // Helper function to get class labels for different datasets
  const getClassLabels = (datasetName: string): string[] => {
    switch (datasetName) {
      case "MNIST":
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
      case "CIFAR-10":
        return [
          "airplane",
          "automobile",
          "bird",
          "cat",
          "deer",
          "dog",
          "frog",
          "horse",
          "ship",
          "truck",
        ];
      case "Iris":
        return ["Setosa", "Versicolor", "Virginica"];
      case "Breast Cancer":
        return ["Benign", "Malignant"];
      case "California Housing":
        return ["Price"];
      default:
        return [];
    }
  };

  // Unified function to prepare architecture payload for backend
  const prepareArchitecturePayload = (
    currentNodes: Node[],
    currentEdges: Edge[]
  ) => {
    console.log("🔄 Preparing unified architecture payload...");

    // Step 1: Always analyze and sequence the architecture (for both template and manual)
    const { sequencedNodes, sequencedEdges } = analyzeAndSequenceArchitecture(
      currentNodes,
      currentEdges
    );

    console.log("📋 Architecture Analysis:");
    console.log(
      "  Nodes:",
      sequencedNodes.map((n) => `${n.id}(${n.type})`).join(" → ")
    );
    console.log(
      "  Edges:",
      sequencedEdges.map((e) => `${e.source}→${e.target}`).join(", ")
    );

    // Step 2: Validate and normalize all nodes
    const validatedNodes = sequencedNodes.map((node) => {
      const nodeType = node.type?.toLowerCase() || "unknown";

      // Create clean node structure
      let cleanNode = {
        id: node.id,
        type: node.type || "unknown",
        position: node.position,
        data: { ...node.data },
      };

      // Layer-specific validation and normalization
      switch (nodeType) {
        case "activation":
          // Ensure activation layers have proper function
          let activationFunction = node.data.function;
          const validActivations = [
            "relu",
            "sigmoid",
            "tanh",
            "softmax",
            "leaky_relu",
            "elu",
            "swish",
            "linear",
          ];

          if (
            !activationFunction ||
            typeof activationFunction !== "string" ||
            activationFunction === "activation" ||
            !validActivations.includes(activationFunction.toLowerCase())
          ) {
            console.warn(
              `⚠️ Invalid activation function "${activationFunction}" for ${node.id}, defaulting to "relu"`
            );
            activationFunction = "relu";
          }

          cleanNode.data = {
            label: node.data.label || "Activation Layer",
            function: activationFunction.toLowerCase(),
          };
          break;

        case "dense":
          cleanNode.data = {
            label: node.data.label || "Dense Layer",
            neurons: parseInt(node.data.neurons) || 64,
            activation: node.data.activation || "None",
          };
          break;

        case "convolution":
          cleanNode.data = {
            label: node.data.label || "Convolution Layer",
            filters: parseInt(node.data.filters) || 32,
            kernelSize: Array.isArray(node.data.kernelSize)
              ? node.data.kernelSize
              : [3, 3],
            stride: Array.isArray(node.data.stride) ? node.data.stride : [1, 1],
            padding: node.data.padding || "same",
            activation: node.data.activation || "None",
          };
          break;

        case "maxpooling":
          cleanNode.data = {
            label: node.data.label || "MaxPooling Layer",
            poolSize: Array.isArray(node.data.poolSize)
              ? node.data.poolSize
              : [2, 2],
            stride: Array.isArray(node.data.stride) ? node.data.stride : [2, 2],
            padding: node.data.padding || "valid",
          };
          break;

        case "dropout":
          cleanNode.data = {
            label: node.data.label || "Dropout Layer",
            rate: parseFloat(node.data.rate) || 0.2,
          };
          break;

        case "batchnormalization":
          cleanNode.data = {
            label: node.data.label || "BatchNorm Layer",
            momentum: parseFloat(node.data.momentum) || 0.99,
            epsilon: parseFloat(node.data.epsilon) || 0.001,
          };
          break;

        case "attention":
          cleanNode.data = {
            label: node.data.label || "Attention Layer",
            heads: parseInt(node.data.heads) || 8,
            keyDim: parseInt(node.data.keyDim) || 64,
            dropout: parseFloat(node.data.dropout) || 0.0,
          };
          break;

        case "output":
          cleanNode.data = {
            label: node.data.label || "Output Layer",
            activation: node.data.activation || "None",
          };
          break;

        case "input":
          cleanNode.data = {
            label: node.data.label || "Input Layer",
          };
          break;

        case "flatten":
          cleanNode.data = {
            label: node.data.label || "Flatten Layer",
          };
          break;

        case "globalaveragepool":
          cleanNode.data = {
            label: node.data.label || "GlobalAvgPool Layer",
          };
          break;

        case "addlayer":
          cleanNode.data = {
            label: node.data.label || "Add Layer",
          };
          break;

        case "resnetblock":
          cleanNode.data = {
            label: node.data.label || "ResNet Block",
            blockType: node.data.blockType || "Basic",
            filters: parseInt(node.data.filters) || 64,
            stride: Array.isArray(node.data.stride) ? node.data.stride : [1, 1],
            activation: node.data.activation || "ReLU",
          };
          break;

        case "customblock":
          cleanNode.data = {
            label: node.data.label || "Custom Block",
            blockName: node.data.blockName || "Custom Block",
            layers: node.data.layers || [],
          };
          break;

        default:
          // Keep original data for unknown types
          console.warn(`⚠️ Unknown layer type: ${nodeType}`);
          break;
      }

      return cleanNode;
    });

    // Step 3: Create clean edges structure
    const cleanEdges = sequencedEdges.map((edge, index) => ({
      id: edge.id || `e${index + 1}`,
      source: edge.source,
      target: edge.target,
    }));

    // Step 4: Log validation results
    const activationLayers = validatedNodes.filter(
      (n) => n.type === "activation"
    );
    console.log(
      "✅ Validated activation layers:",
      activationLayers.map((n) => ({ id: n.id, function: n.data.function }))
    );

    return {
      nodes: validatedNodes,
      edges: cleanEdges,
    };
  };

  const handleSaveModel = async (): Promise<void> => {
    if (!selectedDataset) {
      setValidationErrors(["Please select a dataset first"]);
      return;
    }

    // Set saving flag to prevent isModelSaved from being reset
    isSavingRef.current = true;

    // Get the latest training config from localStorage
    const storedConfig = localStorage.getItem("trainingConfig");
    const latestTrainingConfig = storedConfig
      ? JSON.parse(storedConfig)
      : trainingConfig;

    console.log("💾 Saving model with unified payload system...");

    // Double check if dataset is selected
    if (!selectedDataset || selectedDataset.trim() === "") {
      console.error("Error: No dataset selected in handleSaveModel");
      showToast(
        "No dataset selected! Please select a dataset before saving.",
        "error"
      );
      isSavingRef.current = false; // Reset flag on error
      return;
    }

    // Get validation issues but don't block saving on warnings
    const parameterErrors = validateLayerParameters();
    const sizeErrors = validateInputOutputSizes();
    const errors = [...parameterErrors, ...sizeErrors];
    console.log("Parameter validation results:", parameterErrors);
    console.log("Size validation results:", sizeErrors);
    console.log("Combined validation results:", errors);

    // Filter critical errors (those marked with [Critical])
    const criticalErrors = errors.filter((error) =>
      error.includes("[Critical]")
    );

    if (criticalErrors.length > 0) {
      console.warn("Critical validation errors found:", criticalErrors);
      setValidationErrors(errors);
      displayValidationErrors(errors);

      // Show only critical errors in the toast, without the [Critical] prefix
      const formattedErrors = criticalErrors.map((err) =>
        err.replace("[Critical] ", "")
      );
      showToast(
        `Cannot save model due to critical issues: ${formattedErrors.join(
          "; "
        )}`,
        "error"
      );
      isSavingRef.current = false; // Reset flag on error
      return;
    }

    try {
      // Show saving indicator
      const saveButton = document.querySelector(
        ".save-button"
      ) as HTMLButtonElement;
      if (saveButton) {
        saveButton.innerHTML =
          '<i class="fas fa-spinner fa-spin"></i> Saving...';
        saveButton.disabled = true;
      }

      // 🚀 Use unified payload preparation (works for both template and manual)
      const { nodes: validatedNodes, edges: cleanEdges } =
        prepareArchitecturePayload(nodes, edges);

      // Prepare model architecture and training configuration
      const modelData = {
        architecture: {
          nodes: validatedNodes,
          edges: cleanEdges,
          dataset: normalizeDatasetName(selectedDataset),
        },
        training_config: {
          epochs: latestTrainingConfig.epochs,
          batchSize: latestTrainingConfig.batchSize,
          optimizer: latestTrainingConfig.optimizer,
          lossFunction: latestTrainingConfig.lossFunction,
          learningRate: latestTrainingConfig.learningRate,
          validationSplit: latestTrainingConfig.validationSplit,
        },
      };

      console.log("📦 Unified model payload:", modelData);

      console.log("Sending model data:", modelData);

      // Make a POST request to the backend
      const response = await fetch(`${API_BASE_URL}/api/save_model`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(modelData),
      });

      // Reset button state
      if (saveButton) {
        saveButton.innerHTML = '<i class="fas fa-save"></i> Save Model';
        saveButton.disabled = false;
      }

      if (!response.ok) {
        const errorData = await response.json();
        console.error("Backend Error:", errorData);

        // Display a more detailed error message
        const errorMessage =
          errorData.error || "Unknown error occurred while saving the model";
        showToast(`Saving failed: ${errorMessage}`, "error");
        isSavingRef.current = false; // Reset flag on error
        return;
      }

      const data = await response.json();
      console.log("Response from Backend:", data);

      // Set model as saved if successful
      if (
        data.message &&
        (data.message.includes("saved successfully") ||
          data.message.includes("Model architecture saved"))
      ) {
        console.log(
          "✅ Model saved successfully, setting isModelSaved to true"
        );
        setIsModelSaved(true);

        // Display success message
        showToast(
          "Model architecture saved successfully! Note: This only saves the model structure.",
          "success"
        );
      } else {
        // Display the exact message from the backend
        showToast(
          "Model architecture saved successfully! Note: This only saves the model structure.",
          "info"
        );
      }

      // Reset saving flag after successful save
      isSavingRef.current = false;
    } catch (error) {
      console.error("Error saving model:", error);

      // Reset button state
      const saveButton = document.querySelector(
        ".save-button"
      ) as HTMLButtonElement;
      if (saveButton) {
        saveButton.innerHTML = '<i class="fas fa-save"></i> Save Model';
        saveButton.disabled = false;
      }

      // Display a more detailed error message
      showToast(
        `An error occurred while saving the model: ${
          error instanceof Error ? error.message : String(error)
        }`,
        "error"
      );

      // Reset saving flag on error
      isSavingRef.current = false;
    }
  };

  // Start training the model
  const handleStartTraining = async () => {
    // Check if model is saved first
    console.log(
      "🔍 Checking isModelSaved in handleStartTraining:",
      isModelSaved
    );
    if (!isModelSaved) {
      console.log("❌ Model not saved, preventing training");
      showToast(
        "Please save the model before training. Click 'Save Model' first.",
        "error"
      );
      return;
    }

    console.log("✅ Model is saved, proceeding with training");
    // Skip full validation and just check critical requirements using preValidateModel
    const validationIssue = preValidateModel();
    if (validationIssue) {
      console.warn("Cannot start training:", validationIssue);
      showToast(`Cannot start training: ${validationIssue}`, "error");
      return;
    }

    if (!selectedDataset) {
      showToast(
        "No dataset selected! Please select a dataset before training.",
        "error"
      );
      return;
    }

    // Get the latest training config from localStorage
    const storedConfig = localStorage.getItem("trainingConfig");
    const latestTrainingConfig = storedConfig
      ? JSON.parse(storedConfig)
      : trainingConfig;

    // Set training state
    setIsTraining(true);

    // Reset training progress and chart data
    setTrainingProgress({
      currentEpoch: 0,
      totalEpochs: parseInt(latestTrainingConfig.epochs.toString()),
      accuracy: 0,
      loss: 0,
      valAccuracy: 0,
      valLoss: 0,
    });

    // Reset chart data
    setLabels([]);
    setLossData([]);
    setAccuracyData([]);
    setValLossData([]);
    setValAccuracyData([]);

    // Reset confusion matrix
    setConfusionMatrix([]);

    // Reset California Housing specific visualizations
    if (selectedDataset === "California Housing") {
      setHeatmapImage(null);
      setResiduals([]);
      setPredictedValues([]);
      setRmse(null);
      setR2(null);
    }

    // Prepare training payload with proper type conversions
    const trainingPayload = {
      dataset: normalizeDatasetName(selectedDataset),
      lossFunction: latestTrainingConfig.lossFunction,
      optimizer: latestTrainingConfig.optimizer.toLowerCase(),
      batchSize: parseInt(latestTrainingConfig.batchSize.toString()),
      epochs: parseInt(latestTrainingConfig.epochs.toString()),
      learningRate: parseFloat(latestTrainingConfig.learningRate.toString()),
    };

    console.log("Training payload sent:", trainingPayload);

    // Emit the start_training event
    if (socketRef.current) {
      socketRef.current.emit("start_training", trainingPayload);
      console.log("Start training event emitted");
    } else {
      console.error("Socket connection not established");
      showToast(
        "Socket connection not established. Please refresh the page and try again.",
        "error"
      );
      setIsTraining(false);
      return;
    }

    // Show success message
    console.log("Training started successfully!");
  };

  // Function to handle stopping the training
  const handleStopTraining = (): void => {
    if (socketRef.current) {
      socketRef.current.emit("stop_training");
      console.log("Stop training signal sent");
      setIsTraining(false);
      setWandbUrl(null); // Clear W&B URL when stopping training
      showToast("Training stopped by user", "info");
    }
  };

  // Function to handle back button click in visualization section

  // Function to handle custom layer save
  const handleCustomLayerSave = (blockName: string, layers: any[]) => {
    // Initialize layers with default parameters to match backend expectations
    const layersWithDefaults = layers.map((layer) => {
      const layerType = layer.id.toLowerCase(); // Use id field for layer type

      // Define default parameters for each layer type (matching backend expectations)
      const defaultParameters: Record<string, any> = {
        dense: { neurons: 64, activation: "relu" },
        convolution: {
          filters: 32,
          kernelSize: [3, 3],
          stride: [1, 1],
          padding: "same",
          activation: "relu",
        },
        maxpooling: { poolSize: [2, 2], stride: [2, 2], padding: "same" },
        globalaveragepool: {},
        flatten: {},
        dropout: { rate: 0.25 },
        batchnormalization: { momentum: 0.99, epsilon: 0.001 },
        attention: { heads: 8, keyDim: 64, dropout: 0.0 },
        addlayer: {},
        activation: { function: "relu" },
      };

      return {
        ...layer,
        parameters: defaultParameters[layerType] || {},
      };
    });

    // Create the custom layer object
    const customLayer = {
      id: `custom-${blockName
        .toLowerCase()
        .replace(/\s+/g, "-")}-${Date.now()}`,
      name: blockName.trim(),
      layers: layersWithDefaults,
      icon: "fas fa-cube",
      type: "CustomBlock",
    };

    // Add to custom layers list
    setCustomLayers((prevLayers) => [...prevLayers, customLayer]);

    // Create a new custom block node on the canvas
    addLayer("customblock", {
      blockName,
      layers: layersWithDefaults,
      customBlock: true,
    });

    // Close the modal
    setShowCustomLayerModal(false);

    // Show success message
    showToast(
      `Custom layer "${blockName}" created and added to available layers!\n\nNote: Custom layers are only available during this session and will be cleared when you refresh or close the page.`,
      "success"
    );
  };

  // Function to add a new layer
  const addLayer = (type: string, extraParams?: Record<string, any>): Node => {
    // Get count of existing layers of this type to create a unique name
    const existingLayersOfType = nodes.filter(
      (node) => node.type?.toLowerCase() === type.toLowerCase()
    ).length;

    const layerNumber = existingLayersOfType + 1;

    // Create a formatted layer name
    const getDefaultLayerName = (type: string): string => {
      switch (type.toLowerCase()) {
        case "input":
          return `Input Layer ${layerNumber}`;
        case "dense":
          return `Dense Layer ${layerNumber}`;
        case "convolution":
          return `Conv2D Layer ${layerNumber}`;
        case "maxpooling":
          return `MaxPool2D Layer ${layerNumber}`;
        case "globalaveragepool":
          return `GlobalAvgPool Layer ${layerNumber}`;
        case "flatten":
          return `Flatten Layer ${layerNumber}`;
        case "dropout":
          return `Dropout Layer ${layerNumber}`;
        case "batchnormalization":
          return `BatchNorm Layer ${layerNumber}`;
        case "attention":
          return `Attention Layer ${layerNumber}`;
        case "resnetblock":
          return `ResNet Block ${layerNumber}`;
        case "output":
          return `Output Layer ${layerNumber}`;
        case "addlayer":
          return `Add Layer Node ${layerNumber}`;
        case "activation":
          return `Activation Layer ${layerNumber}`;
        case "customblock":
          return extraParams?.blockName || `Custom Block ${layerNumber}`;
        default:
          return `${type} Layer ${layerNumber}`;
      }
    };

    const defaultParams: Record<string, any> = {
      input: {}, // Input layer has no parameters
      output: { activation: "None" },
      dense: { neurons: 64, activation: "None" },
      convolution: { filters: 32, kernelSize: [3, 3], stride: [1, 1] },
      maxpooling: { poolSize: [2, 2], stride: [2, 2], padding: "none" },
      globalaveragepool: {}, // Global average pooling has no parameters
      flatten: {},
      dropout: { rate: 0.2 },
      batchnormalization: { momentum: 0.99, epsilon: 0.001 },
      attention: { heads: 8, keyDim: 64, dropout: 0.0 },
      resnetblock: {
        blockType: "Basic",
        filters: 64,
        stride: [1, 1],
        activation: "ReLU",
      },
      addlayer: {}, // Add Layer has no parameters
      activation: { function: "relu" }, // Default activation function
      customblock: { blockName: "Custom Block", layers: [] }, // Default custom block
    };

    // Merge the default parameters with any extra parameters
    const mergedParams = {
      ...defaultParams[type.toLowerCase()],
      ...extraParams,
    };

    const newNode: Node = {
      id: `${type}-${Date.now()}`,
      data: {
        label: getDefaultLayerName(type),
        ...mergedParams,
      },
      position: { x: Math.random() * 600, y: Math.random() * 400 },
      type: type.toLowerCase(),
    };

    setNodes((nds) => [...nds, newNode]);

    // Return the new node
    return newNode;
  };

  // Function to load a template
  const loadTemplate = (templateKey: string): void => {
    const templates: Record<string, Node[]> = {
      "Simple Feedforward": [
        {
          id: "Input-1",
          data: { label: "Input Layer" },
          position: { x: 100, y: 100 },
          type: "input",
        },
        {
          id: "Dense-1",
          data: {
            label: "Hidden Dense Layer",
            neurons: 64,
            activation: "None",
          },
          position: { x: 300, y: 200 },
          type: "dense",
        },
        {
          id: "Activation-1",
          data: {
            label: "Activation Layer 1",
            function: "relu",
          },
          position: { x: 300, y: 250 },
          type: "activation",
        },
        {
          id: "Output-1",
          data: { label: "Output Layer", activation: "None" },
          position: { x: 500, y: 300 },
          type: "output",
        },
        {
          id: "Activation-2",
          data: {
            label: "Activation Layer 2",
            function: "softmax",
          },
          position: { x: 500, y: 350 },
          type: "activation",
        },
      ],
      CNN: [
        {
          id: "Input-1",
          data: { label: "Input Layer" },
          position: { x: 100, y: 100 },
          type: "input",
        },
        {
          id: "Conv-1",
          data: {
            label: "Convolution Layer",
            filters: 32,
            kernelSize: [3, 3],
            stride: [1, 1],
            activation: "None",
          },
          position: { x: 300, y: 200 },
          type: "convolution",
        },
        {
          id: "Activation-1",
          data: {
            label: "Activation Layer 1",
            function: "relu",
          },
          position: { x: 300, y: 250 },
          type: "activation",
        },
        {
          id: "MaxPool-1",
          data: {
            label: "MaxPooling Layer",
            poolSize: [2, 2],
            stride: [2, 2],
            padding: "valid",
          },
          position: { x: 500, y: 300 },
          type: "maxpooling",
        },
        {
          id: "Flatten-1",
          data: { label: "Flatten Layer" },
          position: { x: 700, y: 400 },
          type: "flatten",
        },
        {
          id: "Dense-1",
          data: { label: "Dense Layer", neurons: 64, activation: "None" },
          position: { x: 900, y: 500 },
          type: "dense",
        },
        {
          id: "Activation-2",
          data: {
            label: "Activation Layer 2",
            function: "relu",
          },
          position: { x: 900, y: 550 },
          type: "activation",
        },
        {
          id: "Output-1",
          data: { label: "Output Layer", activation: "None" },
          position: { x: 1100, y: 600 },
          type: "output",
        },
        {
          id: "Activation-3",
          data: {
            label: "Activation Layer 3",
            function: "softmax",
          },
          position: { x: 1100, y: 650 },
          type: "activation",
        },
      ],

      // New Transformer Encoder template with standard architecture and skip connections
      "Transformer Encoder": [
        // Input layer
        {
          id: "Input-1",
          data: { label: "Input Layer" },
          position: { x: 150, y: 250 },
          type: "input",
        },

        // First normalization layer (LN1)
        {
          id: "LayerNorm-1",
          data: {
            label: "Layer Normalization 1",
            momentum: 0.99,
            epsilon: 0.001,
          },
          position: { x: 350, y: 150 },
          type: "batchnormalization",
        },

        // Multi-head attention
        {
          id: "Attention-1",
          data: {
            label: "Multi-Head Attention",
            heads: 8,
            keyDim: 64,
            dropout: 0.1,
          },
          position: { x: 550, y: 150 },
          type: "attention",
        },

        // First Add layer (for skip connection after attention)
        {
          id: "Add-1",
          data: {
            label: "Add Layer 1",
          },
          position: { x: 750, y: 150 },
          type: "addlayer",
        },

        // Second normalization layer (LN2)
        {
          id: "LayerNorm-2",
          data: {
            label: "Layer Normalization 2",
            momentum: 0.99,
            epsilon: 0.001,
          },
          position: { x: 950, y: 150 },
          type: "batchnormalization",
        },

        // First dense layer of FFN
        {
          id: "Dense-1",
          data: {
            label: "FFN - Dense 1",
            neurons: 512,
            activation: "None",
          },
          position: { x: 950, y: 250 },
          type: "dense",
        },

        {
          id: "Activation-1",
          data: {
            label: "FFN - Activation",
            function: "relu",
          },
          position: { x: 950, y: 300 },
          type: "activation",
        },

        // Optional dropout for FFN
        {
          id: "Dropout-1",
          data: {
            label: "FFN - Dropout",
            rate: 0.1,
          },
          position: { x: 950, y: 350 },
          type: "dropout",
        },

        // Second dense layer of FFN
        {
          id: "Dense-2",
          data: {
            label: "FFN - Dense 2",
            neurons: 256,
            activation: "None",
          },
          position: { x: 750, y: 350 },
          type: "dense",
        },

        // Second Add layer (for skip connection after FFN)
        {
          id: "Add-2",
          data: {
            label: "Add Layer 2",
          },
          position: { x: 550, y: 350 },
          type: "addlayer",
        },

        // Output layer
        {
          id: "Output-1",
          data: {
            label: "Output Layer",
            activation: "None",
          },
          position: { x: 350, y: 350 },
          type: "output",
        },

        // Output activation
        {
          id: "Activation-2",
          data: {
            label: "Output Activation",
            function: "softmax",
          },
          position: { x: 350, y: 450 },
          type: "activation",
        },
      ],
      "Fully Connected Regression": [
        {
          id: "Input-1",
          data: { label: "Input Layer" },
          position: { x: 100, y: 100 },
          type: "input",
        },
        {
          id: "Dense-1",
          data: { label: "Dense Layer", neurons: 64, activation: "None" },
          position: { x: 300, y: 200 },
          type: "dense",
        },
        {
          id: "Activation-1",
          data: {
            label: "Activation Layer 1",
            function: "relu",
          },
          position: { x: 300, y: 250 },
          type: "activation",
        },
        {
          id: "Dense-2",
          data: { label: "Dense Layer", neurons: 32, activation: "None" },
          position: { x: 500, y: 300 },
          type: "dense",
        },
        {
          id: "Activation-2",
          data: {
            label: "Activation Layer 2",
            function: "relu",
          },
          position: { x: 500, y: 350 },
          type: "activation",
        },
        {
          id: "Output-1",
          data: { label: "Output Layer", activation: "None" },
          position: { x: 700, y: 400 },
          type: "output",
        },
      ],
      Blank: [
        {
          id: "Input-1",
          data: { label: "Input Layer" },
          position: { x: 100, y: 100 },
          type: "input",
        },
        {
          id: "Output-1",
          data: { label: "Output Layer" },
          position: { x: 700, y: 300 },
          type: "output",
        },
      ],
      "Convolutional Network": [
        {
          id: "Input-1",
          data: { label: "Input Layer" },
          position: { x: 100, y: 100 },
          type: "input",
        },
        {
          id: "Conv-1",
          data: {
            label: "Convolution Layer",
            filters: 32,
            kernelSize: [3, 3],
            stride: [1, 1],
            activation: "None",
          },
          position: { x: 300, y: 200 },
          type: "convolution",
        },
        {
          id: "Activation-1",
          data: {
            label: "Activation Layer 1",
            function: "relu",
          },
          position: { x: 300, y: 250 },
          type: "activation",
        },
        {
          id: "MaxPool-1",
          data: {
            label: "MaxPooling Layer",
            poolSize: [2, 2],
            stride: [2, 2],
            padding: "valid",
          },
          position: { x: 500, y: 300 },
          type: "maxpooling",
        },
        {
          id: "Flatten-1",
          data: { label: "Flatten Layer" },
          position: { x: 700, y: 400 },
          type: "flatten",
        },
        {
          id: "Dense-1",
          data: { label: "Dense Layer", neurons: 64, activation: "None" },
          position: { x: 900, y: 500 },
          type: "dense",
        },
        {
          id: "Activation-2",
          data: {
            label: "Activation Layer 2",
            function: "relu",
          },
          position: { x: 900, y: 550 },
          type: "activation",
        },
        {
          id: "Output-1",
          data: { label: "Output Layer", activation: "None" },
          position: { x: 1100, y: 600 },
          type: "output",
        },
        {
          id: "Activation-3",
          data: {
            label: "Activation Layer 3",
            function: "softmax",
          },
          position: { x: 1100, y: 650 },
          type: "activation",
        },
      ],
      "ResNet-18": [
        // Input layer
        {
          id: "input-1",
          data: { label: "Input Layer" },
          position: { x: 100, y: 300 },
          type: "input",
        },
        // Initial Conv + BatchNorm + MaxPool
        {
          id: "conv-1",
          data: {
            label: "Initial Conv",
            filters: 64,
            kernelSize: [7, 7],
            stride: [2, 2],
            padding: "same",
            activation: "ReLU",
          },
          position: { x: 250, y: 300 },
          type: "convolution",
        },
        {
          id: "batchnorm-1",
          data: {
            label: "BatchNorm",
            momentum: 0.9,
            epsilon: 1e-5,
          },
          position: { x: 400, y: 300 },
          type: "batchnormalization",
        },
        {
          id: "maxpool-1",
          data: {
            label: "MaxPool",
            poolSize: [3, 3],
            stride: [2, 2],
            padding: "same",
          },
          position: { x: 550, y: 300 },
          type: "maxpooling",
        },

        // First ResNet Block (Layer 1-1)
        {
          id: "conv-1-1-1",
          data: {
            label: "Conv1-1-1",
            filters: 64,
            kernelSize: [3, 3],
            stride: [1, 1],
            padding: "same",
            activation: "ReLU",
          },
          position: { x: 700, y: 240 },
          type: "convolution",
        },
        {
          id: "batchnorm-1-1-1",
          data: {
            label: "BN1-1-1",
            momentum: 0.9,
            epsilon: 1e-5,
          },
          position: { x: 800, y: 240 },
          type: "batchnormalization",
        },
        {
          id: "conv-1-1-2",
          data: {
            label: "Conv1-1-2",
            filters: 64,
            kernelSize: [3, 3],
            stride: [1, 1],
            padding: "same",
          },
          position: { x: 900, y: 240 },
          type: "convolution",
        },
        {
          id: "batchnorm-1-1-2",
          data: {
            label: "BN1-1-2",
            momentum: 0.9,
            epsilon: 1e-5,
          },
          position: { x: 1000, y: 240 },
          type: "batchnormalization",
        },
        {
          id: "add-1-1",
          data: {
            label: "Add1-1",
          },
          position: { x: 1100, y: 270 },
          type: "addlayer",
        },
        {
          id: "activation-1-1",
          data: {
            label: "ReLU1-1",
            activation: "ReLU",
          },
          position: { x: 1200, y: 270 },
          type: "activation",
        },

        // Second ResNet Block (Layer 1-2)
        {
          id: "conv-1-2-1",
          data: {
            label: "Conv1-2-1",
            filters: 64,
            kernelSize: [3, 3],
            stride: [1, 1],
            padding: "same",
            activation: "ReLU",
          },
          position: { x: 1300, y: 240 },
          type: "convolution",
        },
        {
          id: "batchnorm-1-2-1",
          data: {
            label: "BN1-2-1",
            momentum: 0.9,
            epsilon: 1e-5,
          },
          position: { x: 1400, y: 240 },
          type: "batchnormalization",
        },
        {
          id: "conv-1-2-2",
          data: {
            label: "Conv1-2-2",
            filters: 64,
            kernelSize: [3, 3],
            stride: [1, 1],
            padding: "same",
          },
          position: { x: 1500, y: 240 },
          type: "convolution",
        },
        {
          id: "batchnorm-1-2-2",
          data: {
            label: "BN1-2-2",
            momentum: 0.9,
            epsilon: 1e-5,
          },
          position: { x: 1600, y: 240 },
          type: "batchnormalization",
        },
        {
          id: "add-1-2",
          data: {
            label: "Add1-2",
          },
          position: { x: 1700, y: 270 },
          type: "addlayer",
        },
        {
          id: "activation-1-2",
          data: {
            label: "ReLU1-2",
            activation: "ReLU",
          },
          position: { x: 1800, y: 270 },
          type: "activation",
        },

        // First Block of Layer 2 with downsampling (stride 2)
        {
          id: "conv-2-1-1",
          data: {
            label: "Conv2-1-1",
            filters: 128,
            kernelSize: [3, 3],
            stride: [2, 2],
            padding: "same",
            activation: "ReLU",
          },
          position: { x: 700, y: 380 },
          type: "convolution",
        },
        {
          id: "batchnorm-2-1-1",
          data: {
            label: "BN2-1-1",
            momentum: 0.9,
            epsilon: 1e-5,
          },
          position: { x: 800, y: 380 },
          type: "batchnormalization",
        },
        {
          id: "conv-2-1-2",
          data: {
            label: "Conv2-1-2",
            filters: 128,
            kernelSize: [3, 3],
            stride: [1, 1],
            padding: "same",
          },
          position: { x: 900, y: 380 },
          type: "convolution",
        },
        {
          id: "batchnorm-2-1-2",
          data: {
            label: "BN2-1-2",
            momentum: 0.9,
            epsilon: 1e-5,
          },
          position: { x: 1000, y: 380 },
          type: "batchnormalization",
        },
        // Projection shortcut for downsampling
        {
          id: "conv-2-1-skip",
          data: {
            label: "Conv2-1-Skip",
            filters: 128,
            kernelSize: [1, 1],
            stride: [2, 2],
            padding: "same",
          },
          position: { x: 850, y: 470 },
          type: "convolution",
        },
        {
          id: "batchnorm-2-1-skip",
          data: {
            label: "BN2-1-Skip",
            momentum: 0.9,
            epsilon: 1e-5,
          },
          position: { x: 950, y: 470 },
          type: "batchnormalization",
        },
        {
          id: "add-2-1",
          data: {
            label: "Add2-1",
          },
          position: { x: 1100, y: 420 },
          type: "addlayer",
        },
        {
          id: "activation-2-1",
          data: {
            label: "ReLU2-1",
            activation: "ReLU",
          },
          position: { x: 1200, y: 420 },
          type: "activation",
        },

        // Second Block of Layer 2
        {
          id: "conv-2-2-1",
          data: {
            label: "Conv2-2-1",
            filters: 128,
            kernelSize: [3, 3],
            stride: [1, 1],
            padding: "same",
            activation: "ReLU",
          },
          position: { x: 1300, y: 380 },
          type: "convolution",
        },
        {
          id: "batchnorm-2-2-1",
          data: {
            label: "BN2-2-1",
            momentum: 0.9,
            epsilon: 1e-5,
          },
          position: { x: 1400, y: 380 },
          type: "batchnormalization",
        },
        {
          id: "conv-2-2-2",
          data: {
            label: "Conv2-2-2",
            filters: 128,
            kernelSize: [3, 3],
            stride: [1, 1],
            padding: "same",
          },
          position: { x: 1500, y: 380 },
          type: "convolution",
        },
        {
          id: "batchnorm-2-2-2",
          data: {
            label: "BN2-2-2",
            momentum: 0.9,
            epsilon: 1e-5,
          },
          position: { x: 1600, y: 380 },
          type: "batchnormalization",
        },
        {
          id: "add-2-2",
          data: {
            label: "Add2-2",
          },
          position: { x: 1700, y: 420 },
          type: "addlayer",
        },
        {
          id: "activation-2-2",
          data: {
            label: "ReLU2-2",
            activation: "ReLU",
          },
          position: { x: 1800, y: 420 },
          type: "activation",
        },

        // Global Pooling and Output
        {
          id: "globalavgpool-1",
          data: {
            label: "Global AvgPool",
          },
          position: { x: 900, y: 550 },
          type: "globalaveragepool",
        },
        {
          id: "flatten-1",
          data: {
            label: "Flatten",
          },
          position: { x: 1050, y: 550 },
          type: "flatten",
        },
        {
          id: "dense-1",
          data: {
            label: "FC Layer",
            neurons: 1000,
            activation: "ReLU",
          },
          position: { x: 1200, y: 550 },
          type: "dense",
        },
        {
          id: "output-1",
          data: {
            label: "Output",
            activation: "Softmax",
          },
          position: { x: 1350, y: 550 },
          type: "output",
        },
      ],

      // Add ResNet-34 template
      "ResNet-34": [
        {
          id: "input-1",
          data: { label: "Input Layer" },
          position: { x: 100, y: 300 },
          type: "input",
        },
        {
          id: "conv-1",
          data: {
            label: "Initial Conv Layer",
            filters: 64,
            kernelSize: [7, 7],
            stride: [2, 2],
            padding: "same",
            activation: "ReLU",
          },
          position: { x: 250, y: 300 },
          type: "convolution",
        },
        {
          id: "batchnorm-1",
          data: {
            label: "BatchNorm Layer",
            momentum: 0.9,
            epsilon: 1e-5,
          },
          position: { x: 400, y: 300 },
          type: "batchnormalization",
        },
        {
          id: "maxpool-1",
          data: {
            label: "Initial MaxPool",
            poolSize: [3, 3],
            stride: [2, 2],
            padding: "same",
          },
          position: { x: 550, y: 300 },
          type: "maxpooling",
        },
        // Layer 1 - 3 Basic Blocks for ResNet-34
        {
          id: "resblock-1-1",
          data: {
            label: "ResBlock 1-1",
            blockType: "Basic",
            filters: 64,
            stride: [1, 1],
            activation: "ReLU",
          },
          position: { x: 700, y: 200 },
          type: "resnetblock",
        },
        {
          id: "resblock-1-2",
          data: {
            label: "ResBlock 1-2",
            blockType: "Basic",
            filters: 64,
            stride: [1, 1],
            activation: "ReLU",
          },
          position: { x: 700, y: 300 },
          type: "resnetblock",
        },
        {
          id: "resblock-1-3",
          data: {
            label: "ResBlock 1-3",
            blockType: "Basic",
            filters: 64,
            stride: [1, 1],
            activation: "ReLU",
          },
          position: { x: 700, y: 400 },
          type: "resnetblock",
        },
        // Layer 2 - 4 Basic Blocks
        {
          id: "resblock-2-1",
          data: {
            label: "ResBlock 2-1",
            blockType: "Basic",
            filters: 128,
            stride: [2, 2],
            activation: "ReLU",
          },
          position: { x: 900, y: 150 },
          type: "resnetblock",
        },
        {
          id: "resblock-2-2",
          data: {
            label: "ResBlock 2-2",
            blockType: "Basic",
            filters: 128,
            stride: [1, 1],
            activation: "ReLU",
          },
          position: { x: 900, y: 250 },
          type: "resnetblock",
        },
        {
          id: "resblock-2-3",
          data: {
            label: "ResBlock 2-3",
            blockType: "Basic",
            filters: 128,
            stride: [1, 1],
            activation: "ReLU",
          },
          position: { x: 900, y: 350 },
          type: "resnetblock",
        },
        {
          id: "resblock-2-4",
          data: {
            label: "ResBlock 2-4",
            blockType: "Basic",
            filters: 128,
            stride: [1, 1],
            activation: "ReLU",
          },
          position: { x: 900, y: 450 },
          type: "resnetblock",
        },
        // Layer 3 - 6 Basic Blocks
        {
          id: "resblock-3-1",
          data: {
            label: "ResBlock 3-1",
            blockType: "Basic",
            inChannels: 128,
            outChannels: 256,
            stride: [2, 2],
            activation: "ReLU",
            useSkipConnection: true,
            downsampleType: "Conv1x1",
          },
          position: { x: 1100, y: 100 },
          type: "resnetblock",
        },
        {
          id: "resblock-3-2",
          data: {
            label: "ResBlock 3-2",
            blockType: "Basic",
            inChannels: 256,
            outChannels: 256,
            stride: [1, 1],
            activation: "ReLU",
            useSkipConnection: true,
            downsampleType: "None",
          },
          position: { x: 1100, y: 200 },
          type: "resnetblock",
        },
        {
          id: "resblock-3-3",
          data: {
            label: "ResBlock 3-3",
            blockType: "Basic",
            inChannels: 256,
            outChannels: 256,
            stride: [1, 1],
            activation: "ReLU",
            useSkipConnection: true,
            downsampleType: "None",
          },
          position: { x: 1100, y: 300 },
          type: "resnetblock",
        },
        {
          id: "resblock-3-4",
          data: {
            label: "ResBlock 3-4",
            blockType: "Basic",
            inChannels: 256,
            outChannels: 256,
            stride: [1, 1],
            activation: "ReLU",
            useSkipConnection: true,
            downsampleType: "None",
          },
          position: { x: 1100, y: 400 },
          type: "resnetblock",
        },
        {
          id: "resblock-3-5",
          data: {
            label: "ResBlock 3-5",
            blockType: "Basic",
            inChannels: 256,
            outChannels: 256,
            stride: [1, 1],
            activation: "ReLU",
            useSkipConnection: true,
            downsampleType: "None",
          },
          position: { x: 1100, y: 500 },
          type: "resnetblock",
        },
        {
          id: "resblock-3-6",
          data: {
            label: "ResBlock 3-6",
            blockType: "Basic",
            inChannels: 256,
            outChannels: 256,
            stride: [1, 1],
            activation: "ReLU",
            useSkipConnection: true,
            downsampleType: "None",
          },
          position: { x: 1100, y: 600 },
          type: "resnetblock",
        },
        // Layer 4 - 3 Basic Blocks
        {
          id: "resblock-4-1",
          data: {
            label: "ResBlock 4-1",
            blockType: "Basic",
            inChannels: 256,
            outChannels: 512,
            stride: [2, 2],
            activation: "ReLU",
            useSkipConnection: true,
            downsampleType: "Conv1x1",
          },
          position: { x: 1300, y: 250 },
          type: "resnetblock",
        },
        {
          id: "resblock-4-2",
          data: {
            label: "ResBlock 4-2",
            blockType: "Basic",
            inChannels: 512,
            outChannels: 512,
            stride: [1, 1],
            activation: "ReLU",
            useSkipConnection: true,
            downsampleType: "None",
          },
          position: { x: 1300, y: 350 },
          type: "resnetblock",
        },
        {
          id: "resblock-4-3",
          data: {
            label: "ResBlock 4-3",
            blockType: "Basic",
            inChannels: 512,
            outChannels: 512,
            stride: [1, 1],
            activation: "ReLU",
            useSkipConnection: true,
            downsampleType: "None",
          },
          position: { x: 1300, y: 450 },
          type: "resnetblock",
        },
        // Global Average Pooling
        {
          id: "globalavgpool-1",
          data: {
            label: "Global AvgPool",
          },
          position: { x: 1500, y: 300 },
          type: "globalaveragepool",
        },
        {
          id: "flatten-1",
          data: { label: "Flatten Layer" },
          position: { x: 1650, y: 300 },
          type: "flatten",
        },
        {
          id: "dense-1",
          data: {
            label: "FC Layer",
            neurons: 1000,
            activation: "ReLU",
          },
          position: { x: 1800, y: 300 },
          type: "dense",
        },
        {
          id: "output-1",
          data: { label: "Output Layer", activation: "Softmax" },
          position: { x: 1950, y: 300 },
          type: "output",
        },
      ],

      // Add ResNet-50 template
      "ResNet-50": [
        {
          id: "input-1",
          data: { label: "Input Layer" },
          position: { x: 100, y: 300 },
          type: "input",
        },
        {
          id: "conv-1",
          data: {
            label: "Initial Conv Layer",
            filters: 64,
            kernelSize: [7, 7],
            stride: [2, 2],
            padding: "same",
            activation: "ReLU",
          },
          position: { x: 250, y: 300 },
          type: "convolution",
        },
        {
          id: "batchnorm-1",
          data: {
            label: "BatchNorm Layer",
            momentum: 0.9,
            epsilon: 1e-5,
          },
          position: { x: 400, y: 300 },
          type: "batchnormalization",
        },
        {
          id: "maxpool-1",
          data: {
            label: "Initial MaxPool",
            poolSize: [3, 3],
            stride: [2, 2],
            padding: "same",
          },
          position: { x: 550, y: 300 },
          type: "maxpooling",
        },
        // Layer 1 - 3 Bottleneck blocks
        {
          id: "resblock-1-1",
          data: {
            label: "ResBlock 1-1",
            blockType: "Bottleneck",
            filters: 256,
            stride: [1, 1],
            activation: "ReLU",
          },
          position: { x: 700, y: 200 },
          type: "resnetblock",
        },
        {
          id: "resblock-1-2",
          data: {
            label: "ResBlock 1-2",
            blockType: "Bottleneck",
            filters: 256,
            stride: [1, 1],
            activation: "ReLU",
          },
          position: { x: 700, y: 300 },
          type: "resnetblock",
        },
        {
          id: "resblock-1-3",
          data: {
            label: "ResBlock 1-3",
            blockType: "Bottleneck",
            inChannels: 256,
            outChannels: 256,
            stride: [1, 1],
            activation: "ReLU",
            useSkipConnection: true,
            downsampleType: "None",
          },
          position: { x: 700, y: 400 },
          type: "resnetblock",
        },
        // Layer 2 - 4 Bottleneck blocks
        {
          id: "resblock-2-1",
          data: {
            label: "ResBlock 2-1",
            blockType: "Bottleneck",
            inChannels: 256,
            outChannels: 512,
            stride: [2, 2],
            activation: "ReLU",
            useSkipConnection: true,
            downsampleType: "Conv1x1",
          },
          position: { x: 900, y: 150 },
          type: "resnetblock",
        },
        {
          id: "resblock-2-2",
          data: {
            label: "ResBlock 2-2",
            blockType: "Bottleneck",
            inChannels: 512,
            outChannels: 512,
            stride: [1, 1],
            activation: "ReLU",
            useSkipConnection: true,
            downsampleType: "None",
          },
          position: { x: 900, y: 250 },
          type: "resnetblock",
        },
        {
          id: "resblock-2-3",
          data: {
            label: "ResBlock 2-3",
            blockType: "Bottleneck",
            inChannels: 512,
            outChannels: 512,
            stride: [1, 1],
            activation: "ReLU",
            useSkipConnection: true,
            downsampleType: "None",
          },
          position: { x: 900, y: 350 },
          type: "resnetblock",
        },
        {
          id: "resblock-2-4",
          data: {
            label: "ResBlock 2-4",
            blockType: "Bottleneck",
            inChannels: 512,
            outChannels: 512,
            stride: [1, 1],
            activation: "ReLU",
            useSkipConnection: true,
            downsampleType: "None",
          },
          position: { x: 900, y: 450 },
          type: "resnetblock",
        },
        // Layer 3 - 6 Bottleneck blocks
        {
          id: "resblock-3-1",
          data: {
            label: "ResBlock 3-1",
            blockType: "Bottleneck",
            inChannels: 512,
            outChannels: 1024,
            stride: [2, 2],
            activation: "ReLU",
            useSkipConnection: true,
            downsampleType: "Conv1x1",
          },
          position: { x: 1100, y: 100 },
          type: "resnetblock",
        },
        {
          id: "resblock-3-2",
          data: {
            label: "ResBlock 3-2",
            blockType: "Bottleneck",
            inChannels: 1024,
            outChannels: 1024,
            stride: [1, 1],
            activation: "ReLU",
            useSkipConnection: true,
            downsampleType: "None",
          },
          position: { x: 1100, y: 180 },
          type: "resnetblock",
        },
        {
          id: "resblock-3-3",
          data: {
            label: "ResBlock 3-3",
            blockType: "Bottleneck",
            inChannels: 1024,
            outChannels: 1024,
            stride: [1, 1],
            activation: "ReLU",
            useSkipConnection: true,
            downsampleType: "None",
          },
          position: { x: 1100, y: 260 },
          type: "resnetblock",
        },
        {
          id: "resblock-3-4",
          data: {
            label: "ResBlock 3-4",
            blockType: "Bottleneck",
            inChannels: 1024,
            outChannels: 1024,
            stride: [1, 1],
            activation: "ReLU",
            useSkipConnection: true,
            downsampleType: "None",
          },
          position: { x: 1100, y: 340 },
          type: "resnetblock",
        },
        {
          id: "resblock-3-5",
          data: {
            label: "ResBlock 3-5",
            blockType: "Bottleneck",
            inChannels: 1024,
            outChannels: 1024,
            stride: [1, 1],
            activation: "ReLU",
            useSkipConnection: true,
            downsampleType: "None",
          },
          position: { x: 1100, y: 420 },
          type: "resnetblock",
        },
        {
          id: "resblock-3-6",
          data: {
            label: "ResBlock 3-6",
            blockType: "Bottleneck",
            inChannels: 1024,
            outChannels: 1024,
            stride: [1, 1],
            activation: "ReLU",
            useSkipConnection: true,
            downsampleType: "None",
          },
          position: { x: 1100, y: 500 },
          type: "resnetblock",
        },
        // Layer 4 - 3 Bottleneck blocks
        {
          id: "resblock-4-1",
          data: {
            label: "ResBlock 4-1",
            blockType: "Bottleneck",
            inChannels: 1024,
            outChannels: 2048,
            stride: [2, 2],
            activation: "ReLU",
            useSkipConnection: true,
            downsampleType: "Conv1x1",
          },
          position: { x: 1300, y: 250 },
          type: "resnetblock",
        },
        {
          id: "resblock-4-2",
          data: {
            label: "ResBlock 4-2",
            blockType: "Bottleneck",
            inChannels: 2048,
            outChannels: 2048,
            stride: [1, 1],
            activation: "ReLU",
            useSkipConnection: true,
            downsampleType: "None",
          },
          position: { x: 1300, y: 350 },
          type: "resnetblock",
        },
        {
          id: "resblock-4-3",
          data: {
            label: "ResBlock 4-3",
            blockType: "Bottleneck",
            inChannels: 2048,
            outChannels: 2048,
            stride: [1, 1],
            activation: "ReLU",
            useSkipConnection: true,
            downsampleType: "None",
          },
          position: { x: 1300, y: 450 },
          type: "resnetblock",
        },
        // Global Average Pooling
        {
          id: "globalavgpool-1",
          data: {
            label: "Global AvgPool",
          },
          position: { x: 1500, y: 300 },
          type: "globalaveragepool",
        },
        {
          id: "flatten-1",
          data: { label: "Flatten Layer" },
          position: { x: 1650, y: 300 },
          type: "flatten",
        },
        {
          id: "dense-1",
          data: {
            label: "FC Layer",
            neurons: 1000,
            activation: "ReLU",
          },
          position: { x: 1800, y: 300 },
          type: "dense",
        },
        {
          id: "output-1",
          data: { label: "Output Layer", activation: "Softmax" },
          position: { x: 1950, y: 300 },
          type: "output",
        },
      ],
    };

    // Get the template and load it properly through the unified system
    const templateNodes = templates[templateKey];
    if (templateNodes) {
      console.log(`🎯 Loading template: ${templateKey}`);

      // Step 1: Set the template nodes
      setNodes(templateNodes);

      // Step 2: Define template edges (to be processed through unified system)
      let templateEdges: Edge[] = [];

      if (templateKey === "Transformer Encoder") {
        templateEdges = [
          // Main path connections
          { id: "e1", source: "Input-1", target: "LayerNorm-1" },
          { id: "e2", source: "LayerNorm-1", target: "Attention-1" },
          { id: "e3", source: "Attention-1", target: "Add-1" },
          { id: "e4", source: "Add-1", target: "LayerNorm-2" },
          { id: "e5", source: "LayerNorm-2", target: "Dense-1" },
          { id: "e6", source: "Dense-1", target: "Activation-1" },
          { id: "e7", source: "Activation-1", target: "Dropout-1" },
          { id: "e8", source: "Dropout-1", target: "Dense-2" },
          { id: "e9", source: "Dense-2", target: "Add-2" },
          { id: "e10", source: "Add-2", target: "Output-1" },
          { id: "e11", source: "Output-1", target: "Activation-2" },
          // Skip connections
          { id: "skip1", source: "Input-1", target: "Add-1" },
          { id: "skip2", source: "Add-1", target: "Add-2" },
        ];
      } else if (templateKey === "ResNet-18") {
        templateEdges = [
          // Initial layers
          { id: "e1", source: "input-1", target: "conv-1" },
          { id: "e2", source: "conv-1", target: "batchnorm-1" },
          { id: "e3", source: "batchnorm-1", target: "maxpool-1" },
          // Layer 1-1: First ResNet Block main path
          { id: "e4", source: "maxpool-1", target: "conv-1-1-1" },
          { id: "e5", source: "conv-1-1-1", target: "batchnorm-1-1-1" },
          { id: "e6", source: "batchnorm-1-1-1", target: "conv-1-1-2" },
          { id: "e7", source: "conv-1-1-2", target: "batchnorm-1-1-2" },
          { id: "e8", source: "batchnorm-1-1-2", target: "add-1-1" },
          // Layer 1-1: Skip connection
          { id: "e9", source: "maxpool-1", target: "add-1-1" },
          // Layer 1-1: Post-addition activation
          { id: "e10", source: "add-1-1", target: "activation-1-1" },
          // Layer 1-2: Second ResNet Block main path
          { id: "e11", source: "activation-1-1", target: "conv-1-2-1" },
          { id: "e12", source: "conv-1-2-1", target: "batchnorm-1-2-1" },
          { id: "e13", source: "batchnorm-1-2-1", target: "conv-1-2-2" },
          { id: "e14", source: "conv-1-2-2", target: "batchnorm-1-2-2" },
          { id: "e15", source: "batchnorm-1-2-2", target: "add-1-2" },
          // Layer 1-2: Skip connection
          { id: "e16", source: "activation-1-1", target: "add-1-2" },
          // Layer 1-2: Post-addition activation
          { id: "e17", source: "add-1-2", target: "activation-1-2" },
          // Layer 2-1: First block of layer 2 with downsampling (main path)
          { id: "e18", source: "activation-1-2", target: "conv-2-1-1" },
          { id: "e19", source: "conv-2-1-1", target: "batchnorm-2-1-1" },
          { id: "e20", source: "batchnorm-2-1-1", target: "conv-2-1-2" },
          { id: "e21", source: "conv-2-1-2", target: "batchnorm-2-1-2" },
          { id: "e22", source: "batchnorm-2-1-2", target: "add-2-1" },
          // Layer 2-1: Skip connection with projection
          { id: "e23", source: "activation-1-2", target: "conv-2-1-skip" },
          { id: "e24", source: "conv-2-1-skip", target: "batchnorm-2-1-skip" },
          { id: "e25", source: "batchnorm-2-1-skip", target: "add-2-1" },
          // Layer 2-1: Post-addition activation
          { id: "e26", source: "add-2-1", target: "activation-2-1" },
          // Layer 2-2: Second block of layer 2 (main path)
          { id: "e27", source: "activation-2-1", target: "conv-2-2-1" },
          { id: "e28", source: "conv-2-2-1", target: "batchnorm-2-2-1" },
          { id: "e29", source: "batchnorm-2-2-1", target: "conv-2-2-2" },
          { id: "e30", source: "conv-2-2-2", target: "batchnorm-2-2-2" },
          { id: "e31", source: "batchnorm-2-2-2", target: "add-2-2" },
          // Layer 2-2: Skip connection
          { id: "e32", source: "activation-2-1", target: "add-2-2" },
          // Layer 2-2: Post-addition activation
          { id: "e33", source: "add-2-2", target: "activation-2-2" },
          // Final classification layers
          { id: "e34", source: "activation-2-2", target: "globalavgpool-1" },
          { id: "e35", source: "globalavgpool-1", target: "flatten-1" },
          { id: "e36", source: "flatten-1", target: "dense-1" },
          { id: "e37", source: "dense-1", target: "output-1" },
        ];
      } else if (templateKey === "ResNet-34") {
        templateEdges = [
          // Initial layers
          { id: "e1", source: "input-1", target: "conv-1" },
          { id: "e2", source: "conv-1", target: "batchnorm-1" },
          { id: "e3", source: "batchnorm-1", target: "maxpool-1" },
          // Stage 1 - 3 blocks
          { id: "e4", source: "maxpool-1", target: "resblock-1-1" },
          { id: "e5", source: "resblock-1-1", target: "resblock-1-2" },
          { id: "e6", source: "resblock-1-2", target: "resblock-1-3" },
          // Stage 2 - 4 blocks
          { id: "e7", source: "resblock-1-3", target: "resblock-2-1" },
          { id: "e8", source: "resblock-2-1", target: "resblock-2-2" },
          { id: "e9", source: "resblock-2-2", target: "resblock-2-3" },
          { id: "e10", source: "resblock-2-3", target: "resblock-2-4" },
          // Stage 3 - 6 blocks
          { id: "e11", source: "resblock-2-4", target: "resblock-3-1" },
          { id: "e12", source: "resblock-3-1", target: "resblock-3-2" },
          { id: "e13", source: "resblock-3-2", target: "resblock-3-3" },
          { id: "e14", source: "resblock-3-3", target: "resblock-3-4" },
          { id: "e15", source: "resblock-3-4", target: "resblock-3-5" },
          { id: "e16", source: "resblock-3-5", target: "resblock-3-6" },
          // Stage 4 - 3 blocks
          { id: "e17", source: "resblock-3-6", target: "resblock-4-1" },
          { id: "e18", source: "resblock-4-1", target: "resblock-4-2" },
          { id: "e19", source: "resblock-4-2", target: "resblock-4-3" },
          // Final layers
          { id: "e20", source: "resblock-4-3", target: "globalavgpool-1" },
          { id: "e21", source: "globalavgpool-1", target: "flatten-1" },
          { id: "e22", source: "flatten-1", target: "dense-1" },
          { id: "e23", source: "dense-1", target: "output-1" },
        ];
      } else if (templateKey === "ResNet-50") {
        templateEdges = [
          // Initial layers
          { id: "e1", source: "input-1", target: "conv-1" },
          { id: "e2", source: "conv-1", target: "batchnorm-1" },
          { id: "e3", source: "batchnorm-1", target: "maxpool-1" },
          // Stage 1 - 3 Bottleneck blocks
          { id: "e4", source: "maxpool-1", target: "resblock-1-1" },
          { id: "e5", source: "resblock-1-1", target: "resblock-1-2" },
          { id: "e6", source: "resblock-1-2", target: "resblock-1-3" },
          // Stage 2 - 4 Bottleneck blocks
          { id: "e7", source: "resblock-1-3", target: "resblock-2-1" },
          { id: "e8", source: "resblock-2-1", target: "resblock-2-2" },
          { id: "e9", source: "resblock-2-2", target: "resblock-2-3" },
          { id: "e10", source: "resblock-2-3", target: "resblock-2-4" },
          // Stage 3 - 6 Bottleneck blocks
          { id: "e11", source: "resblock-2-4", target: "resblock-3-1" },
          { id: "e12", source: "resblock-3-1", target: "resblock-3-2" },
          { id: "e13", source: "resblock-3-2", target: "resblock-3-3" },
          { id: "e14", source: "resblock-3-3", target: "resblock-3-4" },
          { id: "e15", source: "resblock-3-4", target: "resblock-3-5" },
          { id: "e16", source: "resblock-3-5", target: "resblock-3-6" },
          // Stage 4 - 3 Bottleneck blocks
          { id: "e17", source: "resblock-3-6", target: "resblock-4-1" },
          { id: "e18", source: "resblock-4-1", target: "resblock-4-2" },
          { id: "e19", source: "resblock-4-2", target: "resblock-4-3" },
          // Final layers
          { id: "e20", source: "resblock-4-3", target: "globalavgpool-1" },
          { id: "e21", source: "globalavgpool-1", target: "flatten-1" },
          { id: "e22", source: "flatten-1", target: "dense-1" },
          { id: "e23", source: "dense-1", target: "output-1" },
        ];
      } else if (templateKey !== "Blank") {
        // For simpler templates, create sequential connections
        for (let i = 0; i < templateNodes.length - 1; i++) {
          templateEdges.push({
            id: `e${i + 1}`,
            source: templateNodes[i].id,
            target: templateNodes[i + 1].id,
          });
        }
      }

      // Step 3: Use the unified system to process the template architecture
      console.log(
        `📋 Processing template with ${templateNodes.length} nodes and ${templateEdges.length} edges`
      );

      // Apply the unified payload processing to the template
      const { nodes: processedNodes, edges: processedEdges } =
        prepareArchitecturePayload(templateNodes, templateEdges);

      // Step 4: Set the processed nodes and edges (making templates fully editable)
      setNodes(processedNodes);
      setEdges(processedEdges);

      console.log(`✅ Template loaded and processed through unified system!`);
      console.log(
        `   Nodes: ${processedNodes.length} | Edges: ${processedEdges.length}`
      );
    }
  };

  // Function to update node parameters
  const updateParameter = (param: string, value: any): void => {
    if (!selectedNode) return;

    setNodes((nds) =>
      nds.map((node) =>
        node.id === selectedNode.id
          ? { ...node, data: { ...node.data, [param]: value } }
          : node
      )
    );

    setSelectedNode((prevNode) =>
      prevNode
        ? { ...prevNode, data: { ...prevNode.data, [param]: value } }
        : null
    );
  };

  // ReactFlow event handlers
  const onNodesChange: OnNodesChange = (changes: NodeChange[]) => {
    setNodes((nds) => applyNodeChanges(changes, nds));
  };

  const onEdgesChange: OnEdgesChange = (changes: EdgeChange[]) => {
    setEdges((eds) => applyEdgeChanges(changes, eds));
  };

  // Function to analyze and sequence the manual architecture properly
  const analyzeAndSequenceArchitecture = (
    currentNodes: Node[],
    currentEdges: Edge[]
  ) => {
    // Find the input node (starting point)
    const inputNode = currentNodes.find((node) => node.type === "input");
    if (!inputNode)
      return { sequencedNodes: currentNodes, sequencedEdges: currentEdges };

    // Keep all edges as they are - allow output->activation connections like templates
    const correctedEdges = currentEdges;

    // Build adjacency list for traversal
    const adjacencyList = new Map<string, string[]>();
    currentNodes.forEach((node) => adjacencyList.set(node.id, []));

    correctedEdges.forEach((edge) => {
      if (edge.source && edge.target) {
        const neighbors = adjacencyList.get(edge.source) || [];
        neighbors.push(edge.target);
        adjacencyList.set(edge.source, neighbors);
      }
    });

    // Perform topological sort starting from input
    const visited = new Set<string>();
    const sequencedNodeIds: string[] = [];

    const dfs = (nodeId: string) => {
      if (visited.has(nodeId)) return;
      visited.add(nodeId);
      sequencedNodeIds.push(nodeId);

      const neighbors = adjacencyList.get(nodeId) || [];
      neighbors.forEach((neighbor) => dfs(neighbor));
    };

    dfs(inputNode.id);

    // Create clean, sequenced edges with proper IDs
    const cleanEdges: Edge[] = [];
    let edgeIndex = 1;

    // Rebuild edges in the correct sequence
    sequencedNodeIds.forEach((nodeId) => {
      const neighbors = adjacencyList.get(nodeId) || [];
      neighbors.forEach((targetId) => {
        cleanEdges.push({
          id: `e${edgeIndex}`,
          source: nodeId,
          target: targetId,
        });
        edgeIndex++;
      });
    });

    // Check for completely disconnected nodes (optional warning)
    const connectedNodeIds = new Set<string>();
    cleanEdges.forEach((edge) => {
      connectedNodeIds.add(edge.source);
      connectedNodeIds.add(edge.target);
    });

    const disconnectedNodes = currentNodes.filter(
      (node) => !connectedNodeIds.has(node.id)
    );

    if (disconnectedNodes.length > 0) {
      console.info(
        `Found disconnected nodes: ${disconnectedNodes
          .map((n) => n.id)
          .join(", ")}. These will be included at the end of the sequence.`
      );
    }

    // Sequence the nodes in the same order as the flow
    const sequencedNodes: Node[] = [];

    // Add nodes in the order they appear in the flow
    sequencedNodeIds.forEach((nodeId) => {
      const node = currentNodes.find((n) => n.id === nodeId);
      if (node) {
        sequencedNodes.push(node);
      }
    });

    // Add any remaining nodes that weren't in the flow (disconnected nodes)
    currentNodes.forEach((node) => {
      if (!sequencedNodeIds.includes(node.id)) {
        sequencedNodes.push(node);
      }
    });

    return { sequencedNodes, sequencedEdges: cleanEdges };
  };

  // Enhanced connection validation
  const isValidConnection = (connection: Connection): boolean => {
    if (!connection.source || !connection.target) return false;

    const sourceNode = nodes.find((n) => n.id === connection.source);
    const targetNode = nodes.find((n) => n.id === connection.target);

    if (!sourceNode || !targetNode) return false;
    if (!sourceNode.type || !targetNode.type) return false;

    // Rule 1: Input layers can only be sources, never targets
    if (targetNode.type === "input") {
      console.log("❌ Cannot connect TO input layer");
      return false;
    }

    // Rule 2: Prevent self-connections
    if (connection.source === connection.target) {
      console.log("❌ Cannot connect layer to itself");
      return false;
    }

    // Rule 3: Prevent duplicate connections
    const existingConnection = edges.find(
      (edge) =>
        edge.source === connection.source && edge.target === connection.target
    );
    if (existingConnection) {
      console.log("❌ Connection already exists");
      return false;
    }

    // Rule 4: Allow flexible layer connections
    // All layer types can be sources - activation, dropout, batchnorm, flatten etc. can all pass data forward

    // Rule 5: Prevent connections to layers that don't accept inputs
    const nonInputLayers = ["input"];
    if (targetNode.type && nonInputLayers.includes(targetNode.type)) {
      console.log(`❌ Cannot connect to ${targetNode.type} layer`);
      return false;
    }

    // 🎯 SMART CONNECTION POLICY: Allow valid connections while preventing impossible ones!
    // - Allow skip connections (ResNet, Transformer, etc.)
    // - Allow backward connections for valid architectures
    // - Allow multiple inputs to layers that support it (Add layers, complex architectures)
    // - Allow output→activation connections (final activation layers)
    // - Allow any creative but valid architecture the user wants to build!

    console.log(
      `✅ Allowing valid connection: ${sourceNode.type} → ${targetNode.type}`
    );
    return true;
  };

  const onConnect = (connection: Connection): void => {
    console.log("🔗 Connecting nodes:", connection);

    // Validate the connection first
    if (!isValidConnection(connection)) {
      console.warn("❌ Invalid connection attempted:", connection);
      return;
    }

    // Log special connection types
    const sourceNode = nodes.find((n) => n.id === connection.source);
    const targetNode = nodes.find((n) => n.id === connection.target);
    if (sourceNode?.type === "output" && targetNode?.type === "activation") {
      console.log(
        "✅ Creating output→activation connection (final activation layer)"
      );
    }

    // Add the connection with a temporary ID
    const tempConnection = {
      ...connection,
      id: `temp-${Date.now()}`,
    };

    // Create new edges and use the unified payload system for consistency
    const newEdges = addEdge(tempConnection, edges);

    console.log("🔄 Processing new connection through unified system...");
    const { nodes: processedNodes, edges: processedEdges } =
      prepareArchitecturePayload(nodes, newEdges);

    console.log("✅ Manual connection created and processed:");
    console.log(
      "  Edge flow:",
      processedEdges.map((edge) => `${edge.source} → ${edge.target}`).join(", ")
    );
    console.log(
      "  Node sequence:",
      processedNodes
        .map((node: Node) => `${node.id} (${node.type})`)
        .join(" → ")
    );

    // Update both nodes and edges with the processed versions (fully editable)
    setNodes(processedNodes);
    setEdges(processedEdges);
  };

  const onNodeClick = (_: React.MouseEvent, node: Node): void => {
    setSelectedNode(node);
    setActiveRightSidebarTab("parameters");
  };

  // Validate layer parameters before saving or training
  const validateLayerParameters = (): ValidationErrors => {
    const errors: ValidationErrors = [];

    // Check if dataset is selected
    if (!selectedDataset) {
      errors.push(
        "[Critical] Please select a dataset before saving or training"
      );
      return errors;
    }

    // Check if there are at least two nodes (input and output)
    if (nodes.length < 2) {
      errors.push(
        "[Critical] Model must have at least input and output layers"
      );
      return errors;
    }

    // Find input and output layers
    const inputLayer = nodes.find((node) => node.type === "input");
    const outputLayer = nodes.find((node) => node.type === "output");

    // Check if input layer exists
    if (!inputLayer) {
      errors.push("[Critical] Model must have an input layer");
    } else {
      // Check if input layer is connected
      const isInputConnected = edges.some(
        (edge) => edge.source === inputLayer.id
      );
      if (!isInputConnected) {
        errors.push(
          `[Critical] Input layer ${inputLayer.id} must have at least one outgoing connection`
        );
      }
    }

    // Check if output layer exists
    if (!outputLayer) {
      errors.push("[Critical] Model must have an output layer");
    } else {
      // Check if output layer is connected
      const isOutputConnected = edges.some(
        (edge) => edge.target === outputLayer.id
      );
      if (!isOutputConnected) {
        errors.push(
          `[Critical] Output layer ${outputLayer.id} must have at least one incoming connection`
        );
      }
    }

    // Check if all nodes are connected
    const connectedNodeIds = new Set<string>();

    // Add source and target nodes from edges
    edges.forEach((edge) => {
      connectedNodeIds.add(edge.source);
      connectedNodeIds.add(edge.target);
    });

    // Find disconnected nodes
    const disconnectedNodes = nodes.filter(
      (node) => !connectedNodeIds.has(node.id)
    );
    if (disconnectedNodes.length > 0) {
      errors.push(
        `[Critical] The following layers are disconnected: ${disconnectedNodes
          .map((node) => node.id)
          .join(", ")}`
      );
    }

    // Validate parameters for each layer type
    nodes.forEach((node) => {
      switch (node.type) {
        case "dense":
          if (!Number.isInteger(node.data.neurons) || node.data.neurons <= 0) {
            errors.push(
              `Dense layer ${node.id}: Number of neurons must be a positive integer`
            );
          }
          break;
        case "convolution":
          if (!Number.isInteger(node.data.filters) || node.data.filters <= 0) {
            errors.push(
              `Convolution layer ${node.id}: Number of filters must be a positive integer`
            );
          }
          if (
            !Array.isArray(node.data.kernelSize) ||
            node.data.kernelSize.length !== 2 ||
            node.data.kernelSize.some(
              (size: number) => !Number.isInteger(size) || size <= 0
            )
          ) {
            errors.push(
              `Convolution layer ${node.id}: Kernel size must be an array of two positive integers`
            );
          }
          if (
            !Array.isArray(node.data.stride) ||
            node.data.stride.length !== 2 ||
            node.data.stride.some(
              (size: number) => !Number.isInteger(size) || size <= 0
            )
          ) {
            errors.push(
              `Convolution layer ${node.id}: Stride must be an array of two positive integers`
            );
          }
          break;
        case "maxpooling":
          if (
            !Array.isArray(node.data.poolSize) ||
            node.data.poolSize.length !== 2 ||
            node.data.poolSize.some(
              (size: number) => !Number.isInteger(size) || size <= 0
            )
          ) {
            errors.push(
              `MaxPooling layer ${node.id}: Pool size must be an array of two positive integers`
            );
          }
          if (
            !Array.isArray(node.data.stride) ||
            node.data.stride.length !== 2 ||
            node.data.stride.some(
              (size: number) => !Number.isInteger(size) || size <= 0
            )
          ) {
            errors.push(
              `MaxPooling layer ${node.id}: Stride must be an array of two positive integers`
            );
          }
          if (
            !["valid", "same", "none"].includes(
              node.data.padding?.toLowerCase()
            )
          ) {
            errors.push(
              `MaxPooling layer ${node.id}: Padding must be 'valid', 'same', or 'none'`
            );
          }
          break;
        case "dropout":
          if (
            typeof node.data.rate !== "number" ||
            node.data.rate < 0 ||
            node.data.rate >= 1
          ) {
            errors.push(
              `Dropout layer ${node.id}: Rate must be a number between 0 and 1`
            );
          }
          break;
        case "batchnormalization":
          if (
            typeof node.data.momentum !== "number" ||
            node.data.momentum <= 0 ||
            node.data.momentum >= 1
          ) {
            errors.push(
              `BatchNormalization layer ${node.id}: Momentum must be a number between 0 and 1`
            );
          }
          if (typeof node.data.epsilon !== "number" || node.data.epsilon <= 0) {
            errors.push(
              `BatchNormalization layer ${node.id}: Epsilon must be a positive number`
            );
          }
          break;
        case "attention":
          if (!Number.isInteger(node.data.heads) || node.data.heads <= 0) {
            errors.push(
              `Attention layer ${node.id}: Number of heads must be a positive integer`
            );
          }
          if (!Number.isInteger(node.data.keyDim) || node.data.keyDim <= 0) {
            errors.push(
              `Attention layer ${node.id}: Key dimension must be a positive integer`
            );
          }
          if (
            typeof node.data.dropout !== "number" ||
            node.data.dropout < 0 ||
            node.data.dropout >= 1
          ) {
            errors.push(
              `Attention layer ${node.id}: Dropout rate must be a number between 0 and 1`
            );
          }
          break;
        case "resnetblock":
          // Validate block type
          if (!["Basic", "Bottleneck"].includes(node.data.blockType)) {
            errors.push(
              `ResNet block ${node.id}: Block type must be 'Basic' or 'Bottleneck'`
            );
          }

          // Validate filters
          if (!Number.isInteger(node.data.filters) || node.data.filters <= 0) {
            errors.push(
              `ResNet block ${node.id}: Filters must be a positive integer`
            );
          }

          // For Bottleneck blocks, validate that filters is divisible by 4
          if (
            node.data.blockType === "Bottleneck" &&
            node.data.filters % 4 !== 0
          ) {
            errors.push(
              `ResNet block ${node.id}: For Bottleneck blocks, filters should be divisible by 4 (current: ${node.data.filters})`
            );
          }

          // Validate stride
          if (
            !Array.isArray(node.data.stride) ||
            node.data.stride.length !== 2 ||
            node.data.stride.some(
              (size: number) => !Number.isInteger(size) || size <= 0
            )
          ) {
            errors.push(
              `ResNet block ${node.id}: Stride must be an array of two positive integers`
            );
          }

          // Validate activation
          if (
            !["relu", "leaky_relu"].includes(
              node.data.activation?.toLowerCase()
            )
          ) {
            errors.push(`ResNet block ${node.id}: Invalid activation function`);
          }
          break;
        case "output":
          if (
            !["none", "sigmoid", "softmax"].includes(
              node.data.activation?.toLowerCase()
            )
          ) {
            errors.push(
              `Output layer ${node.id}: Invalid activation function. Must be None, Sigmoid, or Softmax`
            );
          }
          break;
        case "activation":
          if (
            ![
              "relu",
              "sigmoid",
              "tanh",
              "softmax",
              "leaky_relu",
              "elu",
              "swish",
              "linear",
            ].includes(node.data.function)
          ) {
            errors.push(
              `Activation layer ${node.id}: Invalid activation function`
            );
          }
          break;
      }
    });

    // Dataset-specific validations
    if (selectedDataset) {
      // MNIST and CIFAR-10 specific validations for image datasets
      if (selectedDataset === "MNIST" || selectedDataset === "CIFAR-10") {
        // Check if there's a convolutional layer for image datasets
        const hasConvolutionLayer = nodes.some(
          (node) => node.type === "convolution"
        );

        // Check if there's a ResNet block for image datasets (also valid for image processing)
        const hasResNetBlock = nodes.some(
          (node) => node.type === "resnetblock"
        );

        // Check if there's a GlobalAveragePool layer which is common in ResNet architectures
        const hasGlobalAveragePool = nodes.some(
          (node) => node.type === "globalaveragepool"
        );

        // Allow flexibility - either conv or resnet is fine for image data, but suggest having at least one
        if (!hasConvolutionLayer && !hasResNetBlock) {
          errors.push(
            `Suggestion: ${selectedDataset} dataset works best with at least one convolutional layer or ResNet block`
          );
        }

        // Suggest GlobalAveragePooling for ResNet models
        if (
          hasResNetBlock &&
          !hasGlobalAveragePool &&
          !nodes.some((node) => node.type === "flatten")
        ) {
          errors.push(
            `Suggestion: ResNet models typically use GlobalAveragePooling instead of MaxPooling at the end of the network`
          );
        }

        // Check if convolutional, maxpooling, or ResNet layers are properly connected to flatten or GlobalAveragePool layers
        // We'll perform a more sophisticated check in validateResNetArchitecture, but do a basic check here
        const convOrPoolNodes = nodes.filter(
          (node) => node.type === "convolution" || node.type === "maxpooling"
        );

        // Only check direct connections to flatten for non-ResNet blocks
        // (ResNet architecture is checked more thoroughly in validateResNetArchitecture)
        if (!hasResNetBlock) {
          for (const node of convOrPoolNodes) {
            // Find all edges from this node
            const outgoingEdges = edges.filter(
              (edge) => edge.source === node.id
            );

            // Check if any of these edges connect to a flatten layer or another layer type
            // that could lead to a flatten layer eventually
            const connectsToValidLayer = outgoingEdges.some((edge) => {
              const targetNode = nodes.find((n) => n.id === edge.target);
              return (
                targetNode &&
                (targetNode.type === "flatten" ||
                  targetNode.type === "globalaveragepool" ||
                  targetNode.type === "convolution" ||
                  targetNode.type === "maxpooling" ||
                  targetNode.type === "resnetblock")
              );
            });

            if (!connectsToValidLayer) {
              errors.push(
                `${node.type} layer ${node.id} should eventually connect to a Flatten or GlobalAveragePooling layer, or another convolutional layer`
              );
            }
          }
        }

        // Output layer must use Softmax for classification (this is a strict requirement)
        if (outputLayer && outputLayer.data.activation !== "Softmax") {
          // For standalone activation layers, we'll check connectivity instead
          // Find all activation layers with Softmax
          const softmaxLayers = nodes.filter(
            (node) =>
              node.type === "activation" && node.data.function === "softmax"
          );

          let needsActivationError = true;

          if (softmaxLayers.length > 0) {
            // Build a connectivity graph to check for both direct and indirect connections
            const graph = new Map<string, Set<string>>();

            // Initialize graph with all nodes
            nodes.forEach((node) => {
              graph.set(node.id, new Set<string>());
            });

            // Add connections to the graph (in both directions for undirected traversal)
            edges.forEach((edge) => {
              if (graph.has(edge.source)) {
                graph.get(edge.source)!.add(edge.target);
              }
              if (graph.has(edge.target)) {
                graph.get(edge.target)!.add(edge.source);
              }
            });

            // Check if there's any path between output layer and any Softmax activation layer
            let isSoftmaxConnected = false;

            // BFS to find Softmax activation connected to output layer
            const queue: string[] = [outputLayer.id];
            const visited = new Set<string>([outputLayer.id]);

            while (queue.length > 0 && !isSoftmaxConnected) {
              const currentNodeId = queue.shift()!;

              if (softmaxLayers.some((node) => node.id === currentNodeId)) {
                isSoftmaxConnected = true;
                break;
              }

              const neighbors = graph.get(currentNodeId) || new Set<string>();
              neighbors.forEach((neighborId) => {
                if (!visited.has(neighborId)) {
                  visited.add(neighborId);
                  queue.push(neighborId);
                }
              });
            }

            // If output is connected to Softmax activation, we're good
            if (isSoftmaxConnected) {
              // Skip the error as we found a valid connection
              needsActivationError = false;
            }
          }

          // If we get here, neither the output has Softmax activation nor is connected to a Softmax layer
          if (needsActivationError) {
            errors.push(
              `Output layer must use Softmax activation for ${selectedDataset} dataset`
            );
          }
        }
      }

      // Tabular datasets - avoid image-specific layers but provide some flexibility
      else if (
        ["Iris", "Breast Cancer", "California Housing"].includes(
          selectedDataset
        )
      ) {
        // Allow flexibility with warning for more advanced users
        const imageSpecificLayers = nodes.filter((node) =>
          ["convolution", "maxpooling"].includes(node.type || "")
        );

        // Separate ResNet blocks for a different type of warning
        const resNetBlocks = nodes.filter(
          (node) => node.type === "resnetblock"
        );

        if (imageSpecificLayers.length > 0) {
          errors.push(
            `Warning: ${selectedDataset} dataset typically doesn't need convolutional layers, but they can be used if you're applying advanced techniques`
          );
        }

        if (resNetBlocks.length > 0) {
          errors.push(
            `Warning: ${selectedDataset} dataset typically doesn't use ResNet blocks, but they can be used if you're applying advanced techniques`
          );
        }

        // Output activations still have strict requirements based on task type
        if (selectedDataset === "Iris") {
          // Multi-class classification
          // Check if an activation layer with Softmax is connected to the output layer (in either direction)
          if (outputLayer) {
            // Find all activation layers with Softmax
            const activationLayers = nodes.filter(
              (node) =>
                node.type === "activation" && node.data.function === "softmax"
            );

            // Build a connectivity graph to check for both direct and indirect connections
            const graph = new Map<string, Set<string>>();

            // Initialize graph with all nodes
            nodes.forEach((node) => {
              graph.set(node.id, new Set<string>());
            });

            // Add connections to the graph (in both directions for undirected traversal)
            edges.forEach((edge) => {
              if (graph.has(edge.source)) {
                graph.get(edge.source)!.add(edge.target);
              }
              if (graph.has(edge.target)) {
                graph.get(edge.target)!.add(edge.source);
              }
            });

            // Check if there's any path between output layer and any Softmax activation layer
            let isSoftmaxConnected = false;

            // Breadth-first search to find connections
            const bfs = (startNodeId: string) => {
              const queue: string[] = [startNodeId];
              const visited = new Set<string>([startNodeId]);

              while (queue.length > 0) {
                const currentNodeId = queue.shift()!;

                // Check if this is a Softmax activation layer
                if (
                  activationLayers.some((node) => node.id === currentNodeId)
                ) {
                  isSoftmaxConnected = true;
                  break;
                }

                // Add connected nodes to the queue
                const neighbors = graph.get(currentNodeId) || new Set<string>();
                neighbors.forEach((neighborId) => {
                  if (!visited.has(neighborId)) {
                    visited.add(neighborId);
                    queue.push(neighborId);
                  }
                });
              }
            };

            // Start BFS from the output layer
            bfs(outputLayer.id);

            // Also check the reverse: if any activation layer connects to the output
            if (!isSoftmaxConnected) {
              activationLayers.forEach((activationLayer) => {
                bfs(activationLayer.id);
              });
            }

            if (!isSoftmaxConnected) {
              errors.push(
                `Output layer must be connected to a Softmax activation layer for Iris dataset (multi-class classification)`
              );
            }
          }
        } else if (selectedDataset === "Breast Cancer") {
          // Binary classification
          // Check if an activation layer with Sigmoid is connected to the output layer
          if (outputLayer) {
            // Find all activation layers with Sigmoid
            const activationLayers = nodes.filter(
              (node) =>
                node.type === "activation" && node.data.function === "sigmoid"
            );

            // Build a connectivity graph to check for both direct and indirect connections
            const graph = new Map<string, Set<string>>();

            // Initialize graph with all nodes
            nodes.forEach((node) => {
              graph.set(node.id, new Set<string>());
            });

            // Add connections to the graph (in both directions for undirected traversal)
            edges.forEach((edge) => {
              if (graph.has(edge.source)) {
                graph.get(edge.source)!.add(edge.target);
              }
              if (graph.has(edge.target)) {
                graph.get(edge.target)!.add(edge.source);
              }
            });

            // Check if there's any path between output layer and any Sigmoid activation layer
            let isSigmoidConnected = false;

            // Breadth-first search to find connections
            const bfs = (startNodeId: string) => {
              const queue: string[] = [startNodeId];
              const visited = new Set<string>([startNodeId]);

              while (queue.length > 0) {
                const currentNodeId = queue.shift()!;

                // Check if this is a Sigmoid activation layer
                if (
                  activationLayers.some((node) => node.id === currentNodeId)
                ) {
                  isSigmoidConnected = true;
                  break;
                }

                // Add connected nodes to the queue
                const neighbors = graph.get(currentNodeId) || new Set<string>();
                neighbors.forEach((neighborId) => {
                  if (!visited.has(neighborId)) {
                    visited.add(neighborId);
                    queue.push(neighborId);
                  }
                });
              }
            };

            // Start BFS from the output layer
            bfs(outputLayer.id);

            // Also check the reverse: if any activation layer connects to the output
            if (!isSigmoidConnected) {
              activationLayers.forEach((activationLayer) => {
                bfs(activationLayer.id);
              });
            }

            if (!isSigmoidConnected) {
              errors.push(
                `Output layer must be connected to a Sigmoid activation layer for Breast Cancer dataset (binary classification)`
              );
            }
          }
        } else if (selectedDataset === "California Housing") {
          // Regression - no activation needed (linear output)
          // Check that no activation is connected to the output, or if connected, it's set to linear/none
          if (outputLayer) {
            // Find potentially invalid activation layers
            const nonLinearActivations = nodes.filter(
              (node) =>
                node.type === "activation" &&
                !["None", "Linear"].includes(node.data.function)
            );

            // Build a connectivity graph to check for both direct and indirect connections
            const graph = new Map<string, Set<string>>();

            // Initialize graph with all nodes
            nodes.forEach((node) => {
              graph.set(node.id, new Set<string>());
            });

            // Add connections to the graph (in both directions for undirected traversal)
            edges.forEach((edge) => {
              if (graph.has(edge.source)) {
                graph.get(edge.source)!.add(edge.target);
              }
              if (graph.has(edge.target)) {
                graph.get(edge.target)!.add(edge.source);
              }
            });

            // Check if there's any path between output layer and any non-linear activation
            let hasInvalidActivation = false;

            // Breadth-first search to find connections
            const bfs = (startNodeId: string) => {
              const queue: string[] = [startNodeId];
              const visited = new Set<string>([startNodeId]);

              while (queue.length > 0) {
                const currentNodeId = queue.shift()!;

                // Check if this is a non-linear activation layer
                if (
                  nonLinearActivations.some((node) => node.id === currentNodeId)
                ) {
                  hasInvalidActivation = true;
                  break;
                }

                // Add connected nodes to the queue
                const neighbors = graph.get(currentNodeId) || new Set<string>();
                neighbors.forEach((neighborId) => {
                  if (!visited.has(neighborId)) {
                    visited.add(neighborId);
                    queue.push(neighborId);
                  }
                });
              }
            };

            // Start BFS from the output layer
            bfs(outputLayer.id);

            // Also check the reverse: if any activation layer connects to the output
            if (!hasInvalidActivation) {
              nonLinearActivations.forEach((activationLayer) => {
                bfs(activationLayer.id);
              });
            }

            if (hasInvalidActivation) {
              errors.push(
                `Output layer should not have an activation or should use a linear activation for California Housing dataset (regression)`
              );
            }
          }
        }
      }
    }

    // Validate ResNet architecture if ResNet blocks are present
    const resNetBlocks = nodes.filter((node) => node.type === "resnetblock");
    if (resNetBlocks.length > 0) {
      const resNetArchErrors = validateResNetArchitecture(
        resNetBlocks,
        edges,
        nodes
      );
      errors.push(...resNetArchErrors);
    }

    return errors;
  };

  // Function to validate ResNet architecture
  const validateResNetArchitecture = (
    resNetBlocks: Node[],
    edges: Edge<any>[],
    allNodes: Node[]
  ): ValidationErrors => {
    const errors: ValidationErrors = [];

    // Check for orphaned ResNet blocks (no incoming or outgoing connections)
    for (const block of resNetBlocks) {
      const incomingEdges = edges.filter((edge) => edge.target === block.id);
      const outgoingEdges = edges.filter((edge) => edge.source === block.id);

      if (incomingEdges.length === 0) {
        errors.push(
          `ResNet block ${block.id} must have at least one incoming connection`
        );
      }

      if (outgoingEdges.length === 0) {
        errors.push(
          `ResNet block ${block.id} must have at least one outgoing connection`
        );
      }
    }

    // Check for ResNet design patterns
    // Pattern 1: Check for excessive stacking without dimension reduction
    let consecutiveBlocksWithoutDimReduction = 0;
    const orderedBlocks = [...resNetBlocks].sort(
      (a, b) => a.position.y - b.position.y
    );

    for (let i = 1; i < orderedBlocks.length; i++) {
      const prevBlock = orderedBlocks[i - 1];
      const currBlock = orderedBlocks[i];

      // Check if there's an edge connecting them
      const isConnected = edges.some(
        (edge) => edge.source === prevBlock.id && edge.target === currBlock.id
      );

      if (isConnected) {
        // Check if no dimension reduction occurs
        if (currBlock.data.stride[0] === 1 && currBlock.data.stride[1] === 1) {
          consecutiveBlocksWithoutDimReduction++;

          // Warn if too many consecutive blocks without dimension changes
          if (consecutiveBlocksWithoutDimReduction > 10) {
            errors.push(
              `Warning: There are more than 10 consecutive ResNet blocks without dimension reduction, which may be inefficient`
            );
            break; // Only report this once
          }
        } else {
          consecutiveBlocksWithoutDimReduction = 0;
        }
      }
    }

    // Check for the presence of BatchNormalization with ResNet (recommended practice)
    const hasBatchNorm = allNodes.some(
      (node) => node.type === "batchnormalization"
    );
    if (!hasBatchNorm && resNetBlocks.length > 0) {
      errors.push(
        "Warning: ResNet architectures typically perform better with BatchNormalization layers"
      );
    }

    // For image datasets, check specific ResNet architecture patterns
    if (selectedDataset === "MNIST" || selectedDataset === "CIFAR-10") {
      // Check for proper ordering: typically Conv -> ResNet -> ResNet -> ... -> GlobalAveragePool/Flatten
      const convExists = allNodes.some((node) => node.type === "convolution");
      const flattenExists = allNodes.some((node) => node.type === "flatten");
      const globalAvgPoolExists = allNodes.some(
        (node) => node.type === "globalaveragepool"
      );

      // If no convolution layer before ResNet blocks, suggest adding one (but don't enforce)
      if (resNetBlocks.length > 0 && !convExists) {
        errors.push(
          "Suggestion: Typical ResNet architecture starts with a Convolution layer before ResNet blocks"
        );
      }

      // Suggest GlobalAveragePooling for ResNet models if neither exists
      if (resNetBlocks.length > 0 && !flattenExists && !globalAvgPoolExists) {
        errors.push(
          "ResNet architectures typically use GlobalAveragePooling before the final Dense layers"
        );
      }

      // If using Flatten with ResNet, suggest GlobalAveragePooling as a better alternative
      if (resNetBlocks.length > 0 && flattenExists && !globalAvgPoolExists) {
        errors.push(
          "Suggestion: Consider using GlobalAveragePooling instead of Flatten for ResNet architectures - it's more parameter-efficient"
        );
      }

      // Check if there's a path from ResNet blocks to a Flatten or GlobalAveragePooling layer
      let pathToFlattenOrGlobalPoolFound = false;

      if (flattenExists || globalAvgPoolExists) {
        // Build adjacency list for graph traversal
        const adjacencyList = new Map<string, string[]>();
        edges.forEach((edge) => {
          if (!adjacencyList.has(edge.source)) {
            adjacencyList.set(edge.source, []);
          }
          adjacencyList.get(edge.source)!.push(edge.target);
        });

        // Depth First Search function to find path to flatten or global average pool
        const dfs = (nodeId: string, visited = new Set<string>()): boolean => {
          if (visited.has(nodeId)) return false;
          visited.add(nodeId);

          const node = allNodes.find((n) => n.id === nodeId);
          if (
            node &&
            (node.type === "flatten" || node.type === "globalaveragepool")
          )
            return true;

          const neighbors = adjacencyList.get(nodeId) || [];
          for (const neighbor of neighbors) {
            if (dfs(neighbor, visited)) return true;
          }

          return false;
        };

        // Check if any ResNet block has a path to flatten or global average pool
        for (const block of resNetBlocks) {
          if (dfs(block.id)) {
            pathToFlattenOrGlobalPoolFound = true;
            break;
          }
        }

        if (!pathToFlattenOrGlobalPoolFound) {
          errors.push(
            "ResNet blocks for image data should eventually connect to a GlobalAveragePooling or Flatten layer"
          );
        }
      } else {
        errors.push(
          "Image classification models typically need a Flatten layer after convolutional/ResNet blocks"
        );
      }
    }

    // Check for a proper final dense layer with softmax (for classification datasets)
    if (
      selectedDataset === "MNIST" ||
      selectedDataset === "CIFAR-10" ||
      selectedDataset === "Iris"
    ) {
      const outputLayer = allNodes.find((node) => node.type === "output");
      const finalDense = allNodes.find(
        (node) =>
          node.type === "dense" &&
          edges.some(
            (edge) =>
              edge.source === node.id &&
              outputLayer &&
              edge.target === outputLayer.id
          )
      );

      if (!finalDense) {
        errors.push(
          "Suggestion: Classification models typically include a Dense layer before the Output layer"
        );
      }
    }

    return errors;
  };

  // Display validation errors
  const displayValidationErrors = (errors: ValidationErrors) => {
    // Only show errors if there are any
    if (errors.length > 0) {
      setShowValidationErrors(true);
      // Auto-hide after 5 seconds
      setTimeout(() => {
        setShowValidationErrors(false);
      }, 5000);
    }
  };

  // Helper function to get icon based on layer type
  const getLayerIcon = (layerType: string): string => {
    switch (layerType.toLowerCase()) {
      case "input":
        return "fa-sign-in-alt";
      case "output":
        return "fa-sign-out-alt";
      case "dense":
        return "fa-network-wired";
      case "convolution":
        return "fa-border-all";
      case "maxpooling":
        return "fa-filter";
      case "globalaveragepool":
        return "fa-compress"; // Using a compress icon for GlobalAveragePooling
      case "flatten":
        return "fa-compress-arrows-alt";
      case "dropout":
        return "fa-random";
      case "batchnormalization":
        return "fa-balance-scale";
      case "attention":
        return "fa-brain";
      case "resnetblock":
        return "fa-code-branch";
      case "addlayer":
        return "fa-plus-circle"; // Plus circle icon for Add Layer
      case "activation":
        return "fa-bolt"; // Bolt icon for Activation Layer
      case "customblock":
        return "fa-cogs"; // Cogs icon for Custom Block
      default:
        return "fa-cube";
    }
  };

  // Render the content based on the active sidebar option
  const renderSidebarContent = () => {
    switch (activeSidebarOption) {
      case "layers":
        return (
          <div className="sidebar-content-section">
            <h3>Available Layers</h3>
            <div className="layer-list">
              {/* Input and Output Layers */}
              <div className="layer-item input-layer">
                <span>
                  <i className="fas fa-sign-in-alt"></i> Input
                </span>
                <button
                  className="add-button"
                  onClick={() => addLayer("Input")}
                >
                  <i className="fas fa-plus"></i>
                </button>
              </div>
              <div className="layer-item output-layer">
                <span>
                  <i className="fas fa-sign-out-alt"></i> Output
                </span>
                <button
                  className="add-button"
                  onClick={() => addLayer("Output")}
                >
                  <i className="fas fa-plus"></i>
                </button>
              </div>
              {/* Basic Layers */}
              <div className="layer-item dense-layer">
                <span>
                  <i className="fas fa-network-wired"></i> Dense
                </span>
                <button
                  className="add-button"
                  onClick={() => addLayer("Dense")}
                >
                  <i className="fas fa-plus"></i>
                </button>
              </div>
              <div className="layer-item convolution-layer">
                <span>
                  <i className="fas fa-border-all"></i> Convolution
                </span>
                <button
                  className="add-button"
                  onClick={() => addLayer("Convolution")}
                >
                  <i className="fas fa-plus"></i>
                </button>
              </div>
              <div className="layer-item maxpooling-layer">
                <span>
                  <i className="fas fa-filter"></i> MaxPooling
                </span>
                <button
                  className="add-button"
                  onClick={() => addLayer("MaxPooling")}
                >
                  <i className="fas fa-plus"></i>
                </button>
              </div>
              <div className="layer-item globalaveragepool-layer">
                <span>
                  <i className="fas fa-compress"></i> Global Avg Pool
                </span>
                <button
                  className="add-button"
                  onClick={() => addLayer("GlobalAveragePool")}
                >
                  <i className="fas fa-plus"></i>
                </button>
              </div>
              <div className="layer-item flatten-layer">
                <span>
                  <i className="fas fa-compress-arrows-alt"></i> Flatten
                </span>
                <button
                  className="add-button"
                  onClick={() => addLayer("Flatten")}
                >
                  <i className="fas fa-plus"></i>
                </button>
              </div>
              <div className="layer-item dropout-layer">
                <span>
                  <i className="fas fa-random"></i> Dropout
                </span>
                <button
                  className="add-button"
                  onClick={() => addLayer("Dropout")}
                >
                  <i className="fas fa-plus"></i>
                </button>
              </div>
              <div className="layer-item batchnorm-layer">
                <span>
                  <i className="fas fa-balance-scale"></i> BatchNormalization
                </span>
                <button
                  className="add-button"
                  onClick={() => addLayer("BatchNormalization")}
                >
                  <i className="fas fa-plus"></i>
                </button>
              </div>

              <div className="layer-item attention-layer">
                <span>
                  <i className="fas fa-brain"></i> Attention
                </span>
                <button
                  className="add-button"
                  onClick={() => addLayer("Attention")}
                >
                  <i className="fas fa-plus"></i>
                </button>
              </div>

              <div className="layer-item add-layer">
                <span>
                  <i className="fas fa-plus-circle"></i> Add Layer
                </span>
                <button
                  className="add-button"
                  onClick={() => addLayer("AddLayer")}
                >
                  <i className="fas fa-plus"></i>
                </button>
              </div>

              {/* Custom Layers */}
              {customLayers.map((customLayer) => (
                <div
                  key={customLayer.id}
                  className="layer-item custom-layer-item"
                >
                  <span>
                    <i className={customLayer.icon}></i> {customLayer.name}
                  </span>
                  <button
                    className="add-button"
                    onClick={() =>
                      addLayer("customblock", {
                        blockName: customLayer.name,
                        layers: customLayer.layers,
                        customBlock: true,
                      })
                    }
                  >
                    <i className="fas fa-plus"></i>
                  </button>
                </div>
              ))}

              {/* Custom Layer Button */}
              <div className="layer-item custom-layer">
                <span>
                  <i className="fas fa-cogs"></i> Custom Layer
                </span>
                <button
                  className="add-button"
                  onClick={() => setShowCustomLayerModal(true)}
                >
                  <i className="fas fa-plus"></i>
                </button>
              </div>
            </div>
          </div>
        );
      case "templates":
        return (
          <div className="sidebar-content-section">
            <h3>
              <i className="fas fa-shapes"></i> Model Templates
            </h3>
            <div className="template-list">
              {/* Save as Template Card */}
              <div
                className="template-item"
                style={{ borderLeftColor: "#22c55e" }}
              >
                <span>
                  <i className="fas fa-save" style={{ color: "#22c55e" }}></i>{" "}
                  Save Current Model as Template
                </span>
                <button
                  className="add-button"
                  style={{ color: "#22c55e", border: "1.5px solid #22c55e" }}
                  onClick={handleSaveAsTemplate}
                >
                  <i className="fas fa-plus"></i>
                </button>
              </div>
              {/* Render session templates as cards */}
              {sessionTemplates.map((template, idx) => (
                <div
                  className="template-item"
                  style={{ borderLeftColor: "#6366f1" }}
                  key={template.name + idx}
                >
                  <span>
                    <i className="fas fa-user" style={{ color: "#6366f1" }}></i>{" "}
                    {template.name}
                  </span>
                  <button
                    className="add-button"
                    style={{ color: "#6366f1", border: "1.5px solid #6366f1" }}
                    onClick={() => loadSessionTemplate(template)}
                  >
                    <i className="fas fa-plus"></i>
                  </button>
                </div>
              ))}
              {/* Existing hardcoded templates... */}
              <div className="template-item simple-feedforward">
                <span>
                  <i className="fas fa-network-wired"></i> Simple Feedforward
                </span>
                <button
                  className="add-button"
                  onClick={() => loadTemplate("Simple Feedforward")}
                >
                  <i className="fas fa-plus"></i>
                </button>
              </div>
              <div className="template-item">
                <span>
                  <i className="fas fa-border-all"></i> Convolutional Network
                </span>
                <button
                  className="add-button"
                  onClick={() => loadTemplate("Convolutional Network")}
                >
                  <i className="fas fa-plus"></i>
                </button>
              </div>

              <div className="template-item transformer-template">
                <span>
                  <i className="fas fa-project-diagram"></i> Transformer Encoder
                </span>
                <button
                  className="add-button"
                  onClick={() => loadTemplate("Transformer Encoder")}
                >
                  <i className="fas fa-plus"></i>
                </button>
              </div>

              <div className="template-item regression-template">
                <span>
                  <i className="fas fa-chart-line"></i> Fully Connected
                  Regression
                </span>
                <button
                  className="add-button"
                  onClick={() => loadTemplate("Fully Connected Regression")}
                >
                  <i className="fas fa-plus"></i>
                </button>
              </div>
              <div className="template-item resnet-template">
                <span>
                  <i className="fas fa-code-branch"></i> ResNet-18
                </span>
                <button
                  className="add-button"
                  onClick={() => loadTemplate("ResNet-18")}
                >
                  <i className="fas fa-plus"></i>
                </button>
              </div>
            </div>
          </div>
        );
      case "activations":
        return (
          <div className="sidebar-content-section">
            <h3>
              <i className="fas fa-bolt"></i> Activation Functions
            </h3>
            <p className="sidebar-description">
              Add standalone activation layers to your model
            </p>
            <div className="activation-list">
              <div className="activation-item relu-activation">
                <span>
                  <i className="fas fa-bolt"></i> ReLU
                  <div className="activation-visual relu-visual"></div>
                </span>
                <button
                  className="add-button"
                  onClick={() =>
                    addLayer("Activation", {
                      function: "relu",
                      label: `ReLU Activation ${
                        nodes.filter(
                          (n) =>
                            n.type === "activation" &&
                            n.data.function === "relu"
                        ).length + 1
                      }`,
                    })
                  }
                  title="Add ReLU activation layer"
                >
                  <i className="fas fa-plus"></i>
                </button>
              </div>
              <div className="activation-item sigmoid-activation">
                <span>
                  <i className="fas fa-bolt"></i> Sigmoid
                  <div className="activation-visual sigmoid-visual"></div>
                </span>
                <button
                  className="add-button"
                  onClick={() =>
                    addLayer("Activation", {
                      function: "sigmoid",
                      label: `Sigmoid Activation ${
                        nodes.filter(
                          (n) =>
                            n.type === "activation" &&
                            n.data.function === "sigmoid"
                        ).length + 1
                      }`,
                    })
                  }
                  title="Add Sigmoid activation layer"
                >
                  <i className="fas fa-plus"></i>
                </button>
              </div>
              <div className="activation-item tanh-activation">
                <span>
                  <i className="fas fa-bolt"></i> Tanh
                  <div className="activation-visual tanh-visual"></div>
                </span>
                <button
                  className="add-button"
                  onClick={() => addLayer("Activation", { function: "tanh" })}
                  title="Add Tanh activation layer"
                >
                  <i className="fas fa-plus"></i>
                </button>
              </div>
              <div className="activation-item softmax-activation">
                <span>
                  <i className="fas fa-bolt"></i> Softmax
                  <div className="activation-visual softmax-visual"></div>
                </span>
                <button
                  className="add-button"
                  onClick={() =>
                    addLayer("Activation", { function: "softmax" })
                  }
                  title="Add Softmax activation layer"
                >
                  <i className="fas fa-plus"></i>
                </button>
              </div>
              <div className="activation-item leaky-relu-activation">
                <span>
                  <i className="fas fa-bolt"></i> Leaky ReLU
                  <div className="activation-visual leaky-relu-visual"></div>
                </span>
                <button
                  className="add-button"
                  onClick={() =>
                    addLayer("Activation", { function: "leaky_relu" })
                  }
                  title="Add Leaky ReLU activation layer"
                >
                  <i className="fas fa-plus"></i>
                </button>
              </div>
            </div>
          </div>
        );
      default:
        return (
          <div className="sidebar-content-section">
            <h3>Select a sidebar option</h3>
          </div>
        );
    }
  };

  // Render custom block parameters
  const renderCustomBlockParameters = () => {
    if (!selectedNode || selectedNode.type !== "customblock") {
      return null;
    }

    const layers = selectedNode.data.layers || [];
    console.log("Custom block layers:", layers);
    console.log(
      "Custom block layers structure:",
      JSON.stringify(layers, null, 2)
    );

    // Helper functions for parameter management
    const updateLayerParameter = (
      layerIndex: number,
      paramName: string,
      value: any
    ) => {
      if (!selectedNode) return;

      const updatedLayers = [...layers];
      if (!updatedLayers[layerIndex].parameters) {
        updatedLayers[layerIndex].parameters = {};
      }
      updatedLayers[layerIndex].parameters[paramName] = value;

      setNodes((nds) =>
        nds.map((node) =>
          node.id === selectedNode.id
            ? { ...node, data: { ...node.data, layers: updatedLayers } }
            : node
        )
      );

      setSelectedNode((prevNode) =>
        prevNode
          ? { ...prevNode, data: { ...prevNode.data, layers: updatedLayers } }
          : null
      );
    };

    const getLayerParameter = (
      layerIndex: number,
      paramName: string,
      defaultValue: any
    ) => {
      return layers[layerIndex]?.parameters?.[paramName] ?? defaultValue;
    };

    return (
      <>
        {/* Block Name Input */}
        <div className="param-group">
          <input
            type="text"
            value={selectedNode.data.label || ""}
            onChange={(e) => updateParameter("label", e.target.value)}
            placeholder="Block Name"
            className="param-input full-width"
          />
        </div>

        {/* Block Info */}
        <div className="param-group">
          <div className="block-info">
            <div className="info-row">
              <span className="info-label">Internal Layers:</span>
              <span className="info-value">{layers.length}</span>
            </div>
          </div>
        </div>

        {/* Layer Parameters */}
        {layers.map((layer: any, index: number) => {
          const layerType =
            layer.id?.toLowerCase() || layer.type?.toLowerCase();

          return (
            <div key={index} className="param-group">
              <div className="layer-section-header">
                <i className={`fas ${getLayerIcon(layerType)}`}></i>
                <label>
                  {index + 1}. {layer.name || layer.type || "Unknown Layer"}
                </label>
              </div>

              {/* Dense Layer Parameters */}
              {layerType === "dense" && (
                <>
                  <div className="param-subgroup">
                    <label>Neurons</label>
                    <div className="number-input-with-controls">
                      <input
                        type="number"
                        min="1"
                        value={getLayerParameter(index, "neurons", 64)}
                        onChange={(e) =>
                          updateLayerParameter(
                            index,
                            "neurons",
                            parseInt(e.target.value) || 64
                          )
                        }
                        className="param-input"
                      />
                      <div className="param-controls">
                        <button
                          className="control-btn"
                          onClick={() =>
                            updateLayerParameter(
                              index,
                              "neurons",
                              Math.max(
                                1,
                                getLayerParameter(index, "neurons", 64) - 1
                              )
                            )
                          }
                        >
                          <i className="fas fa-minus"></i>
                        </button>
                        <button
                          className="control-btn"
                          onClick={() =>
                            updateLayerParameter(
                              index,
                              "neurons",
                              getLayerParameter(index, "neurons", 64) + 1
                            )
                          }
                        >
                          <i className="fas fa-plus"></i>
                        </button>
                      </div>
                    </div>
                  </div>

                  <div className="param-subgroup">
                    <label>Activation</label>
                    <select
                      value={getLayerParameter(index, "activation", "relu")}
                      onChange={(e) =>
                        updateLayerParameter(
                          index,
                          "activation",
                          e.target.value
                        )
                      }
                      className="param-select"
                    >
                      <option value="relu">ReLU</option>
                      <option value="sigmoid">Sigmoid</option>
                      <option value="tanh">Tanh</option>
                      <option value="softmax">Softmax</option>
                      <option value="linear">Linear</option>
                    </select>
                  </div>
                </>
              )}

              {/* Convolution Layer Parameters */}
              {layerType === "convolution" && (
                <>
                  <div className="param-subgroup">
                    <label>Filters</label>
                    <div className="number-input-with-controls">
                      <input
                        type="number"
                        min="1"
                        value={getLayerParameter(index, "filters", 32)}
                        onChange={(e) =>
                          updateLayerParameter(
                            index,
                            "filters",
                            parseInt(e.target.value) || 32
                          )
                        }
                        className="param-input"
                      />
                      <div className="param-controls">
                        <button
                          className="control-btn"
                          onClick={() =>
                            updateLayerParameter(
                              index,
                              "filters",
                              Math.max(
                                1,
                                getLayerParameter(index, "filters", 32) - 1
                              )
                            )
                          }
                        >
                          <i className="fas fa-minus"></i>
                        </button>
                        <button
                          className="control-btn"
                          onClick={() =>
                            updateLayerParameter(
                              index,
                              "filters",
                              getLayerParameter(index, "filters", 32) + 1
                            )
                          }
                        >
                          <i className="fas fa-plus"></i>
                        </button>
                      </div>
                    </div>
                  </div>

                  <div className="param-subgroup">
                    <label>Kernel Size</label>
                    <div className="dimension-input">
                      <input
                        type="number"
                        min="1"
                        value={
                          getLayerParameter(index, "kernelSize", [3, 3])[0]
                        }
                        onChange={(e) =>
                          updateLayerParameter(index, "kernelSize", [
                            parseInt(e.target.value) || 3,
                            getLayerParameter(index, "kernelSize", [3, 3])[1],
                          ])
                        }
                        className="param-input"
                      />
                      <span>×</span>
                      <input
                        type="number"
                        min="1"
                        value={
                          getLayerParameter(index, "kernelSize", [3, 3])[1]
                        }
                        onChange={(e) =>
                          updateLayerParameter(index, "kernelSize", [
                            getLayerParameter(index, "kernelSize", [3, 3])[0],
                            parseInt(e.target.value) || 3,
                          ])
                        }
                        className="param-input"
                      />
                    </div>
                  </div>

                  <div className="param-subgroup">
                    <label>Stride</label>
                    <div className="dimension-input">
                      <input
                        type="number"
                        min="1"
                        value={getLayerParameter(index, "stride", [1, 1])[0]}
                        onChange={(e) =>
                          updateLayerParameter(index, "stride", [
                            parseInt(e.target.value) || 1,
                            getLayerParameter(index, "stride", [1, 1])[1],
                          ])
                        }
                        className="param-input"
                      />
                      <span>×</span>
                      <input
                        type="number"
                        min="1"
                        value={getLayerParameter(index, "stride", [1, 1])[1]}
                        onChange={(e) =>
                          updateLayerParameter(index, "stride", [
                            getLayerParameter(index, "stride", [1, 1])[0],
                            parseInt(e.target.value) || 1,
                          ])
                        }
                        className="param-input"
                      />
                    </div>
                  </div>

                  <div className="param-subgroup">
                    <label>Padding</label>
                    <select
                      value={getLayerParameter(index, "padding", "valid")}
                      onChange={(e) =>
                        updateLayerParameter(index, "padding", e.target.value)
                      }
                      className="param-select"
                    >
                      <option value="valid">Valid</option>
                      <option value="same">Same</option>
                    </select>
                  </div>
                </>
              )}

              {/* Dropout Layer Parameters */}
              {layerType === "dropout" && (
                <div className="param-subgroup">
                  <label>Rate</label>
                  <div className="slider-with-value">
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={getLayerParameter(index, "rate", 0.2)}
                      onChange={(e) =>
                        updateLayerParameter(
                          index,
                          "rate",
                          parseFloat(e.target.value) || 0.2
                        )
                      }
                      className="param-slider"
                    />
                    <input
                      type="number"
                      min="0"
                      max="1"
                      step="0.01"
                      value={getLayerParameter(index, "rate", 0.2)}
                      onChange={(e) =>
                        updateLayerParameter(
                          index,
                          "rate",
                          parseFloat(e.target.value) || 0.2
                        )
                      }
                      className="param-input small"
                    />
                  </div>
                </div>
              )}

              {/* MaxPooling Layer Parameters */}
              {layerType === "maxpooling" && (
                <>
                  <div className="param-item">
                    <label>Pool Size</label>
                    <div className="dimension-control">
                      <div className="number-control">
                        <button
                          className="control-btn"
                          onClick={() =>
                            updateParameter("poolSize", [
                              (selectedNode.data.poolSize?.[0] || 2) - 1,
                              selectedNode.data.poolSize?.[1] || 2,
                            ])
                          }
                        >
                          <i className="fas fa-minus"></i>
                        </button>
                        <input
                          type="number"
                          min="1"
                          value={selectedNode.data.poolSize?.[0] || "2"}
                          onChange={(e) =>
                            updateParameter("poolSize", [
                              parseInt(e.target.value) || 2,
                              selectedNode.data.poolSize?.[1] || 2,
                            ])
                          }
                          className="number-input"
                        />
                        <button
                          className="control-btn"
                          onClick={() =>
                            updateParameter("poolSize", [
                              (selectedNode.data.poolSize?.[0] || 2) + 1,
                              selectedNode.data.poolSize?.[1] || 2,
                            ])
                          }
                        >
                          <i className="fas fa-plus"></i>
                        </button>
                      </div>
                      <span>×</span>
                      <div className="number-control">
                        <button
                          className="control-btn"
                          onClick={() =>
                            updateParameter("poolSize", [
                              selectedNode.data.poolSize?.[0] || 2,
                              (selectedNode.data.poolSize?.[1] || 2) - 1,
                            ])
                          }
                        >
                          <i className="fas fa-minus"></i>
                        </button>
                        <input
                          type="number"
                          min="1"
                          value={selectedNode.data.poolSize?.[1] || "2"}
                          onChange={(e) =>
                            updateParameter("poolSize", [
                              selectedNode.data.poolSize?.[0] || 2,
                              parseInt(e.target.value) || 2,
                            ])
                          }
                          className="number-input"
                        />
                        <button
                          className="control-btn"
                          onClick={() =>
                            updateParameter("poolSize", [
                              selectedNode.data.poolSize?.[0] || 2,
                              (selectedNode.data.poolSize?.[1] || 2) + 1,
                            ])
                          }
                        >
                          <i className="fas fa-plus"></i>
                        </button>
                      </div>
                    </div>
                  </div>

                  <div className="param-item">
                    <label>Stride</label>
                    <div className="dimension-control">
                      <div className="number-control">
                        <button
                          className="control-btn"
                          onClick={() =>
                            updateParameter("stride", [
                              (selectedNode.data.stride?.[0] || 2) - 1,
                              selectedNode.data.stride?.[1] || 2,
                            ])
                          }
                        >
                          <i className="fas fa-minus"></i>
                        </button>
                        <input
                          type="number"
                          min="1"
                          value={selectedNode.data.stride?.[0] || "2"}
                          onChange={(e) =>
                            updateParameter("stride", [
                              +e.target.value,
                              selectedNode.data.stride?.[1] || 2,
                            ])
                          }
                          className="number-input"
                        />
                        <button
                          className="control-btn"
                          onClick={() =>
                            updateParameter("stride", [
                              (selectedNode.data.stride?.[0] || 2) + 1,
                              selectedNode.data.stride?.[1] || 2,
                            ])
                          }
                        >
                          <i className="fas fa-plus"></i>
                        </button>
                      </div>
                      <span>×</span>
                      <div className="number-control">
                        <button
                          className="control-btn"
                          onClick={() =>
                            updateParameter("stride", [
                              selectedNode.data.stride?.[0] || 2,
                              (selectedNode.data.stride?.[1] || 2) - 1,
                            ])
                          }
                        >
                          <i className="fas fa-minus"></i>
                        </button>
                        <input
                          type="number"
                          min="1"
                          value={selectedNode.data.stride?.[1] || "2"}
                          onChange={(e) =>
                            updateParameter("stride", [
                              selectedNode.data.stride?.[0] || 2,
                              +e.target.value,
                            ])
                          }
                          className="number-input"
                        />
                        <button
                          className="control-btn"
                          onClick={() =>
                            updateParameter("stride", [
                              selectedNode.data.stride?.[0] || 2,
                              (selectedNode.data.stride?.[1] || 2) + 1,
                            ])
                          }
                        >
                          <i className="fas fa-plus"></i>
                        </button>
                      </div>
                    </div>
                  </div>

                  <div className="param-item">
                    <label>Padding</label>
                    <select
                      value={selectedNode.data.padding || "valid"}
                      onChange={(e) =>
                        updateParameter("padding", e.target.value)
                      }
                      className="param-select"
                    >
                      <option value="valid">Valid</option>
                      <option value="same">Same</option>
                    </select>
                  </div>
                </>
              )}

              {/* BatchNormalization Layer Parameters */}
              {layerType === "batchnormalization" && (
                <>
                  <div className="param-subgroup">
                    <label>Momentum</label>
                    <div className="slider-with-value">
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={getLayerParameter(index, "momentum", 0.99)}
                        onChange={(e) =>
                          updateLayerParameter(
                            index,
                            "momentum",
                            parseFloat(e.target.value) || 0.99
                          )
                        }
                        className="param-slider"
                      />
                      <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.01"
                        value={getLayerParameter(index, "momentum", 0.99)}
                        onChange={(e) =>
                          updateLayerParameter(
                            index,
                            "momentum",
                            parseFloat(e.target.value) || 0.99
                          )
                        }
                        className="param-input small"
                      />
                    </div>
                  </div>

                  <div className="param-subgroup">
                    <label>Epsilon</label>
                    <input
                      type="number"
                      min="0.0001"
                      max="0.1"
                      step="0.0001"
                      value={getLayerParameter(index, "epsilon", 0.001)}
                      onChange={(e) =>
                        updateLayerParameter(
                          index,
                          "epsilon",
                          parseFloat(e.target.value) || 0.001
                        )
                      }
                      className="param-input"
                    />
                  </div>
                </>
              )}

              {/* Attention Layer Parameters */}
              {layerType === "attention" && (
                <>
                  <div className="param-subgroup">
                    <label>Heads</label>
                    <div className="number-input-with-controls">
                      <input
                        type="number"
                        min="1"
                        value={getLayerParameter(index, "heads", 8)}
                        onChange={(e) =>
                          updateLayerParameter(
                            index,
                            "heads",
                            parseInt(e.target.value) || 8
                          )
                        }
                        className="param-input"
                      />
                      <div className="param-controls">
                        <button
                          className="control-btn"
                          onClick={() =>
                            updateLayerParameter(
                              index,
                              "heads",
                              Math.max(
                                1,
                                getLayerParameter(index, "heads", 8) - 1
                              )
                            )
                          }
                        >
                          <i className="fas fa-minus"></i>
                        </button>
                        <button
                          className="control-btn"
                          onClick={() =>
                            updateLayerParameter(
                              index,
                              "heads",
                              getLayerParameter(index, "heads", 8) + 1
                            )
                          }
                        >
                          <i className="fas fa-plus"></i>
                        </button>
                      </div>
                    </div>
                  </div>

                  <div className="param-subgroup">
                    <label>Key Dimension</label>
                    <div className="number-input-with-controls">
                      <input
                        type="number"
                        min="1"
                        value={getLayerParameter(index, "keyDim", 64)}
                        onChange={(e) =>
                          updateLayerParameter(
                            index,
                            "keyDim",
                            parseInt(e.target.value) || 64
                          )
                        }
                        className="param-input"
                      />
                      <div className="param-controls">
                        <button
                          className="control-btn"
                          onClick={() =>
                            updateLayerParameter(
                              index,
                              "keyDim",
                              Math.max(
                                1,
                                getLayerParameter(index, "keyDim", 64) - 1
                              )
                            )
                          }
                        >
                          <i className="fas fa-minus"></i>
                        </button>
                        <button
                          className="control-btn"
                          onClick={() =>
                            updateLayerParameter(
                              index,
                              "keyDim",
                              getLayerParameter(index, "keyDim", 64) + 1
                            )
                          }
                        >
                          <i className="fas fa-plus"></i>
                        </button>
                      </div>
                    </div>
                  </div>

                  <div className="param-subgroup">
                    <label>Dropout Rate</label>
                    <div className="slider-with-value">
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={getLayerParameter(index, "dropout", 0.1)}
                        onChange={(e) =>
                          updateLayerParameter(
                            index,
                            "dropout",
                            parseFloat(e.target.value) || 0.1
                          )
                        }
                        className="param-slider"
                      />
                      <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.01"
                        value={getLayerParameter(index, "dropout", 0.1)}
                        onChange={(e) =>
                          updateLayerParameter(
                            index,
                            "dropout",
                            parseFloat(e.target.value) || 0.1
                          )
                        }
                        className="param-input small"
                      />
                    </div>
                  </div>
                </>
              )}

              {/* Activation Layer Parameters */}
              {layerType === "activation" && (
                <div className="param-subgroup">
                  <label>Activation Function</label>
                  <select
                    value={getLayerParameter(index, "function", "relu")}
                    onChange={(e) =>
                      updateLayerParameter(index, "function", e.target.value)
                    }
                    className="param-select"
                  >
                    <option value="relu">ReLU</option>
                    <option value="sigmoid">Sigmoid</option>
                    <option value="tanh">Tanh</option>
                    <option value="softmax">Softmax</option>
                    <option value="linear">Linear</option>
                    <option value="leaky_relu">Leaky ReLU</option>
                    <option value="elu">ELU</option>
                    <option value="swish">Swish</option>
                  </select>
                </div>
              )}

              {/* Layers with no parameters */}
              {(layerType === "flatten" ||
                layerType === "globalaveragepool" ||
                layerType === "addlayer") && (
                <div className="param-subgroup">
                  <p className="param-info">
                    This layer has no configurable parameters.
                  </p>
                </div>
              )}
            </div>
          );
        })}
      </>
    );
  };

  // Render the node parameters panel

  // Render visualization based on selected option
  const renderVisualization = () => {
    console.log("Rendering visualization:", selectedVisualization, {
      heatmapStatus: heatmapImage
        ? `Available (${heatmapImage.substring(0, 50)}...)`
        : "Not available",
      residualStatus: {
        residuals: residuals.length,
        predictedValues: predictedValues.length,
      },
      datasetSelected: selectedDataset,
      chartKey: chartKey,
    });

    switch (selectedVisualization) {
      case "accuracy":
        return (
          <div className="visualization-container">
            <h4>Accuracy Over Time</h4>
            <div
              className="chart-container"
              style={{
                height: "280px",
                overflow: "visible",
              }}
            >
              {isTraining || labels.length > 0 ? (
                <Line
                  key={`accuracy-chart-${chartKey}`}
                  data={accuracyChartData}
                  options={chartOptions}
                />
              ) : (
                <div className="empty-chart">
                  <i className="fas fa-chart-line"></i>
                  <p>Start training to see accuracy metrics</p>
                </div>
              )}
            </div>
          </div>
        );
      case "loss":
        return (
          <div className="visualization-container">
            <h4>Loss Over Time</h4>
            <div
              className="chart-container"
              style={{
                height: "280px",
                overflow: "visible",
              }}
            >
              {isTraining || labels.length > 0 ? (
                <Line
                  key={`loss-chart-${chartKey}`}
                  data={lossChartData}
                  options={chartOptions}
                />
              ) : (
                <div className="empty-chart">
                  <i className="fas fa-chart-area"></i>
                  <p>Start training to see loss metrics</p>
                </div>
              )}
            </div>
          </div>
        );
      case "confusion_matrix":
        return (
          <div className="visualization-container">
            <h4>Confusion Matrix</h4>
            {confusionMatrix.length > 0 ? (
              <>
                <p className="matrix-info">
                  Showing confusion matrix for {selectedDataset} dataset
                </p>
                <ConfusionMatrix
                  matrix={confusionMatrix}
                  labels={getClassLabels(selectedDataset)}
                />
              </>
            ) : (
              <div className="empty-chart">
                <i className="fas fa-table"></i>
                <p>
                  Confusion matrix visualization will be shown here after
                  training
                </p>
              </div>
            )}
          </div>
        );
      case "heatmap":
        console.log(
          "Rendering heatmap visualization with image:",
          heatmapImage ? "available" : "not available",
          "- chartKey:",
          chartKey
        );
        return (
          <div className="visualization-container">
            <h4>Multicollinearity Heatmap</h4>
            <div
              className="chart-container"
              style={{
                height: "280px", // Changed from auto to 280px to match other visualization containers
                minHeight: "280px",
                overflow: "visible",
              }}
            >
              {heatmapImage ? (
                <div className="heatmap-container">
                  <img
                    key={`heatmap-${chartKey}`}
                    src={heatmapImage}
                    alt="Multicollinearity Heatmap"
                    style={{
                      width: "100%",
                      maxWidth: "800px",
                      margin: "0 auto",
                      display: "block",
                    }}
                    onLoad={() =>
                      console.log("Heatmap image loaded successfully")
                    }
                    onError={(e) => {
                      console.error("Error loading heatmap image:", e);
                      (e.target as HTMLImageElement).style.display = "none";
                    }}
                  />
                </div>
              ) : (
                <div className="empty-chart">
                  <p>Train the model to generate a multicollinearity heatmap</p>
                </div>
              )}
            </div>
          </div>
        );
      case "class_distribution":
        return (
          <div className="visualization-container">
            <h4>Class Distribution</h4>
            <div className="placeholder-visualization">
              <p>Class distribution visualization will be shown here</p>
              <div className="distribution-placeholder"></div>
            </div>
          </div>
        );
      case "feature_importance":
        return (
          <div className="visualization-container">
            <h4>Feature Importance</h4>
            <div className="placeholder-visualization">
              <p>
                Feature importance visualization will be shown here after
                training
              </p>
              <div className="feature-importance-placeholder"></div>
            </div>
          </div>
        );
      case "scatter_plot":
        return (
          <div className="visualization-container">
            <h4>Scatter Plot</h4>
            <div className="placeholder-visualization">
              <p>Scatter plot visualization will be shown here</p>
              <div className="scatter-placeholder"></div>
            </div>
          </div>
        );
      case "pca":
        return (
          <div className="visualization-container">
            <h4>PCA Visualization</h4>
            <div className="placeholder-visualization">
              <p>PCA visualization will be shown here after training</p>
              <div className="pca-placeholder"></div>
            </div>
          </div>
        );
      case "roc_curve":
        return (
          <div className="visualization-container">
            <h4>ROC Curve</h4>
            <div className="placeholder-visualization">
              <p>ROC curve will be shown here after training</p>
              <div className="roc-placeholder"></div>
            </div>
          </div>
        );
      case "residual_plot":
        console.log(
          "Rendering residual plot with data - Residuals:",
          residuals?.length || 0,
          "Predicted values:",
          predictedValues?.length || 0,
          "- chartKey:",
          chartKey
        );

        // Create residual plot data
        const residualPlotData = {
          datasets: [
            {
              label: "Residuals",
              data: predictedValues.map((pred, index) => ({
                x: pred,
                y: residuals[index],
              })),
              backgroundColor: "rgba(255, 159, 64, 0.2)",
              borderColor: "rgba(255, 159, 64, 1)",
              borderWidth: 1,
              pointRadius: 4,
              pointHoverRadius: 6,
              showLine: false,
            },
          ],
        };

        const residualPlotOptions = {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: true },
            tooltip: { enabled: true },
          },
          scales: {
            x: {
              type: "linear" as const,
              position: "bottom" as const,
              title: {
                display: true,
                text: "Predicted Values",
              },
            },
            y: {
              title: {
                display: true,
                text: "Residuals (Actual - Predicted)",
              },
            },
          },
        };

        return (
          <div className="visualization-container">
            <h4>Residual Plot</h4>
            <div
              className="chart-container"
              style={{
                height: "280px", // Changed from 400px to 280px to match other visualization containers
                minHeight: "280px",
                overflow: "visible",
              }}
            >
              {residuals.length > 0 && predictedValues.length > 0 ? (
                <div className="residual-plot-container">
                  <Scatter
                    key={`residual-plot-${chartKey}`}
                    data={residualPlotData}
                    options={residualPlotOptions}
                  />
                </div>
              ) : (
                <div className="empty-chart">
                  <p>Train the model to generate a residual plot</p>
                </div>
              )}
            </div>
          </div>
        );
      case "feature_maps":
        return (
          <div className="visualization-container">
            <h4>Feature Maps</h4>
            <div className="placeholder-visualization">
              <p>Feature maps will be shown here after training</p>
              <div className="feature-maps-placeholder"></div>
            </div>
          </div>
        );
      default:
        return (
          <div className="visualization-placeholder">
            <p>Select a visualization type</p>
          </div>
        );
    }
  };

  // Confusion Matrix Component
  const ConfusionMatrix = ({
    matrix,
    labels,
  }: {
    matrix: number[][];
    labels: string[];
  }) => {
    // Calculate total number of samples
    const totalSamples = matrix.flat().reduce((sum, value) => sum + value, 0);

    // Calculate maximum proportion for opacity scaling
    const maxProportion =
      totalSamples > 0 ? Math.max(...matrix.flat()) / totalSamples : 0;

    return (
      <div className="confusion-matrix-container">
        <table className="confusion-matrix-table">
          <thead>
            <tr>
              <th style={{ width: "40px" }}></th>
              {labels.map((label, index) => (
                <th
                  key={`header-${index}`}
                  className="matrix-header"
                  style={{ width: `${100 / (labels.length + 1)}%` }}
                >
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, rowIndex) => (
              <tr key={`row-${rowIndex}`}>
                <td className="matrix-header">{labels[rowIndex]}</td>
                {row.map((value, colIndex) => {
                  // Calculate proportion as percentage
                  const proportion =
                    totalSamples > 0 ? value / totalSamples : 0;
                  const percentage = (proportion * 100).toFixed(1);

                  // Calculate opacity based on proportion for visual intensity
                  const opacity =
                    maxProportion > 0 ? proportion / maxProportion : 0;

                  return (
                    <td
                      key={`cell-${rowIndex}-${colIndex}`}
                      className="matrix-cell"
                      style={{
                        backgroundColor: `rgba(76, 175, 80, ${opacity})`,
                        color: opacity > 0.5 ? "#fff" : "#000",
                      }}
                      title={`${value} samples (${percentage}%)`}
                    >
                      {percentage}%
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  // Add the handleExport function
  const handleExport = async (format: string) => {
    if (!selectedDataset) {
      showToast("⚠️ Please select a dataset before exporting.", "error");
      return;
    }

    // Check if model is saved
    if (!isModelSaved) {
      const saveFirst = window.confirm(
        "The model needs to be saved before exporting. Save now?"
      );
      if (saveFirst) {
        try {
          setExportStatusMessage("Saving model before export...");
          await handleSaveModel();
          if (!isModelSaved) {
            setExportStatusMessage("Failed to save model, export canceled.");
            setTimeout(() => setExportStatusMessage(""), 5000);
            return; // If saving failed, stop export
          }
        } catch (error) {
          console.error("Error saving model:", error);
          setExportStatusMessage("Failed to save model, export canceled.");
          setTimeout(() => setExportStatusMessage(""), 5000);
          return;
        }
      } else {
        return; // User chose not to save, stop export
      }
    }

    const formatDisplayName =
      format === "savedmodel"
        ? "SavedModel"
        : format === "ipynb"
        ? "Jupyter Notebook"
        : format === "py"
        ? "Python"
        : format.toUpperCase();

    setExportStatusMessage(`Exporting model as ${formatDisplayName}...`);

    try {
      const response = await axios.get(`${API_BASE_URL}/api/export/${format}`, {
        responseType: "blob", // Important for file download
        timeout: 30000, // 30 seconds timeout
      });

      // Verify the response blob is valid
      if (!response.data || response.data.size === 0) {
        throw new Error("Received empty file from server");
      }

      // Set correct file extension for each format
      let fileExtension;
      if (format === "savedmodel") {
        fileExtension = "zip";
      } else if (format === "pytorch") {
        fileExtension = "py"; // PyTorch model extension
      } else if (format === "py") {
        fileExtension = "py"; // Python script extension
      } else if (format === "ipynb") {
        fileExtension = "ipynb"; // Jupyter notebook extension
      } else {
        fileExtension = format;
      }

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `trained_model.${fileExtension}`);
      document.body.appendChild(link);
      link.click();

      // Clean up
      window.URL.revokeObjectURL(url);
      document.body.removeChild(link);

      setExportStatusMessage(
        `Model exported successfully as ${formatDisplayName}!`
      );

      // Clear status message after 5 seconds
      setTimeout(() => {
        setExportStatusMessage("");
      }, 5000);
    } catch (error) {
      console.error("Export failed:", error);
      let errorMessage = "Failed to export model.";

      if (error instanceof Error) {
        errorMessage = `Failed to export model as ${formatDisplayName}: ${error.message}`;
      } else {
        errorMessage = `Failed to export model as ${formatDisplayName}.`;
      }

      setExportStatusMessage(errorMessage);

      // Clear error message after 5 seconds
      setTimeout(() => {
        setExportStatusMessage("");
      }, 5000);
    }
  };

  // Add an event listener for the train model event
  useEffect(() => {
    const trainModelHandler = () => {
      // Quick check for dataset selection
      if (!selectedDataset) {
        showToast(
          "No dataset selected! Please select a dataset before training.",
          "error"
        );
        return;
      }

      // Check if model is saved
      if (!isModelSaved) {
        showToast(
          "Please save the model before training. Click 'Save Model' first.",
          "error"
        );
        return;
      }

      // Check if already training
      if (isTraining) {
        console.log("Already training, ignoring train request");
        return;
      }

      // Do basic pre-validation
      const validationIssue = preValidateModel();
      if (validationIssue) {
        console.warn("Cannot start training:", validationIssue);
        showToast(`Cannot start training: ${validationIssue}`, "error");
        return;
      }

      // Proceed with training
      console.log("All validation passed, starting training...");
      handleStartTraining();
    };

    window.addEventListener("trainModel", trainModelHandler);

    // Cleanup function
    return () => {
      window.removeEventListener("trainModel", trainModelHandler);
    };
  }, [selectedDataset, isTraining, nodes, edges, isModelSaved]); // Include relevant dependencies

  // Display export status message if it exists
  useEffect(() => {
    if (exportStatusMessage) {
      console.log("Export status:", exportStatusMessage);
    }
  }, [exportStatusMessage]);

  // Listen for stop training event from NavBar
  useEffect(() => {
    const stopTrainingHandler = () => {
      if (isTraining) {
        handleStopTraining();
      }
    };

    window.addEventListener("stopTraining", stopTrainingHandler);

    // Cleanup function
    return () => {
      window.removeEventListener("stopTraining", stopTrainingHandler);
    };
  }, [isTraining]);

  // Listen for export model event from NavBar
  useEffect(() => {
    const exportModelHandler = (event: CustomEvent) => {
      const format = event.detail?.format;
      if (!format) {
        console.error("Export format not specified");
        return;
      }

      console.log(`Export model event received with format: ${format}`);
      handleExport(format);
    };

    window.addEventListener("exportModel", exportModelHandler as EventListener);

    // Cleanup function
    return () => {
      window.removeEventListener(
        "exportModel",
        exportModelHandler as EventListener
      );
    };
  }, [selectedDataset, isModelSaved, isTraining]);

  // Ref to track current isModelSaved state for event handlers
  const isModelSavedRef = useRef<boolean>(false);

  // Update ref whenever isModelSaved changes
  useEffect(() => {
    isModelSavedRef.current = isModelSaved;
  }, [isModelSaved]);

  // Listen for checkModelSaved event from NavBar
  useEffect(() => {
    const checkModelSavedHandler = (event: CustomEvent) => {
      const currentIsModelSaved = isModelSavedRef.current;
      console.log(
        "📋 checkModelSaved event received, current isModelSaved:",
        currentIsModelSaved
      );
      const callback = event.detail?.callback;
      if (callback && typeof callback === "function") {
        console.log(
          "📋 Calling callback with isModelSaved:",
          currentIsModelSaved
        );
        callback(currentIsModelSaved);
      } else {
        console.warn("📋 No callback provided in checkModelSaved event");
      }
    };

    console.log("📋 Adding checkModelSaved event listener");
    window.addEventListener(
      "checkModelSaved",
      checkModelSavedHandler as EventListener
    );

    // Cleanup function
    return () => {
      console.log("📋 Removing checkModelSaved event listener");
      window.removeEventListener(
        "checkModelSaved",
        checkModelSavedHandler as EventListener
      );
    };
  }, []); // Removed isModelSaved dependency to prevent multiple listeners

  // Render the training metrics that will be displayed on the right
  const renderTrainingMetrics = () => {
    // Check if this is a regression dataset (California Housing)
    const isRegressionDataset = selectedDataset === "California Housing";

    return (
      <div className="training-metrics-card">
        <div className="metrics-header">
          <i className="fas fa-chart-bar"></i>
          <h3>
            {isTraining
              ? "Training in Progress"
              : trainingProgress.currentEpoch > 0
              ? "Training Complete"
              : "Training Status"}
          </h3>
          {isTraining && (
            <div className="training-pulse-indicator">
              <div className="pulse-dot"></div>
              <span>Live</span>
            </div>
          )}
        </div>

        {/* Progress Bar */}
        <div className="progress-container">
          <div className="progress-info">
            <div className="progress-label">
              {isTraining
                ? `Epoch ${trainingProgress.currentEpoch}/${trainingProgress.totalEpochs}`
                : `Epoch ${trainingProgress.currentEpoch}/${trainingProgress.totalEpochs}`}
            </div>
            <div className="progress-percentage">
              {isTraining || trainingProgress.currentEpoch > 0
                ? `${Math.round(
                    (trainingProgress.currentEpoch /
                      trainingProgress.totalEpochs) *
                      100
                  )}%`
                : "Not started"}
            </div>
          </div>
          <div className="progress-bar-container">
            <div
              className="progress-bar"
              style={{
                width:
                  isTraining || trainingProgress.currentEpoch > 0
                    ? `${Math.round(
                        (trainingProgress.currentEpoch /
                          trainingProgress.totalEpochs) *
                          100
                      )}%`
                    : "0%",
              }}
            >
              {isTraining && <div className="progress-stripes"></div>}
            </div>
          </div>
        </div>

        {/* Consolidated Metrics Section */}
        <div className="metrics-section">
          <div className="metrics-row">
            {/* Loss and Val Loss Metrics */}
            <div className="metric-compact-item">
              <div className="metric-icon">
                <i className="fas fa-chart-line"></i>
              </div>
              <div className="metric-content">
                <div className="metric-label">Loss</div>
                <div
                  className={`metric-value ${
                    trainingProgress.loss > 0 ? "warning" : "na"
                  }`}
                >
                  {trainingProgress.loss > 0
                    ? trainingProgress.loss.toFixed(4)
                    : "—"}
                </div>
              </div>
            </div>
            <div className="metric-compact-item">
              <div className="metric-icon">
                <i className="fas fa-chart-area"></i>
              </div>
              <div className="metric-content">
                <div className="metric-label">Val Loss</div>
                <div
                  className={`metric-value ${
                    trainingProgress.valLoss > 0 ? "warning" : "na"
                  }`}
                >
                  {trainingProgress.valLoss > 0
                    ? trainingProgress.valLoss.toFixed(4)
                    : "—"}
                </div>
              </div>
            </div>
          </div>

          {/* Conditional row for Regression or Classification metrics */}
          {((isRegressionDataset && (rmse !== null || r2 !== null)) ||
            (!isRegressionDataset &&
              (trainingProgress.accuracy > 0 ||
                trainingProgress.valAccuracy > 0))) && (
            <div className="metrics-row mt-2">
              {/* Regression Metrics */}
              {isRegressionDataset ? (
                <>
                  {rmse !== null && (
                    <div className="metric-compact-item">
                      <div className="metric-icon">
                        <i className="fas fa-ruler"></i>
                      </div>
                      <div className="metric-content">
                        <div className="metric-label">RMSE</div>
                        <div className="metric-value warning">
                          {rmse.toFixed(4)}
                        </div>
                      </div>
                    </div>
                  )}
                  {r2 !== null && (
                    <div className="metric-compact-item">
                      <div className="metric-icon">
                        <i className="fas fa-percentage"></i>
                      </div>
                      <div className="metric-content">
                        <div className="metric-label">R² Score</div>
                        <div className="metric-value good">{r2.toFixed(4)}</div>
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <>
                  <div className="metric-compact-item">
                    <div className="metric-icon">
                      <i className="fas fa-bullseye"></i>
                    </div>
                    <div className="metric-content">
                      <div className="metric-label">Accuracy</div>
                      <div
                        className={`metric-value ${
                          trainingProgress.accuracy > 0 ? "good" : "na"
                        }`}
                      >
                        {trainingProgress.accuracy > 0
                          ? (trainingProgress.accuracy * 100).toFixed(2) + "%"
                          : "—"}
                      </div>
                    </div>
                  </div>
                  <div className="metric-compact-item">
                    <div className="metric-icon">
                      <i className="fas fa-check-circle"></i>
                    </div>
                    <div className="metric-content">
                      <div className="metric-label">Val Accuracy</div>
                      <div
                        className={`metric-value ${
                          trainingProgress.valAccuracy > 0 ? "good" : "na"
                        }`}
                      >
                        {trainingProgress.valAccuracy > 0
                          ? (trainingProgress.valAccuracy * 100).toFixed(2) +
                            "%"
                          : "—"}
                      </div>
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        {/* Weights & Biases Button - Compact Version */}
        {wandbUrl && (
          <div className="wandb-controls-compact">
            <a
              href={wandbUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="wandb-button-compact"
            >
              <div className="wandb-icon-compact">
                <i className="fas fa-chart-line"></i>
              </div>
              <span className="wandb-title-compact">W&B Dashboard</span>
              <div className="wandb-arrow-compact">
                <i className="fas fa-external-link-alt"></i>
              </div>
            </a>
          </div>
        )}
      </div>
    );
  };

  // Add a useEffect hook to log when visualization changes
  useEffect(() => {
    console.log("Selected visualization changed to:", selectedVisualization);

    // If California Housing is selected, check available visualization data
    if (selectedDataset === "California Housing") {
      console.log("California Housing dataset - Visualization state:", {
        selectedVisualization,
        heatmapAvailable: heatmapImage !== null,
        residualPlotAvailable:
          residuals.length > 0 && predictedValues.length > 0,
        rmseValue: rmse,
        r2Value: r2,
      });

      // Force selection of appropriate visualization if data is available but not showing
      if (
        heatmapImage &&
        selectedVisualization !== "heatmap" &&
        (!residuals.length || !predictedValues.length)
      ) {
        console.log(
          "Heatmap data available but not showing - switching to heatmap visualization"
        );
        setSelectedVisualization("heatmap");
      } else if (
        residuals.length > 0 &&
        predictedValues.length > 0 &&
        !heatmapImage &&
        selectedVisualization !== "residual_plot"
      ) {
        console.log(
          "Residual plot data available but not showing - switching to residual plot visualization"
        );
        setSelectedVisualization("residual_plot");
      }
    }

    // Return void or cleanup function
    return;
  }, [
    selectedVisualization,
    selectedDataset,
    heatmapImage,
    residuals,
    predictedValues,
    rmse,
    r2,
  ]);

  // Add this after the other useEffect hooks
  useEffect(() => {
    console.log("trainingConfig changed:", trainingConfig);
  }, [trainingConfig]);

  // Listen for getTrainingConfig event from NavBar
  useEffect(() => {
    const getTrainingConfigHandler = () => {
      localStorage.setItem("trainingConfig", JSON.stringify(trainingConfig));
      console.log("Saved trainingConfig to localStorage:", trainingConfig);
    };
    window.addEventListener(
      "getTrainingConfig",
      getTrainingConfigHandler as EventListener
    );
    return () => {
      window.removeEventListener(
        "getTrainingConfig",
        getTrainingConfigHandler as EventListener
      );
    };
  }, [trainingConfig]);

  // Add state for Save as Template modal and template name
  const [showSaveTemplateModal, setShowSaveTemplateModal] = useState(false);
  const [templateName, setTemplateName] = useState("");
  void templateName;
  void setShowSaveTemplateModal;
  void setTemplateName;
  // Load session templates on mount
  const [sessionTemplates, setSessionTemplates] = useState<any[]>([]);

  // Load session templates on mount
  useEffect(() => {
    const stored = sessionStorage.getItem("sessionTemplates");
    if (stored) {
      setSessionTemplates(JSON.parse(stored));
    }
  }, []);

  // Save session templates to sessionStorage whenever they change
  useEffect(() => {
    sessionStorage.setItem(
      "sessionTemplates",
      JSON.stringify(sessionTemplates)
    );
  }, [sessionTemplates]);

  // Handler to save as template (now saves to sessionStorage and adds to list)
  const handleSaveAsTemplate = () => {
    const name = prompt("Enter a name for your template:");
    if (!name || !name.trim()) {
      showToast("Please enter a valid template name.", "error");
      return;
    }
    const template = { name: name.trim(), nodes, edges };
    setSessionTemplates((prev) => [...prev, template]);
    showToast("Template saved!", "success");
  };

  // Handler to load a session template
  const loadSessionTemplate = (template: any) => {
    if (template.nodes && template.edges) {
      setNodes(template.nodes);
      setEdges(template.edges);
      showToast("Template loaded!", "success");
    } else {
      showToast("Invalid template format.", "error");
    }
  };

  const [activeRightSidebarTab, setActiveRightSidebarTab] =
    useState<RightSidebarTab>("dataset");

  const renderRightSidebarContent = () => {
    switch (activeRightSidebarTab) {
      case "dataset":
        return (
          <>
            {/* Dataset Selection Dropdown */}
            <div className="dataset-section">
              <div className="dataset-header">
                <i className="fas fa-database"></i>
                <h3>Choose Dataset</h3>
              </div>
              <div className="dataset-dropdown-container">
                <select
                  value={selectedDataset}
                  onChange={(e) => {
                    const newDataset = e.target.value;
                    if (newDataset === "__create_custom__") {
                      setShowCustomDatasetModal(true);
                      return;
                    }
                    setSelectedDataset(newDataset);
                    // Manually trigger a dataset change event
                    const event = new CustomEvent("datasetChange", {
                      detail: { dataset: newDataset },
                    });
                    window.dispatchEvent(event);
                  }}
                  className="dataset-select-dropdown"
                >
                  <option value="">Select Dataset</option>
                  <option value="MNIST">MNIST</option>
                  <option value="CIFAR-10">CIFAR-10</option>
                  <option value="Iris">Iris</option>
                  <option value="Breast Cancer">Breast Cancer</option>
                  <option value="California Housing">California Housing</option>
                  {customDatasets.length > 0 && (
                    <>
                      <option disabled>──────────</option>
                      {customDatasets.map((datasetName) => (
                        <option key={datasetName} value={datasetName}>
                          {datasetName} (Custom)
                        </option>
                      ))}
                    </>
                  )}
                  <option disabled>──────────</option>
                  <option value="__create_custom__">
                    ➕ Add Custom Dataset
                  </option>
                </select>
              </div>
            </div>

            {/* Hyperparameters Section */}
            <div className="hyperparams-modern-card">
              <div className="hyperparams-modern-header">
                <i className="fas fa-sliders-h"></i>
                <span>Hyperparameters</span>
              </div>
              <div className="hyperparams-modern-content">
                {/* Batch Size */}
                <div className="modern-param-row">
                  <span>Batch Size</span>
                  <div className="modern-number-input">
                    <button
                      onClick={() =>
                        setTrainingConfig({
                          ...trainingConfig,
                          batchSize: Math.max(1, trainingConfig.batchSize - 1),
                        })
                      }
                    >
                      -
                    </button>
                    <span>{trainingConfig.batchSize}</span>
                    <button
                      onClick={() =>
                        setTrainingConfig({
                          ...trainingConfig,
                          batchSize: trainingConfig.batchSize + 1,
                        })
                      }
                    >
                      +
                    </button>
                  </div>
                </div>
                {/* Epochs */}
                <div className="modern-param-row">
                  <span>Epochs</span>
                  <div className="modern-number-input">
                    <button
                      onClick={() =>
                        setTrainingConfig({
                          ...trainingConfig,
                          epochs: Math.max(1, trainingConfig.epochs - 1),
                        })
                      }
                    >
                      -
                    </button>
                    <span>{trainingConfig.epochs}</span>
                    <button
                      onClick={() =>
                        setTrainingConfig({
                          ...trainingConfig,
                          epochs: trainingConfig.epochs + 1,
                        })
                      }
                    >
                      +
                    </button>
                  </div>
                </div>
                {/* Optimizer */}
                <div className="modern-param-row">
                  <span>Optimizer</span>
                  <select
                    value={trainingConfig.optimizer}
                    onChange={(e) =>
                      setTrainingConfig({
                        ...trainingConfig,
                        optimizer: e.target.value,
                      })
                    }
                  >
                    <option value="Adam">Adam</option>
                    <option value="SGD">SGD</option>
                    <option value="RMSprop">RMSprop</option>
                    <option value="Adagrad">Adagrad</option>
                    <option value="Adadelta">Adadelta</option>
                  </select>
                </div>
                {/* Loss Function */}
                <div className="modern-param-row">
                  <span>Loss Function</span>
                  <select
                    value={trainingConfig.lossFunction}
                    onChange={(e) =>
                      setTrainingConfig({
                        ...trainingConfig,
                        lossFunction: e.target.value,
                      })
                    }
                  >
                    <option value="Categorical Cross-Entropy">
                      Categorical Cross-Entropy
                    </option>
                    <option value="Binary Cross-Entropy">
                      Binary Cross-Entropy
                    </option>
                    <option value="Mean Squared Error">
                      Mean Squared Error
                    </option>
                    <option value="Mean Absolute Error">
                      Mean Absolute Error
                    </option>
                    <option value="Huber Loss">Huber Loss</option>
                  </select>
                </div>
                {/* Validation Split */}
                <div className="modern-param-row">
                  <span>Validation Split</span>
                  <div className="modern-slider-group">
                    <input
                      type="range"
                      min="0.1"
                      max="0.5"
                      step="0.01"
                      value={trainingConfig.validationSplit}
                      onChange={(e) =>
                        setTrainingConfig({
                          ...trainingConfig,
                          validationSplit: parseFloat(e.target.value),
                        })
                      }
                    />
                    <span className="modern-slider-value">
                      {trainingConfig.validationSplit}
                    </span>
                  </div>
                </div>
                {/* Learning Rate */}
                <div className="modern-param-row">
                  <span>Learning Rate</span>
                  <div className="modern-slider-group">
                    <input
                      type="range"
                      min="0.0001"
                      max="0.1"
                      step="0.0001"
                      value={trainingConfig.learningRate}
                      onChange={(e) =>
                        setTrainingConfig({
                          ...trainingConfig,
                          learningRate: parseFloat(e.target.value),
                        })
                      }
                    />
                    <span className="modern-slider-value">
                      {trainingConfig.learningRate}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </>
        );
      case "parameters":
        return (
          <>
            {/* Layer Parameters Section - Only visible when a node is selected */}
            {selectedNode && (
              <div className="layer-params-container">
                <div className="layer-params-header">
                  <div className="header-content">
                    <i
                      className={`fas ${getLayerIcon(selectedNode.type || "")}`}
                    ></i>
                    <h3>
                      {selectedNode.type
                        ? selectedNode.type.charAt(0).toUpperCase() +
                          selectedNode.type.slice(1)
                        : ""}
                    </h3>
                  </div>
                  <button
                    className="delete-layer-btn"
                    onClick={() => deleteNode(selectedNode.id)}
                    title="Delete Layer"
                  >
                    <i className="fas fa-trash"></i>
                  </button>
                </div>
                <div className="layer-params-content">
                  {/* Layer Name Input - Only for non-custom blocks */}
                  {selectedNode.type !== "customblock" && (
                    <div className="param-item">
                      <input
                        type="text"
                        value={selectedNode.data.label || ""}
                        onChange={(e) =>
                          updateParameter("label", e.target.value)
                        }
                        placeholder="Layer Name"
                        className="layer-name-input"
                      />
                    </div>
                  )}

                  {/* Layer Size Information */}
                  {layerSizes[selectedNode.id] && (
                    <div className="layer-size-info">
                      <div className="size-section">
                        <div className="size-header">
                          <i className="fas fa-arrow-right"></i>
                          <span>Input Size</span>
                        </div>
                        <div className="size-display">
                          <span
                            className={`size-value ${
                              layerSizes[selectedNode.id].inputSize ===
                              "Disconnected"
                                ? "disconnected"
                                : ""
                            }`}
                          >
                            {layerSizes[selectedNode.id].inputSize}
                          </span>
                        </div>
                      </div>
                      <div className="size-section">
                        <div className="size-header">
                          <i className="fas fa-arrow-right"></i>
                          <span>Output Size</span>
                        </div>
                        <div className="size-display">
                          <span className="size-value">
                            {layerSizes[selectedNode.id].outputSize}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Dense Layer Parameters */}
                  {selectedNode.type === "dense" && (
                    <div className="param-item">
                      <label>Neurons</label>
                      <div className="number-control">
                        <button
                          className="control-btn"
                          onClick={() =>
                            updateParameter(
                              "neurons",
                              Math.max(1, (selectedNode.data.neurons || 1) - 1)
                            )
                          }
                        >
                          <i className="fas fa-minus"></i>
                        </button>
                        <input
                          type="number"
                          min="1"
                          value={selectedNode.data.neurons || ""}
                          onChange={(e) =>
                            updateParameter("neurons", +e.target.value)
                          }
                          className="number-input"
                        />
                        <button
                          className="control-btn"
                          onClick={() =>
                            updateParameter(
                              "neurons",
                              (selectedNode.data.neurons || 0) + 1
                            )
                          }
                        >
                          <i className="fas fa-plus"></i>
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Convolution Layer Parameters */}
                  {selectedNode.type === "convolution" && (
                    <>
                      <div className="param-item">
                        <label>Filters</label>
                        <div className="number-control">
                          <button
                            className="control-btn"
                            onClick={() =>
                              updateParameter(
                                "filters",
                                Math.max(
                                  1,
                                  (selectedNode.data.filters || 32) - 1
                                )
                              )
                            }
                          >
                            <i className="fas fa-minus"></i>
                          </button>
                          <input
                            type="number"
                            min="1"
                            value={selectedNode.data.filters || "32"}
                            onChange={(e) =>
                              updateParameter("filters", +e.target.value)
                            }
                            className="number-input"
                          />
                          <button
                            className="control-btn"
                            onClick={() =>
                              updateParameter(
                                "filters",
                                (selectedNode.data.filters || 32) + 1
                              )
                            }
                          >
                            <i className="fas fa-plus"></i>
                          </button>
                        </div>
                      </div>

                      <div className="param-item">
                        <label>Kernel Size</label>
                        <div className="dimension-control">
                          <div className="number-control">
                            <button
                              className="control-btn"
                              onClick={() =>
                                updateParameter("kernelSize", [
                                  (selectedNode.data.kernelSize?.[0] || 3) - 1,
                                  selectedNode.data.kernelSize?.[1] || 3,
                                ])
                              }
                            >
                              <i className="fas fa-minus"></i>
                            </button>
                            <input
                              type="number"
                              min="1"
                              value={selectedNode.data.kernelSize?.[0] || "3"}
                              onChange={(e) =>
                                updateParameter("kernelSize", [
                                  +e.target.value,
                                  selectedNode.data.kernelSize?.[1] || 3,
                                ])
                              }
                              className="number-input"
                            />
                            <button
                              className="control-btn"
                              onClick={() =>
                                updateParameter("kernelSize", [
                                  (selectedNode.data.kernelSize?.[0] || 3) + 1,
                                  selectedNode.data.kernelSize?.[1] || 3,
                                ])
                              }
                            >
                              <i className="fas fa-plus"></i>
                            </button>
                          </div>
                          <span>×</span>
                          <div className="number-control">
                            <button
                              className="control-btn"
                              onClick={() =>
                                updateParameter("kernelSize", [
                                  selectedNode.data.kernelSize?.[0] || 3,
                                  (selectedNode.data.kernelSize?.[1] || 3) - 1,
                                ])
                              }
                            >
                              <i className="fas fa-minus"></i>
                            </button>
                            <input
                              type="number"
                              min="1"
                              value={selectedNode.data.kernelSize?.[1] || "3"}
                              onChange={(e) =>
                                updateParameter("kernelSize", [
                                  selectedNode.data.kernelSize?.[0] || 3,
                                  +e.target.value,
                                ])
                              }
                              className="number-input"
                            />
                            <button
                              className="control-btn"
                              onClick={() =>
                                updateParameter("kernelSize", [
                                  selectedNode.data.kernelSize?.[0] || 3,
                                  (selectedNode.data.kernelSize?.[1] || 3) + 1,
                                ])
                              }
                            >
                              <i className="fas fa-plus"></i>
                            </button>
                          </div>
                        </div>
                      </div>

                      <div className="param-item">
                        <label>Stride</label>
                        <div className="dimension-control">
                          <div className="number-control">
                            <button
                              className="control-btn"
                              onClick={() =>
                                updateParameter("stride", [
                                  (selectedNode.data.stride?.[0] || 1) - 1,
                                  selectedNode.data.stride?.[1] || 1,
                                ])
                              }
                            >
                              <i className="fas fa-minus"></i>
                            </button>
                            <input
                              type="number"
                              min="1"
                              value={selectedNode.data.stride?.[0] || "1"}
                              onChange={(e) =>
                                updateParameter("stride", [
                                  +e.target.value,
                                  selectedNode.data.stride?.[1] || 1,
                                ])
                              }
                              className="number-input"
                            />
                            <button
                              className="control-btn"
                              onClick={() =>
                                updateParameter("stride", [
                                  (selectedNode.data.stride?.[0] || 1) + 1,
                                  selectedNode.data.stride?.[1] || 1,
                                ])
                              }
                            >
                              <i className="fas fa-plus"></i>
                            </button>
                          </div>
                          <span>×</span>
                          <div className="number-control">
                            <button
                              className="control-btn"
                              onClick={() =>
                                updateParameter("stride", [
                                  selectedNode.data.stride?.[0] || 1,
                                  (selectedNode.data.stride?.[1] || 1) - 1,
                                ])
                              }
                            >
                              <i className="fas fa-minus"></i>
                            </button>
                            <input
                              type="number"
                              min="1"
                              value={selectedNode.data.stride?.[1] || "1"}
                              onChange={(e) =>
                                updateParameter("stride", [
                                  selectedNode.data.stride?.[0] || 1,
                                  +e.target.value,
                                ])
                              }
                              className="number-input"
                            />
                            <button
                              className="control-btn"
                              onClick={() =>
                                updateParameter("stride", [
                                  selectedNode.data.stride?.[0] || 1,
                                  (selectedNode.data.stride?.[1] || 1) + 1,
                                ])
                              }
                            >
                              <i className="fas fa-plus"></i>
                            </button>
                          </div>
                        </div>
                      </div>
                    </>
                  )}

                  {/* BatchNormalization Layer Parameters */}
                  {selectedNode.type === "batchnormalization" && (
                    <>
                      <div className="param-item">
                        <label>Momentum</label>
                        <div className="number-control">
                          <button
                            className="control-btn"
                            onClick={() =>
                              updateParameter(
                                "momentum",
                                Math.max(
                                  0,
                                  (selectedNode.data.momentum || 0.99) - 0.01
                                )
                              )
                            }
                          >
                            <i className="fas fa-minus"></i>
                          </button>
                          <input
                            type="number"
                            min="0"
                            max="1"
                            step="0.01"
                            value={selectedNode.data.momentum || "0.99"}
                            onChange={(e) =>
                              updateParameter("momentum", +e.target.value)
                            }
                            className="number-input"
                          />
                          <button
                            className="control-btn"
                            onClick={() =>
                              updateParameter(
                                "momentum",
                                Math.min(
                                  1,
                                  (selectedNode.data.momentum || 0.99) + 0.01
                                )
                              )
                            }
                          >
                            <i className="fas fa-plus"></i>
                          </button>
                        </div>
                      </div>

                      <div className="param-item">
                        <label>Epsilon</label>
                        <div className="number-control">
                          <button
                            className="control-btn"
                            onClick={() =>
                              updateParameter(
                                "epsilon",
                                Math.max(
                                  0.00001,
                                  (selectedNode.data.epsilon || 0.001) - 0.0001
                                )
                              )
                            }
                          >
                            <i className="fas fa-minus"></i>
                          </button>
                          <input
                            type="number"
                            min="0.00001"
                            step="0.00001"
                            value={selectedNode.data.epsilon || "0.001"}
                            onChange={(e) =>
                              updateParameter("epsilon", +e.target.value)
                            }
                            className="number-input"
                          />
                          <button
                            className="control-btn"
                            onClick={() =>
                              updateParameter(
                                "epsilon",
                                (selectedNode.data.epsilon || 0.001) + 0.0001
                              )
                            }
                          >
                            <i className="fas fa-plus"></i>
                          </button>
                        </div>
                      </div>
                    </>
                  )}
                  {/* MaxPooling Layer Parameters */}
                  {selectedNode.type === "maxpooling" && (
                    <>
                      <div className="param-item">
                        <label>Pool Size</label>
                        <div className="dimension-control">
                          <div className="number-control">
                            <button
                              className="control-btn"
                              onClick={() =>
                                updateParameter("poolSize", [
                                  (selectedNode.data.poolSize?.[0] || 2) - 1,
                                  selectedNode.data.poolSize?.[1] || 2,
                                ])
                              }
                            >
                              <i className="fas fa-minus"></i>
                            </button>
                            <input
                              type="number"
                              min="1"
                              value={selectedNode.data.poolSize?.[0] || "2"}
                              onChange={(e) =>
                                updateParameter("poolSize", [
                                  parseInt(e.target.value) || 2,
                                  selectedNode.data.poolSize?.[1] || 2,
                                ])
                              }
                              className="number-input"
                            />
                            <button
                              className="control-btn"
                              onClick={() =>
                                updateParameter("poolSize", [
                                  (selectedNode.data.poolSize?.[0] || 2) + 1,
                                  selectedNode.data.poolSize?.[1] || 2,
                                ])
                              }
                            >
                              <i className="fas fa-plus"></i>
                            </button>
                          </div>
                          <span>×</span>
                          <div className="number-control">
                            <button
                              className="control-btn"
                              onClick={() =>
                                updateParameter("poolSize", [
                                  selectedNode.data.poolSize?.[0] || 2,
                                  (selectedNode.data.poolSize?.[1] || 2) - 1,
                                ])
                              }
                            >
                              <i className="fas fa-minus"></i>
                            </button>
                            <input
                              type="number"
                              min="1"
                              value={selectedNode.data.poolSize?.[1] || "2"}
                              onChange={(e) =>
                                updateParameter("poolSize", [
                                  selectedNode.data.poolSize?.[0] || 2,
                                  parseInt(e.target.value) || 2,
                                ])
                              }
                              className="number-input"
                            />
                            <button
                              className="control-btn"
                              onClick={() =>
                                updateParameter("poolSize", [
                                  selectedNode.data.poolSize?.[0] || 2,
                                  (selectedNode.data.poolSize?.[1] || 2) + 1,
                                ])
                              }
                            >
                              <i className="fas fa-plus"></i>
                            </button>
                          </div>
                        </div>
                      </div>

                      <div className="param-item">
                        <label>Stride</label>
                        <div className="dimension-control">
                          <div className="number-control">
                            <button
                              className="control-btn"
                              onClick={() =>
                                updateParameter("stride", [
                                  (selectedNode.data.stride?.[0] || 2) - 1,
                                  selectedNode.data.stride?.[1] || 2,
                                ])
                              }
                            >
                              <i className="fas fa-minus"></i>
                            </button>
                            <input
                              type="number"
                              min="1"
                              value={selectedNode.data.stride?.[0] || "2"}
                              onChange={(e) =>
                                updateParameter("stride", [
                                  parseInt(e.target.value) || 2,
                                  selectedNode.data.stride?.[1] || 2,
                                ])
                              }
                              className="number-input"
                            />
                            <button
                              className="control-btn"
                              onClick={() =>
                                updateParameter("stride", [
                                  (selectedNode.data.stride?.[0] || 2) + 1,
                                  selectedNode.data.stride?.[1] || 2,
                                ])
                              }
                            >
                              <i className="fas fa-plus"></i>
                            </button>
                          </div>
                          <span>×</span>
                          <div className="number-control">
                            <button
                              className="control-btn"
                              onClick={() =>
                                updateParameter("stride", [
                                  selectedNode.data.stride?.[0] || 2,
                                  (selectedNode.data.stride?.[1] || 2) - 1,
                                ])
                              }
                            >
                              <i className="fas fa-minus"></i>
                            </button>
                            <input
                              type="number"
                              min="1"
                              value={selectedNode.data.stride?.[1] || "2"}
                              onChange={(e) =>
                                updateParameter("stride", [
                                  selectedNode.data.stride?.[0] || 2,
                                  parseInt(e.target.value) || 2,
                                ])
                              }
                              className="number-input"
                            />
                            <button
                              className="control-btn"
                              onClick={() =>
                                updateParameter("stride", [
                                  selectedNode.data.stride?.[0] || 2,
                                  (selectedNode.data.stride?.[1] || 2) + 1,
                                ])
                              }
                            >
                              <i className="fas fa-plus"></i>
                            </button>
                          </div>
                        </div>
                      </div>

                      <div className="param-item">
                        <label>Padding</label>
                        <select
                          value={selectedNode.data.padding || "valid"}
                          onChange={(e) =>
                            updateParameter("padding", e.target.value)
                          }
                          className="param-select"
                        >
                          <option value="valid">Valid</option>
                          <option value="same">Same</option>
                        </select>
                      </div>
                    </>
                  )}

                  {/*  Layer Parameters */}
                  {selectedNode.type === "attention" && (
                    <>
                      <div className="param-item">
                        <label>Number of Heads</label>
                        <div className="number-control">
                          <button
                            className="control-btn"
                            onClick={() =>
                              updateParameter(
                                "heads",
                                Math.max(1, (selectedNode.data.heads || 8) - 1)
                              )
                            }
                          >
                            <i className="fas fa-minus"></i>
                          </button>
                          <input
                            type="number"
                            min="1"
                            value={selectedNode.data.heads || "8"}
                            onChange={(e) =>
                              updateParameter("heads", +e.target.value)
                            }
                            className="number-input"
                          />
                          <button
                            className="control-btn"
                            onClick={() =>
                              updateParameter(
                                "heads",
                                (selectedNode.data.heads || 8) + 1
                              )
                            }
                          >
                            <i className="fas fa-plus"></i>
                          </button>
                        </div>
                      </div>

                      <div className="param-item">
                        <label>Key Dimension</label>
                        <div className="number-control">
                          <button
                            className="control-btn"
                            onClick={() =>
                              updateParameter(
                                "keyDim",
                                Math.max(
                                  1,
                                  (selectedNode.data.keyDim || 64) - 1
                                )
                              )
                            }
                          >
                            <i className="fas fa-minus"></i>
                          </button>
                          <input
                            type="number"
                            min="1"
                            value={selectedNode.data.keyDim || "64"}
                            onChange={(e) =>
                              updateParameter("keyDim", +e.target.value)
                            }
                            className="number-input"
                          />
                          <button
                            className="control-btn"
                            onClick={() =>
                              updateParameter(
                                "keyDim",
                                (selectedNode.data.keyDim || 64) + 1
                              )
                            }
                          >
                            <i className="fas fa-plus"></i>
                          </button>
                        </div>
                      </div>

                      <div className="param-item">
                        <label>Dropout Rate</label>
                        <div className="number-control">
                          <button
                            className="control-btn"
                            onClick={() =>
                              updateParameter(
                                "attentionDropout",
                                Math.max(
                                  0,
                                  (selectedNode.data.attentionDropout || 0) -
                                    0.1
                                )
                              )
                            }
                          >
                            <i className="fas fa-minus"></i>
                          </button>
                          <input
                            type="number"
                            min="0"
                            max="1"
                            step="0.1"
                            value={selectedNode.data.attentionDropout || "0"}
                            onChange={(e) =>
                              updateParameter(
                                "attentionDropout",
                                +e.target.value
                              )
                            }
                            className="number-input"
                          />
                          <button
                            className="control-btn"
                            onClick={() =>
                              updateParameter(
                                "attentionDropout",
                                Math.min(
                                  1,
                                  (selectedNode.data.attentionDropout || 0) +
                                    0.1
                                )
                              )
                            }
                          >
                            <i className="fas fa-plus"></i>
                          </button>
                        </div>
                      </div>
                    </>
                  )}
                  {/* Dropout Layer Parameters */}
                  {selectedNode.type === "dropout" && (
                    <div className="param-item">
                      <label>Rate</label>
                      <div className="slider-with-value">
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.01"
                          value={selectedNode.data.rate || 0.2}
                          onChange={(e) =>
                            updateParameter(
                              "rate",
                              parseFloat(e.target.value) || 0
                            )
                          }
                          className="param-slider"
                        />
                        <input
                          type="number"
                          min="0"
                          max="1"
                          step="0.01"
                          value={selectedNode.data.rate || 0.2}
                          onChange={(e) =>
                            updateParameter(
                              "rate",
                              parseFloat(e.target.value) || 0
                            )
                          }
                          className="param-input small"
                        />
                      </div>
                    </div>
                  )}

                  {/* ResNetBlock Layer Parameters */}
                  {selectedNode.type === "resnetblock" && (
                    <>
                      <div className="param-item">
                        <label>Block Type</label>
                        <select
                          value={selectedNode.data.blockType || "Basic"}
                          onChange={(e) =>
                            updateParameter("blockType", e.target.value)
                          }
                          className="param-select"
                        >
                          <option value="Basic">Basic Block</option>
                          <option value="Bottleneck">Bottleneck Block</option>
                        </select>
                      </div>

                      <div className="param-item">
                        <label>Filters</label>
                        <div className="number-control">
                          <button
                            className="control-btn"
                            onClick={() =>
                              updateParameter(
                                "filters",
                                Math.max(
                                  1,
                                  (selectedNode.data.filters || 64) - 1
                                )
                              )
                            }
                          >
                            <i className="fas fa-minus"></i>
                          </button>
                          <input
                            type="number"
                            min="1"
                            value={selectedNode.data.filters || "64"}
                            onChange={(e) =>
                              updateParameter("filters", +e.target.value)
                            }
                            className="number-input"
                          />
                          <button
                            className="control-btn"
                            onClick={() =>
                              updateParameter(
                                "filters",
                                (selectedNode.data.filters || 64) + 1
                              )
                            }
                          >
                            <i className="fas fa-plus"></i>
                          </button>
                        </div>
                      </div>

                      <div className="param-item">
                        <label>Stride</label>
                        <select
                          value={
                            Array.isArray(selectedNode.data.stride)
                              ? selectedNode.data.stride[0]
                              : 1
                          }
                          onChange={(e) =>
                            updateParameter("stride", [
                              parseInt(e.target.value),
                              parseInt(e.target.value),
                            ])
                          }
                          className="param-select"
                        >
                          <option value="1">1x1</option>
                          <option value="2">2x2</option>
                        </select>
                      </div>

                      <div className="param-item">
                        <label>Activation</label>
                        <select
                          value={selectedNode.data.activation || "relu"}
                          onChange={(e) =>
                            updateParameter("activation", e.target.value)
                          }
                          className="param-select"
                        >
                          <option value="relu">ReLU</option>
                          <option value="leaky_relu">Leaky ReLU</option>
                        </select>
                      </div>
                    </>
                  )}

                  {/* Custom Block Parameters */}
                  {selectedNode.type === "customblock" &&
                    renderCustomBlockParameters()}

                  {/* No parameters message for layers without parameters */}
                  {(selectedNode.type === "flatten" ||
                    selectedNode.type === "globalaveragepool" ||
                    selectedNode.type === "addlayer") && (
                    <div className="no-params-message">
                      <p>This layer has no configurable parameters.</p>
                    </div>
                  )}
                </div>
              </div>
            )}
            {/* Display a message when no node is selected */}
            {!selectedNode && (
              <div className="no-layer-selected">
                <i className="fas fa-mouse-pointer"></i>
                <p>Select a layer to edit parameters</p>
              </div>
            )}
          </>
        );
      case "train":
        return (
          <div className="train-metrics-tab-content">
            {/* Visualization Controls */}
            <div className="visualization-controls">
              <label htmlFor="visualization-select">
                Select Visualization:
              </label>
              <div className="custom-select-container">
                <select
                  id="visualization-select"
                  value={selectedVisualization}
                  onChange={(e) => {
                    visualizationUserSelected.current = true;
                    setSelectedVisualization(e.target.value);
                  }}
                  className="visualization-select"
                >
                  {getVisualizationOptions().map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div className="visualization-content">{renderVisualization()}</div>
            {/* Training Metrics Below Visualization */}
            {renderTrainingMetrics()}
            {/* Validation component at the top */}
            <div className="validation-section">
              <Validation
                datasetName={selectedDataset}
                isTrainingComplete={trainingComplete}
              />
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  useEffect(() => {
    const handleTrainModel = () => {
      setActiveRightSidebarTab("train");
    };
    window.addEventListener("trainModel", handleTrainModel);
    return () => window.removeEventListener("trainModel", handleTrainModel);
  }, []);

  // Function to delete a node
  const deleteNode = (nodeId: string): void => {
    // Remove all edges connected to this node
    setEdges((eds) =>
      eds.filter((edge) => edge.source !== nodeId && edge.target !== nodeId)
    );

    // Remove the node
    setNodes((nds) => nds.filter((node) => node.id !== nodeId));

    // Clear selected node if it was the deleted node
    if (selectedNode?.id === nodeId) {
      setSelectedNode(null);
    }
  };

  // Toast notification state
  const [toast, setToast] = useState<{
    message: string;
    type?: ToastType;
  } | null>(null);

  // Helper to show toast
  const showToast = (message: string, type: ToastType = "info") => {
    setToast({ message, type });
  };

  const handleDeleteEdge = (edgeId: string) => {
    setEdges((eds) => eds.filter((e) => e.id !== edgeId));
  };

  // Function to validate input/output sizes throughout the network and populate layerSizes for UI
  const validateInputOutputSizes = useCallback((): ValidationErrors => {
    const errors: ValidationErrors = [];

    // Check if dataset is selected
    if (!selectedDataset || nodes.length === 0) {
      setLayerSizes({});
      if (!selectedDataset) {
        errors.push(
          "[Critical] Please select a dataset before validating sizes"
        );
      }
      return errors;
    }

    // Get dataset specifications
    const getDatasetInputShape = (dataset: string): number[] => {
      switch (dataset) {
        case "MNIST":
          return [28, 28, 1]; // (height, width, channels)
        case "CIFAR-10":
          return [32, 32, 3]; // (height, width, channels)
        case "Iris":
          return [4]; // 4 features
        case "Breast Cancer":
          return [30]; // 30 features
        case "California Housing":
          return [8]; // 8 features
        default:
          return [1]; // Default fallback
      }
    };

    const getDatasetOutputSize = (dataset: string): number => {
      switch (dataset) {
        case "MNIST":
        case "CIFAR-10":
          return 10; // 10 classes
        case "Iris":
          return 3; // 3 classes
        case "Breast Cancer":
          return 1; // Binary classification
        case "California Housing":
          return 1; // Regression
        default:
          return 1; // Default fallback
      }
    };

    const expectedInputShape = getDatasetInputShape(selectedDataset);
    const expectedOutputSize = getDatasetOutputSize(selectedDataset);

    // Find input and output layers
    const inputLayer = nodes.find((node) => node.type === "input");
    const outputLayer = nodes.find((node) => node.type === "output");

    if (!inputLayer) {
      errors.push("[Critical] Input layer is required for size validation");
      return errors;
    }

    if (!outputLayer) {
      errors.push("[Critical] Output layer is required for size validation");
      return errors;
    }

    // Build graph for topological traversal
    const adjacencyList = new Map<string, string[]>();
    const incomingEdges = new Map<string, string[]>();

    edges.forEach((edge) => {
      if (!adjacencyList.has(edge.source)) {
        adjacencyList.set(edge.source, []);
      }
      adjacencyList.get(edge.source)!.push(edge.target);

      if (!incomingEdges.has(edge.target)) {
        incomingEdges.set(edge.target, []);
      }
      incomingEdges.get(edge.target)!.push(edge.source);
    });

    // Track tensor shapes through the network
    const tensorShapes = new Map<string, number[]>();
    const newLayerSizes: Record<
      string,
      { inputSize: string; outputSize: string }
    > = {};

    // Initialize input layer shape
    tensorShapes.set(inputLayer.id, expectedInputShape);

    // Format the shapes for display
    const formatShape = (shape: number[]): string => {
      if (shape.length === 1) {
        return shape[0].toString();
      } else if (shape.length === 2) {
        return `(${shape[0]}, ${shape[1]})`;
      } else if (shape.length === 3) {
        return `(${shape[0]}, ${shape[1]}, ${shape[2]})`;
      } else {
        return `(${shape.join(", ")})`;
      }
    };

    // Add input layer to UI display
    newLayerSizes[inputLayer.id] = {
      inputSize: "N/A",
      outputSize: formatShape(expectedInputShape),
    };

    // Topological sort to process layers in correct order
    const visited = new Set<string>();
    const tempVisited = new Set<string>();
    const sortedNodes: Node[] = [];

    const topologicalSort = (nodeId: string): boolean => {
      if (tempVisited.has(nodeId)) {
        errors.push(
          `[Critical] Circular dependency detected at node ${nodeId}`
        );
        return false;
      }
      if (visited.has(nodeId)) return true;

      tempVisited.add(nodeId);
      const neighbors = adjacencyList.get(nodeId) || [];

      for (const neighbor of neighbors) {
        if (!topologicalSort(neighbor)) return false;
      }

      tempVisited.delete(nodeId);
      visited.add(nodeId);

      const node = nodes.find((n) => n.id === nodeId);
      if (node) sortedNodes.unshift(node);

      return true;
    };

    // Start topological sort from all nodes with no incoming edges
    // Sort node IDs to ensure consistent ordering regardless of visual arrangement
    const sortedNodeIds = nodes.map((n) => n.id).sort();

    for (const nodeId of sortedNodeIds) {
      if (
        !incomingEdges.has(nodeId) ||
        incomingEdges.get(nodeId)!.length === 0
      ) {
        if (!topologicalSort(nodeId)) return errors;
      }
    }

    // Process remaining nodes (in case of disconnected components)
    // Use the same stable ID-based ordering
    for (const nodeId of sortedNodeIds) {
      if (!visited.has(nodeId)) {
        if (!topologicalSort(nodeId)) return errors;
      }
    }

    // Validate shapes through the network
    for (const node of sortedNodes) {
      if (node.type === "input") continue;

      // Get input shapes from predecessor nodes
      const predecessors = incomingEdges.get(node.id) || [];

      if (predecessors.length === 0 && node.type !== "input") {
        errors.push(`[Critical] Layer ${node.id} has no input connections`);
        continue;
      }

      // For multiple inputs, use the first one's shape (could be enhanced for more complex merging)
      const inputShape =
        predecessors.length > 0
          ? tensorShapes.get(predecessors[0]) || expectedInputShape
          : expectedInputShape;

      let outputShape: number[] = [...inputShape];

      try {
        // Calculate output shape based on layer type
        switch (node.type) {
          case "dense":
            const neurons = parseInt(node.data?.neurons) || 1;
            if (neurons <= 0) {
              errors.push(
                `[Critical] Dense layer ${node.id}: Invalid number of neurons (${neurons})`
              );
              continue;
            }

            // Dense layers expect flattened input or 1D input
            if (inputShape.length > 1) {
              outputShape = [neurons];
              // Note: In practice, there should be a flatten layer before dense
              if (
                !predecessors.some((predId) => {
                  const predNode = nodes.find((n) => n.id === predId);
                  return (
                    predNode?.type === "flatten" ||
                    predNode?.type === "globalaveragepool"
                  );
                })
              ) {
                errors.push(
                  `Warning: Dense layer ${
                    node.id
                  } receives multi-dimensional input (${inputShape.join(
                    "×"
                  )}). Consider adding a Flatten layer before it.`
                );
              }
            } else {
              outputShape = [neurons];
            }
            break;

          case "convolution":
            const filters = parseInt(node.data?.filters) || 32;
            const kernelSize = node.data?.kernelSize || [3, 3];
            const stride = node.data?.stride || [1, 1];
            const padding = node.data?.padding || "same";

            if (inputShape.length < 2) {
              errors.push(
                `[Critical] Convolution layer ${node.id}: Requires at least 2D input, got ${inputShape.length}D`
              );
              continue;
            }

            if (inputShape.length === 2) {
              // Assume single channel for 2D input
              const [h, w] = inputShape;
              let newH, newW;

              if (padding === "same") {
                newH = Math.ceil(h / stride[0]);
                newW = Math.ceil(w / stride[1]);
              } else {
                newH = Math.floor((h - kernelSize[0]) / stride[0]) + 1;
                newW = Math.floor((w - kernelSize[1]) / stride[1]) + 1;
              }

              if (newH <= 0 || newW <= 0) {
                errors.push(
                  `[Critical] Convolution layer ${node.id}: Output dimensions would be invalid (${newH}×${newW})`
                );
                continue;
              }

              outputShape = [newH, newW, filters];
            } else if (inputShape.length >= 3) {
              const [h, w] = inputShape;
              let newH, newW;

              if (padding === "same") {
                newH = Math.ceil(h / stride[0]);
                newW = Math.ceil(w / stride[1]);
              } else {
                newH = Math.floor((h - kernelSize[0]) / stride[0]) + 1;
                newW = Math.floor((w - kernelSize[1]) / stride[1]) + 1;
              }

              if (newH <= 0 || newW <= 0) {
                errors.push(
                  `[Critical] Convolution layer ${node.id}: Output dimensions would be invalid (${newH}×${newW}×${filters})`
                );
                continue;
              }

              outputShape = [newH, newW, filters];
            }
            break;

          case "maxpooling":
            const poolSize = node.data?.poolSize || [2, 2];
            const poolStride = node.data?.stride || [2, 2];

            if (inputShape.length < 2) {
              errors.push(
                `[Critical] MaxPooling layer ${node.id}: Requires at least 2D input, got ${inputShape.length}D`
              );
              continue;
            }

            if (inputShape.length === 2) {
              const [h, w] = inputShape;
              const newH = Math.floor((h - poolSize[0]) / poolStride[0]) + 1;
              const newW = Math.floor((w - poolSize[1]) / poolStride[1]) + 1;

              if (newH <= 0 || newW <= 0) {
                errors.push(
                  `[Critical] MaxPooling layer ${node.id}: Output dimensions would be invalid (${newH}×${newW})`
                );
                continue;
              }

              outputShape = [newH, newW];
            } else if (inputShape.length >= 3) {
              const [h, w, c] = inputShape;
              const newH = Math.floor((h - poolSize[0]) / poolStride[0]) + 1;
              const newW = Math.floor((w - poolSize[1]) / poolStride[1]) + 1;

              if (newH <= 0 || newW <= 0) {
                errors.push(
                  `[Critical] MaxPooling layer ${node.id}: Output dimensions would be invalid (${newH}×${newW}×${c})`
                );
                continue;
              }

              outputShape = [newH, newW, c];
            }
            break;

          case "flatten":
            const flattenedSize = inputShape.reduce((a, b) => a * b, 1);
            outputShape = [flattenedSize];
            break;

          case "globalaveragepool":
            if (inputShape.length >= 3) {
              // Global average pooling reduces spatial dimensions to 1
              outputShape = [inputShape[inputShape.length - 1]]; // Keep only channel dimension
            } else {
              errors.push(
                `[Critical] GlobalAveragePooling layer ${node.id}: Requires at least 3D input (H×W×C), got ${inputShape.length}D`
              );
              continue;
            }
            break;

          case "dropout":
          case "batchnormalization":
          case "activation":
            // These layers don't change the shape
            outputShape = [...inputShape];
            break;

          case "output":
            // The output layer should produce the expected number of outputs for the dataset
            // The input can be any size - it's the responsibility of previous layers to provide correct input
            // Just set the output shape to the expected output size
            outputShape = [expectedOutputSize];
            break;

          case "resnetblock":
            // ResNet blocks typically preserve spatial dimensions
            const blockFilters = parseInt(node.data?.filters) || 64;
            const blockStride = node.data?.stride || [1, 1];

            if (inputShape.length >= 3) {
              const [h, w] = inputShape;
              const newH = Math.ceil(h / blockStride[0]);
              const newW = Math.ceil(w / blockStride[1]);
              outputShape = [newH, newW, blockFilters];
            } else {
              errors.push(
                `[Critical] ResNet block ${node.id}: Requires 3D input (H×W×C), got ${inputShape.length}D`
              );
              continue;
            }
            break;

          case "customblock":
            // For custom blocks, we'd need to analyze the internal layers
            // For now, assume they preserve the input shape
            outputShape = [...inputShape];
            errors.push(
              `Warning: Custom block ${node.id}: Size validation for custom blocks is not fully implemented. Please verify manually.`
            );
            break;

          default:
            // Unknown layer type - preserve shape
            outputShape = [...inputShape];
            errors.push(
              `Warning: Unknown layer type ${node.type} in ${node.id}: Assuming shape preservation.`
            );
            break;
        }

        // Store the calculated output shape
        tensorShapes.set(node.id, outputShape);

        // Add to UI layer sizes display
        newLayerSizes[node.id] = {
          inputSize: formatShape(inputShape),
          outputSize: formatShape(outputShape),
        };
      } catch (error) {
        errors.push(
          `[Critical] Error calculating output shape for layer ${node.id}: ${
            error instanceof Error ? error.message : String(error)
          }`
        );
      }
    }

    // Additional validations

    // Check for common size mismatch patterns
    const denseNodes = nodes.filter((node) => node.type === "dense");
    const convNodes = nodes.filter((node) => node.type === "convolution");

    if (denseNodes.length > 0 && convNodes.length > 0) {
      // Check if there's a flatten/global pool between conv and dense layers
      let hasProperTransition = false;

      for (const denseNode of denseNodes) {
        const densePreds = incomingEdges.get(denseNode.id) || [];
        for (const predId of densePreds) {
          const predNode = nodes.find((n) => n.id === predId);
          if (
            predNode &&
            (predNode.type === "flatten" ||
              predNode.type === "globalaveragepool")
          ) {
            hasProperTransition = true;
            break;
          }
        }
        if (hasProperTransition) break;
      }

      if (!hasProperTransition) {
        errors.push(
          "Warning: When using both convolutional and dense layers, include a Flatten or GlobalAveragePooling layer between them to properly transition from 2D/3D to 1D data."
        );
      }
    }

    // Dataset-specific validations
    if (selectedDataset === "MNIST" || selectedDataset === "CIFAR-10") {
      const hasConvOrResNet = nodes.some(
        (n) => n.type === "convolution" || n.type === "resnetblock"
      );
      if (!hasConvOrResNet) {
        errors.push(
          `Suggestion: ${selectedDataset} dataset (image data) typically benefits from convolutional layers or ResNet blocks.`
        );
      }
    } else if (
      ["Iris", "Breast Cancer", "California Housing"].includes(selectedDataset)
    ) {
      const hasConv = nodes.some((n) => n.type === "convolution");
      if (hasConv) {
        errors.push(
          `Warning: ${selectedDataset} dataset (tabular data) typically doesn't require convolutional layers.`
        );
      }
    }

    // Update the layer sizes state for UI display
    setLayerSizes(newLayerSizes);

    return errors;
  }, [selectedDataset, nodes, edges]);

  // Calculate layer sizes when nodes, edges, or dataset changes
  useEffect(() => {
    validateInputOutputSizes();
  }, [validateInputOutputSizes]);

  return (
    <div className="new-build-page">
      {toast && (
        <ToastNotification
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
      <div className="new-build-container">
        {/* Validation Error Messages - Removed, using toast notifications only */}

        {/* Custom Layer Modal */}
        <CustomLayerModal
          isOpen={showCustomLayerModal}
          onClose={() => setShowCustomLayerModal(false)}
          onSave={handleCustomLayerSave}
        />

        <div className="left-panel">
          <div className="sidebar-nav">
            <div
              className={`sidebar-nav-item ${
                activeSidebarOption === "layers" ? "active" : ""
              }`}
              onClick={() => setActiveSidebarOption("layers")}
            >
              <i className="fas fa-layer-group"></i>
              <span>Layers</span>
            </div>
            <div
              className={`sidebar-nav-item ${
                activeSidebarOption === "templates" ? "active" : ""
              }`}
              onClick={() => setActiveSidebarOption("templates")}
            >
              <i className="fas fa-file-code"></i>
              <span>Templates</span>
            </div>
            <div
              className={`sidebar-nav-item ${
                activeSidebarOption === "activations" ? "active" : ""
              }`}
              onClick={() => setActiveSidebarOption("activations")}
            >
              <i className="fas fa-bolt"></i>
              <span>Activations</span>
            </div>
          </div>
          <div className="sidebar-content">{renderSidebarContent()}</div>
        </div>

        <div className="center-panel">
          <div className="canvas-container">
            <ReactFlow
              nodes={nodes}
              edges={edges.map((e) => ({
                ...e,
                type: "custom",
                data: { onDelete: handleDeleteEdge },
              }))}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onNodeClick={onNodeClick}
              nodeTypes={nodeTypes}
              edgeTypes={edgeTypes}
              isValidConnection={isValidConnection}
              connectionLineStyle={{ stroke: "#555", strokeWidth: 2 }}
              defaultEdgeOptions={{
                animated: true,
                style: { stroke: "#555", strokeWidth: 2 },
              }}
              fitView
              style={{ width: "100%", height: "100%" }}
            >
              <Background />
              <Controls />
            </ReactFlow>
          </div>
        </div>

        {/* Right Panel - Conditionally shows either:
         * Dataset Selection + Comprehensive Layer Parameters (before training)
         * Visualization Section (during and after training)
         */}
        <div className="right-panel">
          <div className="right-sidebar-controls">
            <div className="right-sidebar-tabs">
              <button
                className={`tab-button ${
                  activeRightSidebarTab === "dataset" ? "active" : ""
                }`}
                onClick={() => setActiveRightSidebarTab("dataset")}
              >
                <i className="fas fa-database"></i>
                Dataset
              </button>
              <button
                className={`tab-button ${
                  activeRightSidebarTab === "parameters" ? "active" : ""
                }`}
                onClick={() => setActiveRightSidebarTab("parameters")}
              >
                <i className="fas fa-sliders-h"></i>
                Parameters
              </button>
              <button
                className={`tab-button ${
                  activeRightSidebarTab === "train" ? "active" : ""
                }`}
                onClick={() => setActiveRightSidebarTab("train")}
              >
                <i className="fas fa-play-circle"></i>
                Train
              </button>
            </div>
            {renderRightSidebarContent()}
          </div>
        </div>
      </div>

      {/* Custom Dataset Modal */}
      <CustomDatasetModal
        isOpen={showCustomDatasetModal}
        onClose={() => setShowCustomDatasetModal(false)}
        onDatasetCreated={(datasetName) => {
          setSelectedDataset(datasetName);
          setShowCustomDatasetModal(false);
          refreshCustomDatasets();
        }}
      />

      {showSaveTemplateModal && false /* Remove modal rendering */}
    </div>
  );
};
export default NewBuildPage;
