import React, { useState } from "react";
import "../styles/components/TutorialPage.scss";
import { NavLink } from "react-router-dom";

// Define the tutorial sections
interface TutorialSection {
  id: string;
  title: string;
  content: React.ReactNode;
}

const TutorialPage: React.FC = () => {
  const [activeSection, setActiveSection] = useState<string>("introduction");

  // Define all tutorial sections
  const tutorialSections: TutorialSection[] = [
    {
      id: "introduction",
      title: "Introduction",
      content: (
        <div className="tutorial-content">
          <h2>Welcome to NeuroBlock</h2>
          <p>
            NeuroBlock is a drag-and-drop interface for building, training, and
            deploying neural networks without writing code. This tutorial will
            guide you through the various features and components of the
            application.
          </p>
          <p>With NeuroBlock, you can:</p>
          <ul>
            <li>
              Build neural network architectures by dragging and dropping layers
            </li>
            <li>Configure layer parameters with an intuitive interface</li>
            <li>Train your models on various datasets</li>
            <li>Visualize training progress and model performance</li>
            <li>Export your trained models in various formats</li>
          </ul>
          <p>
            Select a topic from the sidebar to learn more about specific
            features.
          </p>
        </div>
      ),
    },
    {
      id: "interface",
      title: "Interface Overview",
      content: (
        <div className="tutorial-content">
          <h2>Interface Overview</h2>
          <p>The NeuroBlock interface is divided into three main panels:</p>

          <h3>Left Panel</h3>
          <p>
            The left panel contains the sidebar navigation and various tools for
            building your neural network:
          </p>
          <ul>
            <li>
              <strong>Layers</strong>: Add different types of neural network
              layers
            </li>
            <li>
              <strong>Templates</strong>: Load pre-configured model
              architectures
            </li>
            <li>
              <strong>Layer Params</strong>: View and select layers to edit
              their parameters
            </li>
            <li>
              <strong>Hyperparams</strong>: Configure training hyperparameters
            </li>
            <li>
              <strong>Training</strong>: Set up training options and monitor
              progress
            </li>
            <li>
              <strong>Model Config</strong>: View the overall model
              configuration
            </li>
            <li>
              <strong>Settings</strong>: Select datasets and other application
              settings
            </li>
          </ul>

          <h3>Center Panel</h3>
          <p>
            The center panel is the canvas where you build your neural network
            by dragging and dropping layers. You can connect layers by dragging
            from one node's output handle to another node's input handle.
          </p>

          <h3>Right Panel</h3>
          <p>
            The right panel displays visualizations, training metrics, and
            export options:
          </p>
          <ul>
            <li>Training progress and metrics</li>
            <li>Visualizations of model performance</li>
            <li>Options to export your trained model in various formats</li>
          </ul>
        </div>
      ),
    },
    {
      id: "building",
      title: "Building Models",
      content: (
        <div className="tutorial-content">
          <h2>Building Neural Network Models</h2>
          <p>
            Building a neural network in NeuroBlock is as simple as dragging and
            dropping layers onto the canvas and connecting them.
          </p>

          <h3>Layer Types</h3>
          <p>NeuroBlock provides various layer types for different purposes:</p>
          <ul>
            <li>
              <strong>Input Layer</strong>: The entry point for data into your
              neural network. Defines the shape of your input data.
            </li>
            <li>
              <strong>Dense Layer</strong>: Fully connected layer where each
              neuron is connected to every neuron in the previous layer. Used
              for learning complex patterns.
            </li>
            <li>
              <strong>Convolution Layer</strong>: Applies filters to detect
              spatial patterns in input data. Essential for image processing
              tasks.
            </li>
            <li>
              <strong>MaxPooling Layer</strong>: Reduces the spatial dimensions
              of the data by taking the maximum value in each window.
            </li>
            <li>
              <strong>Flatten Layer</strong>: Converts multi-dimensional input
              into a 1D vector. Often used between convolutional and dense
              layers.
            </li>
            <li>
              <strong>Dropout Layer</strong>: Randomly sets a fraction of inputs
              to zero during training to prevent overfitting.
            </li>
            <li>
              <strong>Batch Normalization Layer</strong>: Normalizes the
              activations of the previous layer for each batch.
            </li>
            <li>
              <strong>Attention Layer</strong>: Allows the model to focus on
              relevant parts of the input. Important for sequence data and NLP.
            </li>
            <li>
              <strong>Output Layer</strong>: The final layer that produces the
              model's predictions.
            </li>
          </ul>

          <h3>Connecting Layers</h3>
          <p>To connect layers:</p>
          <ol>
            <li>Click and drag from a layer's output handle (right side)</li>
            <li>Drop onto another layer's input handle (left side)</li>
            <li>A connection will be created between the layers</li>
          </ol>

          <h3>Configuring Layer Parameters</h3>
          <p>To configure a layer's parameters:</p>
          <ol>
            <li>Click on a layer in the canvas to select it</li>
            <li>
              Navigate to the <strong>Layer Params</strong> tab in the left
              sidebar
            </li>
            <li>The selected layer's parameters will appear in the panel</li>
            <li>Edit the parameters as needed</li>
            <li>Changes are applied automatically</li>
          </ol>
          <p className="tutorial-note">
            <i className="fas fa-info-circle"></i> <strong>Note:</strong> Layer
            parameters can only be edited in the Layer Params tab. This ensures
            all parameter changes are properly tracked and validated.
          </p>

          <h3>Using Templates</h3>
          <p>For quick starts, you can use pre-configured templates:</p>
          <ol>
            <li>Select the "Templates" tab in the left sidebar</li>
            <li>Click "Load" on the template you want to use</li>
            <li>The template will be loaded onto the canvas</li>
            <li>
              You can then customize the template by editing layer parameters
            </li>
          </ol>
          <p className="tutorial-note">
            <i className="fas fa-info-circle"></i> <strong>Tip:</strong>{" "}
            Templates are a great way to get started with common neural network
            architectures without having to build them from scratch.
          </p>
        </div>
      ),
    },
    {
      id: "training",
      title: "Training Models",
      content: (
        <div className="tutorial-content">
          <h2>Training Your Neural Network</h2>
          <p>
            Once you've built your neural network, you can train it on various
            datasets.
          </p>

          <h3>Selecting a Dataset</h3>
          <p>To select a dataset:</p>
          <ol>
            <li>Click the "Settings" tab in the left sidebar</li>
            <li>Choose a dataset from the dropdown menu</li>
          </ol>

          <h3>Configuring Training Parameters</h3>
          <p>To configure training parameters:</p>
          <ol>
            <li>Click the "Hyperparams" tab in the left sidebar</li>
            <li>Set the batch size, learning rate, and number of epochs</li>
            <li>
              Click the "Training" tab to set the optimizer and loss function
            </li>
          </ol>

          <h3>Starting Training</h3>
          <p>To start training:</p>
          <ol>
            <li>Make sure your model is properly connected and configured</li>
            <li>Click the "Train" button in the navigation bar at the top</li>
            <li>
              The training process will begin, and you can monitor progress in
              real-time
            </li>
          </ol>

          <h3>Monitoring Training Progress</h3>
          <p>During training, you can monitor:</p>
          <ul>
            <li>Current epoch and progress</li>
            <li>Loss and accuracy metrics</li>
            <li>Real-time visualizations in the right panel</li>
          </ul>

          <h3>Stopping Training</h3>
          <p>To stop training before completion:</p>
          <ol>
            <li>
              Click the "Stop" button in the navigation bar that appears during
              training
            </li>
            <li>The model will retain the training it has completed so far</li>
          </ol>
        </div>
      ),
    },
    {
      id: "visualizing",
      title: "Visualizations",
      content: (
        <div className="tutorial-content">
          <h2>Visualizing Model Performance</h2>
          <p>
            NeuroBlock provides various visualizations to help you understand
            your model's performance.
          </p>

          <h3>Accuracy and Loss Charts</h3>
          <p>The accuracy and loss charts show:</p>
          <ul>
            <li>Training accuracy and validation accuracy over epochs</li>
            <li>Training loss and validation loss over epochs</li>
          </ul>

          <h3>Confusion Matrix</h3>
          <p>For classification tasks, the confusion matrix shows:</p>
          <ul>
            <li>
              True positives, false positives, true negatives, and false
              negatives
            </li>
            <li>How well the model distinguishes between different classes</li>
          </ul>

          <h3>Selecting Visualizations</h3>
          <p>To switch between different visualizations:</p>
          <ol>
            <li>Use the visualization dropdown in the right panel</li>
            <li>Select the visualization you want to view</li>
          </ol>
        </div>
      ),
    },
    {
      id: "exporting",
      title: "Exporting Models",
      content: (
        <div className="tutorial-content">
          <h2>Exporting Your Trained Model</h2>
          <p>
            After training your model, you can export it in various formats for
            deployment or further development.
          </p>

          <h3>Available Export Formats</h3>
          <p>NeuroBlock supports the following export formats:</p>
          <ul>
            <li>
              <strong>Python</strong>: A Python script with the model
              implementation
            </li>
            <li>
              <strong>Keras</strong>: A .h5 file containing the trained model
            </li>
            <li>
              <strong>PyTorch</strong>: A PyTorch implementation of your model
            </li>
            <li>
              <strong>SavedModel</strong>: TensorFlow's SavedModel format
            </li>
            <li>
              <strong>Notebook</strong>: A Jupyter notebook with the model and
              visualizations
            </li>
          </ul>

          <h3>Exporting a Model</h3>
          <p>To export your model:</p>
          <ol>
            <li>
              Make sure your model has been saved (click "Save Model" in the
              navbar if needed)
            </li>
            <li>Click on the Export button in the navbar</li>
            <li>Select the export format you want from the dropdown</li>
            <li>The model will be downloaded to your computer</li>
          </ol>

          <h3>Using Exported Models</h3>
          <p>Depending on the export format, you can:</p>
          <ul>
            <li>Integrate the model into your own applications</li>
            <li>
              Continue development in frameworks like TensorFlow or PyTorch
            </li>
            <li>Deploy the model to production environments</li>
            <li>Share the model with others</li>
          </ul>
        </div>
      ),
    },
    {
      id: "tips",
      title: "Tips & Best Practices",
      content: (
        <div className="tutorial-content">
          <h2>Tips and Best Practices</h2>
          <p>
            Here are some tips and best practices to help you get the most out
            of NeuroBlock.
          </p>

          <h3>Model Architecture</h3>
          <ul>
            <li>
              Start with simple architectures and gradually add complexity
            </li>
            <li>Use templates as a starting point for common tasks</li>
            <li>
              Ensure your architecture is appropriate for your dataset (e.g.,
              CNNs for images)
            </li>
            <li>
              Add regularization layers (like Dropout) to prevent overfitting
            </li>
          </ul>

          <h3>Training</h3>
          <ul>
            <li>Start with a small number of epochs to test your model</li>
            <li>Monitor validation metrics to detect overfitting</li>
            <li>Adjust learning rate if training is unstable or slow</li>
            <li>
              Use appropriate batch sizes for your dataset (larger datasets can
              use larger batches)
            </li>
          </ul>

          <h3>Dataset Selection</h3>
          <ul>
            <li>Choose datasets that match your model architecture</li>
            <li>
              For image classification, use MNIST (simple) or CIFAR-10 (more
              complex)
            </li>
            <li>
              For tabular data, use Iris (simple) or Breast Cancer (binary
              classification)
            </li>
            <li>For regression tasks, use California Housing</li>
          </ul>

          <h3>Troubleshooting</h3>
          <ul>
            <li>
              If validation fails, check the error messages for specific issues
            </li>
            <li>Ensure all layers are properly connected</li>
            <li>
              Verify that your output layer's activation matches your task
              (Softmax for multi-class, Sigmoid for binary, None for regression)
            </li>
            <li>
              If training performance is poor, try adjusting hyperparameters or
              changing the architecture
            </li>
          </ul>
        </div>
      ),
    },
  ];

  return (
    <div className="tutorial-page">
      <div className="tutorial-sidebar">
        <h2>Tutorial Contents</h2>
        <ul className="tutorial-nav">
          {tutorialSections.map((section) => (
            <li
              key={section.id}
              className={activeSection === section.id ? "active" : ""}
              onClick={() => setActiveSection(section.id)}
            >
              {section.title}
            </li>
          ))}
        </ul>
        <div className="back-to-app">
          <NavLink to="/newbuild" className="back-button">
            <i className="fas fa-arrow-left"></i> Back to App
          </NavLink>
        </div>
      </div>

      <div className="tutorial-content-wrapper">
        {
          tutorialSections.find((section) => section.id === activeSection)
            ?.content
        }
      </div>
    </div>
  );
};

export default TutorialPage;
