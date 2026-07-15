import React, { useState } from "react";
import "../styles/components/TutorialPage.scss";
import { NavLink } from "react-router-dom";

// Define the tutorial sections
interface TutorialSection {
  id: string;
  title: string;
  icon: string;
  content: React.ReactNode;
}

const TutorialPage: React.FC = () => {
  const [activeSection, setActiveSection] = useState<string>("introduction");

  // Define all tutorial sections
  const tutorialSections: TutorialSection[] = [
    {
      id: "introduction",
      title: "Introduction",
      icon: "fas fa-home",
      content: (
        <div className="tutorial-content">
          <h2>Welcome to NeuroBlock</h2>
          <p>
            NeuroBlock is a powerful drag-and-drop interface for building,
            training, and deploying neural networks without writing a single
            line of code. This comprehensive tutorial will guide you through all
            the features and help you create amazing AI models.
          </p>

          <div className="feature-highlight">
            <h3>🚀 What You Can Do</h3>
            <div className="feature-grid">
              <div className="feature-card">
                <i className="fas fa-puzzle-piece"></i>
                <h4>Visual Model Building</h4>
                <p>
                  Drag and drop layers to create complex neural network
                  architectures
                </p>
              </div>
              <div className="feature-card">
                <i className="fas fa-cogs"></i>
                <h4>Intuitive Configuration</h4>
                <p>
                  Configure layer parameters and training settings with ease
                </p>
              </div>
              <div className="feature-card">
                <i className="fas fa-chart-line"></i>
                <h4>Real-time Training</h4>
                <p>
                  Train your models with live progress monitoring and
                  visualizations
                </p>
              </div>
              <div className="feature-card">
                <i className="fas fa-download"></i>
                <h4>Multiple Export Formats</h4>
                <p>
                  Export your trained models in Python, Keras, PyTorch, and more
                </p>
              </div>
            </div>
          </div>

          <div className="tutorial-note">
            <i className="fas fa-lightbulb"></i>
            <strong>Pro Tip:</strong> Start with the "Getting Started" section
            to learn the basics, then explore advanced features as you become
            more comfortable with the interface.
          </div>
        </div>
      ),
    },
    {
      id: "getting-started",
      title: "Getting Started",
      icon: "fas fa-rocket",
      content: (
        <div className="tutorial-content">
          <h2>Getting Started with NeuroBlock</h2>
          <p>
            Let's get you up and running with your first neural network! Follow
            these steps to create, train, and export your model.
          </p>

          <h3>1. Understanding the Interface</h3>
          <div className="interface-overview">
            <div className="interface-section">
              <h4>
                <i className="fas fa-bars"></i> Left Sidebar
              </h4>
              <ul>
                <li>
                  <strong>Layers:</strong> every building block — press the
                  &quot;+&quot; on a row to add it to the canvas
                </li>
                <li>
                  <strong>Templates:</strong> complete ready-made architectures
                  (Feedforward, Convolutional, ResNet-18, and more)
                </li>
                <li>
                  <strong>Activations:</strong> standalone activation functions
                  (ReLU, Sigmoid, Tanh, Softmax)
                </li>
              </ul>
            </div>
            <div className="interface-section">
              <h4>
                <i className="fas fa-palette"></i> Canvas
              </h4>
              <p>
                The main area where you build your network. A filled color rail
                on a block means the layer learns (it has trainable
                parameters); a hollow rail means it only transforms its input.
              </p>
            </div>
            <div className="interface-section">
              <h4>
                <i className="fas fa-chart-bar"></i> Right Panel
              </h4>
              <ul>
                <li>
                  <strong>Dataset:</strong> pick your dataset and set
                  hyperparameters (batch size, epochs, optimizer, loss)
                </li>
                <li>
                  <strong>Parameters:</strong> click any block on the canvas to
                  edit its settings here
                </li>
                <li>
                  <strong>Train:</strong> live loss and accuracy charts while
                  your model learns
                </li>
              </ul>
            </div>
          </div>

          <h3>2. Your First Model</h3>
          <div className="step-by-step">
            <div className="step">
              <div className="step-number">1</div>
              <div className="step-content">
                <h4>Choose a Dataset</h4>
                <p>
                  In the right panel's <strong>Dataset</strong> tab, pick a
                  dataset — start with <strong>Iris</strong>, it trains in
                  seconds
                </p>
              </div>
            </div>
            <div className="step">
              <div className="step-number">2</div>
              <div className="step-content">
                <h4>Add Layers</h4>
                <p>
                  The canvas starts with Input and Output blocks. Add a{" "}
                  <strong>Dense</strong> layer from the left sidebar with its
                  &quot;+&quot; button
                </p>
              </div>
            </div>
            <div className="step">
              <div className="step-number">3</div>
              <div className="step-content">
                <h4>Connect the Blocks</h4>
                <p>
                  Drag from a block's right handle to the next block's left
                  handle: Input → Dense → Output
                </p>
              </div>
            </div>
            <div className="step">
              <div className="step-number">4</div>
              <div className="step-content">
                <h4>Configure and Save</h4>
                <p>
                  Click a block to tune it in the <strong>Parameters</strong>{" "}
                  tab, then hit <strong>Save Model</strong> in the top bar
                </p>
              </div>
            </div>
            <div className="step">
              <div className="step-number">5</div>
              <div className="step-content">
                <h4>Train and Watch</h4>
                <p>
                  Press the orange <strong>Train</strong> button — the{" "}
                  <strong>Train</strong> tab shows live loss and accuracy as
                  your network learns
                </p>
              </div>
            </div>
          </div>

          <div className="tutorial-note">
            <i className="fas fa-exclamation-triangle"></i>
            <strong>Important:</strong> Always save your model before training!
            The system will warn you if you try to train without saving first.
          </div>
        </div>
      ),
    },
    {
      id: "building-models",
      title: "Building Models",
      icon: "fas fa-puzzle-piece",
      content: (
        <div className="tutorial-content">
          <h2>Building Neural Network Models</h2>
          <p>
            Learn how to create sophisticated neural network architectures using
            our intuitive drag-and-drop interface.
          </p>

          <h3>Available Layer Types</h3>
          <div className="layer-types">
            <div className="layer-category">
              <h4>Input & Output Layers</h4>
              <div className="tutorial-layer-grid">
                <div className="tutorial-layer-item">
                  <i className="fas fa-sign-in-alt"></i>
                  <strong>Input Layer</strong>
                  <p>Defines the shape of your input data</p>
                </div>
                <div className="tutorial-layer-item">
                  <i className="fas fa-sign-out-alt"></i>
                  <strong>Output Layer</strong>
                  <p>Produces the final predictions</p>
                </div>
              </div>
            </div>

            <div className="layer-category">
              <h4>Core Neural Network Layers</h4>
              <div className="tutorial-layer-grid">
                <div className="tutorial-layer-item">
                  <i className="fas fa-network-wired"></i>
                  <strong>Dense Layer</strong>
                  <p>Fully connected layer for learning complex patterns</p>
                </div>
                <div className="tutorial-layer-item">
                  <i className="fas fa-filter"></i>
                  <strong>Convolution Layer</strong>
                  <p>Detects spatial patterns in images</p>
                </div>
                <div className="tutorial-layer-item">
                  <i className="fas fa-compress-arrows-alt"></i>
                  <strong>MaxPooling Layer</strong>
                  <p>Reduces spatial dimensions</p>
                </div>
                <div className="tutorial-layer-item">
                  <i className="fas fa-vector-square"></i>
                  <strong>Flatten Layer</strong>
                  <p>Converts multi-dimensional data to 1D</p>
                </div>
              </div>
            </div>

            <div className="layer-category">
              <h4>Advanced Layers</h4>
              <div className="tutorial-layer-grid">
                <div className="tutorial-layer-item">
                  <i className="fas fa-random"></i>
                  <strong>Dropout Layer</strong>
                  <p>Prevents overfitting by randomly zeroing inputs</p>
                </div>
                <div className="tutorial-layer-item">
                  <i className="fas fa-balance-scale"></i>
                  <strong>Batch Normalization</strong>
                  <p>Normalizes activations for stable training</p>
                </div>
                <div className="tutorial-layer-item">
                  <i className="fas fa-eye"></i>
                  <strong>Attention Layer</strong>
                  <p>Focuses on relevant parts of input data</p>
                </div>
                <div className="tutorial-layer-item">
                  <i className="fas fa-bolt"></i>
                  <strong>Activation Layer</strong>
                  <p>Applies non-linear transformations</p>
                </div>
              </div>
            </div>
          </div>

          <h3>Building Your Architecture</h3>
          <div className="building-steps">
            <div className="building-step">
              <h4>1. Start with Input Layer</h4>
              <p>
                Drag an Input layer to the canvas. This defines your data's
                shape.
              </p>
            </div>
            <div className="building-step">
              <h4>2. Add Processing Layers</h4>
              <p>Add layers that process your data (Dense, Conv, etc.)</p>
            </div>
            <div className="building-step">
              <h4>3. Connect the Layers</h4>
              <p>Drag from output handles (right) to input handles (left)</p>
            </div>
            <div className="building-step">
              <h4>4. End with Output Layer</h4>
              <p>Add an Output layer to produce predictions</p>
            </div>
          </div>

          <h3>Using Templates</h3>
          <p>Templates provide pre-built architectures for common tasks:</p>
          <ul>
            <li>
              <strong>Simple Classifier:</strong> Basic neural network for
              classification
            </li>
            <li>
              <strong>CNN for Images:</strong> Convolutional network for image
              processing
            </li>
            <li>
              <strong>Deep Network:</strong> Multi-layer network for complex
              tasks
            </li>
            <li>
              <strong>ResNet:</strong> Advanced architecture with skip
              connections
            </li>
          </ul>

          <div className="tutorial-note">
            <i className="fas fa-info-circle"></i>
            <strong>Tip:</strong> Start with templates to understand good
            architectures, then customize them for your specific needs.
          </div>
        </div>
      ),
    },
    {
      id: "training-models",
      title: "Training Models",
      icon: "fas fa-cogs",
      content: (
        <div className="tutorial-content">
          <h2>Training Your Neural Network</h2>
          <p>
            Learn how to configure training parameters, monitor progress, and
            optimize your model's performance.
          </p>

          <h3>Pre-Training Checklist</h3>
          <div className="checklist">
            <div className="checklist-item">
              <i className="fas fa-check-circle"></i>
              <span>Dataset selected in Settings tab</span>
            </div>
            <div className="checklist-item">
              <i className="fas fa-check-circle"></i>
              <span>Model architecture is complete and connected</span>
            </div>
            <div className="checklist-item">
              <i className="fas fa-check-circle"></i>
              <span>Model has been saved (Save Model button)</span>
            </div>
            <div className="checklist-item">
              <i className="fas fa-check-circle"></i>
              <span>Training parameters configured</span>
            </div>
          </div>

          <h3>Configuring Training Parameters</h3>
          <div className="training-params">
            <div className="param-group">
              <h4>Basic Parameters</h4>
              <ul>
                <li>
                  <strong>Epochs:</strong> Number of complete passes through the
                  dataset
                </li>
                <li>
                  <strong>Batch Size:</strong> Number of samples processed
                  before updating weights
                </li>
                <li>
                  <strong>Learning Rate:</strong> Step size for weight updates
                </li>
              </ul>
            </div>
            <div className="param-group">
              <h4>Advanced Parameters</h4>
              <ul>
                <li>
                  <strong>Optimizer:</strong> Algorithm for updating weights
                  (Adam, SGD, etc.)
                </li>
                <li>
                  <strong>Loss Function:</strong> Function to measure prediction
                  errors
                </li>
                <li>
                  <strong>Validation Split:</strong> Percentage of data used for
                  validation
                </li>
              </ul>
            </div>
          </div>

          <h3>The Training Process</h3>
          <div className="training-process">
            <div className="process-step">
              <div className="step-icon">
                <i className="fas fa-save"></i>
              </div>
              <div className="step-content">
                <h4>1. Save Your Model</h4>
                <p>
                  Click "Save Model" in the navbar to save your architecture
                </p>
              </div>
            </div>
            <div className="process-step">
              <div className="step-icon">
                <i className="fas fa-play"></i>
              </div>
              <div className="step-content">
                <h4>2. Start Training</h4>
                <p>Click "Train" to begin the training process</p>
              </div>
            </div>
            <div className="process-step">
              <div className="step-icon">
                <i className="fas fa-chart-line"></i>
              </div>
              <div className="step-content">
                <h4>3. Monitor Progress</h4>
                <p>Watch real-time metrics and visualizations</p>
              </div>
            </div>
            <div className="process-step">
              <div className="step-icon">
                <i className="fas fa-stop"></i>
              </div>
              <div className="step-content">
                <h4>4. Stop When Ready</h4>
                <p>Click "Stop" to halt training at any time</p>
              </div>
            </div>
          </div>

          <h3>Understanding Training Metrics</h3>
          <div className="metrics-explanation">
            <div className="metric">
              <h4>Accuracy</h4>
              <p>Percentage of correct predictions. Higher is better.</p>
            </div>
            <div className="metric">
              <h4>Loss</h4>
              <p>Measure of prediction errors. Lower is better.</p>
            </div>
            <div className="metric">
              <h4>Validation Metrics</h4>
              <p>Performance on unseen data. Helps detect overfitting.</p>
            </div>
          </div>

          <div className="tutorial-note warning">
            <i className="fas fa-exclamation-triangle"></i>
            <strong>Important:</strong> The system will warn you if you try to
            train without saving first. Always save your model before training
            to avoid losing your work!
          </div>
        </div>
      ),
    },
    {
      id: "visualizations",
      title: "Visualizations",
      icon: "fas fa-chart-bar",
      content: (
        <div className="tutorial-content">
          <h2>Understanding Model Performance</h2>
          <p>
            NeuroBlock provides comprehensive visualizations to help you
            understand your model's performance and identify areas for
            improvement.
          </p>

          <h3>Training Progress Charts</h3>
          <div className="chart-explanation">
            <div className="chart-type">
              <h4>Accuracy Chart</h4>
              <p>
                Shows training and validation accuracy over epochs. Look for:
              </p>
              <ul>
                <li>Steady increase in accuracy</li>
                <li>Validation accuracy close to training accuracy</li>
                <li>No signs of overfitting (validation accuracy dropping)</li>
              </ul>
            </div>
            <div className="chart-type">
              <h4>Loss Chart</h4>
              <p>Shows training and validation loss over epochs. Look for:</p>
              <ul>
                <li>Decreasing loss values</li>
                <li>Validation loss following training loss</li>
                <li>No divergence between training and validation</li>
              </ul>
            </div>
          </div>

          <h3>Confusion Matrix</h3>
          <p>For classification tasks, the confusion matrix shows:</p>
          <div className="confusion-matrix-explanation">
            <div className="matrix-element">
              <strong>True Positives (TP):</strong> Correctly predicted positive
              cases
            </div>
            <div className="matrix-element">
              <strong>False Positives (FP):</strong> Incorrectly predicted
              positive cases
            </div>
            <div className="matrix-element">
              <strong>True Negatives (TN):</strong> Correctly predicted negative
              cases
            </div>
            <div className="matrix-element">
              <strong>False Negatives (FN):</strong> Incorrectly predicted
              negative cases
            </div>
          </div>

          <h3>Dataset-Specific Visualizations</h3>
          <div className="dataset-visualizations">
            <div className="dataset-viz">
              <h4>Image Datasets (MNIST, CIFAR-10)</h4>
              <ul>
                <li>Confusion matrix showing class-wise performance</li>
                <li>Sample predictions with confidence scores</li>
              </ul>
            </div>
            <div className="dataset-viz">
              <h4>Regression Datasets (California Housing)</h4>
              <ul>
                <li>Predicted vs actual values scatter plot</li>
                <li>Residual analysis</li>
                <li>R² and RMSE metrics</li>
              </ul>
            </div>
            <div className="dataset-viz">
              <h4>Classification Datasets (Iris, Breast Cancer)</h4>
              <ul>
                <li>Confusion matrix</li>
                <li>Classification report with precision/recall</li>
              </ul>
            </div>
          </div>

          <h3>Interpreting Results</h3>
          <div className="interpretation-guide">
            <div className="interpretation-item good">
              <h4>✅ Good Training</h4>
              <ul>
                <li>Both training and validation metrics improve</li>
                <li>Validation metrics close to training metrics</li>
                <li>No signs of overfitting</li>
              </ul>
            </div>
            <div className="interpretation-item warning">
              <h4>⚠️ Overfitting</h4>
              <ul>
                <li>Training metrics improve but validation doesn't</li>
                <li>Gap between training and validation metrics</li>
                <li>Validation metrics start decreasing</li>
              </ul>
            </div>
            <div className="interpretation-item poor">
              <h4>❌ Poor Performance</h4>
              <ul>
                <li>Both training and validation metrics are low</li>
                <li>No improvement over epochs</li>
                <li>High loss values</li>
              </ul>
            </div>
          </div>
        </div>
      ),
    },
    {
      id: "exporting",
      title: "Exporting Models",
      icon: "fas fa-download",
      content: (
        <div className="tutorial-content">
          <h2>Exporting Your Trained Model</h2>
          <p>
            Once your model is trained, you can export it in various formats for
            deployment, further development, or sharing with others.
          </p>

          <h3>Export Formats</h3>
          <div className="export-formats">
            <div className="export-format">
              <div className="format-icon">
                <i className="fab fa-python"></i>
              </div>
              <div className="format-info">
                <h4>Python Script</h4>
                <p>
                  Complete Python implementation with your model architecture
                  and training code
                </p>
                <ul>
                  <li>Ready-to-run Python script</li>
                  <li>Includes data loading and preprocessing</li>
                  <li>Model compilation and training code</li>
                  <li>Evaluation and prediction functions</li>
                </ul>
              </div>
            </div>

            <div className="export-format">
              <div className="format-icon">
                <i className="fas fa-cube"></i>
              </div>
              <div className="format-info">
                <h4>Keras Model</h4>
                <p>Saved Keras model file (.keras) with trained weights</p>
                <ul>
                  <li>Load directly in TensorFlow/Keras</li>
                  <li>Preserves trained weights</li>
                  <li>Easy integration into existing projects</li>
                </ul>
              </div>
            </div>

            <div className="export-format">
              <div className="format-icon">
                <i className="fas fa-fire"></i>
              </div>
              <div className="format-info">
                <h4>PyTorch Model</h4>
                <p>PyTorch implementation of your model architecture</p>
                <ul>
                  <li>PyTorch-compatible code</li>
                  <li>Model definition and training loop</li>
                  <li>Easy to modify and extend</li>
                </ul>
              </div>
            </div>

            <div className="export-format">
              <div className="format-icon">
                <i className="fas fa-save"></i>
              </div>
              <div className="format-info">
                <h4>SavedModel</h4>
                <p>TensorFlow SavedModel format for production deployment</p>
                <ul>
                  <li>Industry-standard format</li>
                  <li>Optimized for production</li>
                  <li>Supports TensorFlow Serving</li>
                </ul>
              </div>
            </div>

            <div className="export-format">
              <div className="format-icon">
                <i className="fas fa-book"></i>
              </div>
              <div className="format-info">
                <h4>Jupyter Notebook</h4>
                <p>Interactive notebook with your model and visualizations</p>
                <ul>
                  <li>Interactive exploration</li>
                  <li>Includes visualizations</li>
                  <li>Educational and documentation purposes</li>
                </ul>
              </div>
            </div>

            <div className="export-format">
              <div className="format-icon">
                <i className="fas fa-mobile-alt"></i>
              </div>
              <div className="format-info">
                <h4>TensorFlow Lite</h4>
                <p>
                  A compact .tflite file — the same model, small enough to run
                  on a phone or microcontroller
                </p>
                <ul>
                  <li>Runs on Android, iOS, and edge devices</li>
                  <li>Single small file, fast inference</li>
                  <li>See how models ship inside real apps</li>
                </ul>
              </div>
            </div>
          </div>

          <h3>Export Process</h3>
          <div className="export-process">
            <div className="export-step">
              <div className="step-number">1</div>
              <div className="step-content">
                <h4>Ensure Model is Saved</h4>
                <p>
                  Make sure your model architecture is saved before exporting
                </p>
              </div>
            </div>
            <div className="export-step">
              <div className="step-number">2</div>
              <div className="step-content">
                <h4>Click Export Button</h4>
                <p>Click the Export button in the navigation bar</p>
              </div>
            </div>
            <div className="export-step">
              <div className="step-number">3</div>
              <div className="step-content">
                <h4>Select Format</h4>
                <p>Choose your desired export format from the dropdown</p>
              </div>
            </div>
            <div className="export-step">
              <div className="step-number">4</div>
              <div className="step-content">
                <h4>Download</h4>
                <p>
                  Your model will be automatically downloaded to your computer
                </p>
              </div>
            </div>
          </div>

          <h3>Using Exported Models</h3>
          <div className="usage-examples">
            <div className="usage-example">
              <h4>Python Script</h4>
              <pre>
                <code>python exported_model.py</code>
              </pre>
            </div>
            <div className="usage-example">
              <h4>Keras Model</h4>
              <pre>
                <code>
                  model = tf.keras.models.load_model('model.h5') predictions =
                  model.predict(data)
                </code>
              </pre>
            </div>
            <div className="usage-example">
              <h4>PyTorch Model</h4>
              <pre>
                <code>
                  from model import MyModel model = MyModel()
                  model.load_state_dict(torch.load('model.pth'))
                </code>
              </pre>
            </div>
          </div>

          <div className="tutorial-note">
            <i className="fas fa-info-circle"></i>
            <strong>Tip:</strong> Choose the export format based on your
            deployment needs. Python scripts are great for learning, while
            SavedModel is ideal for production deployment.
          </div>
        </div>
      ),
    },
    {
      id: "best-practices",
      title: "Best Practices",
      icon: "fas fa-star",
      content: (
        <div className="tutorial-content">
          <h2>Best Practices & Tips</h2>
          <p>
            Learn from experience! Here are proven strategies to build better
            models and avoid common pitfalls.
          </p>

          <h3>Model Architecture Best Practices</h3>
          <div className="best-practices-grid">
            <div className="practice-card">
              <div className="practice-icon">
                <i className="fas fa-layer-group"></i>
              </div>
              <h4>Start Simple</h4>
              <p>
                Begin with simple architectures and gradually add complexity. A
                simple model that works is better than a complex one that
                doesn't.
              </p>
            </div>
            <div className="practice-card">
              <div className="practice-icon">
                <i className="fas fa-link"></i>
              </div>
              <h4>Proper Connections</h4>
              <p>
                Ensure all layers are properly connected. Disconnected layers
                won't contribute to your model's learning.
              </p>
            </div>
            <div className="practice-card">
              <div className="practice-icon">
                <i className="fas fa-shield-alt"></i>
              </div>
              <h4>Use Regularization</h4>
              <p>
                Add Dropout layers to prevent overfitting, especially in deep
                networks.
              </p>
            </div>
            <div className="practice-card">
              <div className="practice-icon">
                <i className="fas fa-balance-scale"></i>
              </div>
              <h4>Batch Normalization</h4>
              <p>
                Use Batch Normalization layers to stabilize training and improve
                convergence.
              </p>
            </div>
          </div>

          <h3>Dataset Selection Guide</h3>
          <div className="dataset-guide">
            <div className="dataset-category">
              <h4>For Beginners</h4>
              <ul>
                <li>
                  <strong>MNIST:</strong> Simple digit recognition (10 classes)
                </li>
                <li>
                  <strong>Iris:</strong> Simple flower classification (3
                  classes)
                </li>
              </ul>
            </div>
            <div className="dataset-category">
              <h4>For Intermediate Users</h4>
              <ul>
                <li>
                  <strong>CIFAR-10:</strong> Color image classification (10
                  classes)
                </li>
                <li>
                  <strong>Breast Cancer:</strong> Binary classification with
                  tabular data
                </li>
              </ul>
            </div>
            <div className="dataset-category">
              <h4>For Advanced Users</h4>
              <ul>
                <li>
                  <strong>California Housing:</strong> Regression task with
                  multiple features
                </li>
                <li>
                  <strong>Custom Datasets:</strong> Upload your own data
                </li>
              </ul>
            </div>
          </div>

          <h3>Training Optimization</h3>
          <div className="training-tips">
            <div className="tip-group">
              <h4>Learning Rate</h4>
              <ul>
                <li>Start with 0.001 (default Adam learning rate)</li>
                <li>If training is unstable, reduce to 0.0001</li>
                <li>If training is too slow, increase to 0.01</li>
              </ul>
            </div>
            <div className="tip-group">
              <h4>Batch Size</h4>
              <ul>
                <li>32 is a good starting point</li>
                <li>Larger batches (64, 128) for more stable gradients</li>
                <li>Smaller batches (16, 8) for better generalization</li>
              </ul>
            </div>
            <div className="tip-group">
              <h4>Epochs</h4>
              <ul>
                <li>Start with 10-20 epochs</li>
                <li>Monitor validation metrics to avoid overfitting</li>
                <li>Use early stopping if available</li>
              </ul>
            </div>
          </div>

          <h3>Common Issues & Solutions</h3>
          <div className="troubleshooting">
            <div className="issue">
              <h4>❌ Model Not Learning</h4>
              <p>
                <strong>Possible causes:</strong>
              </p>
              <ul>
                <li>Learning rate too high or too low</li>
                <li>Inappropriate loss function for the task</li>
                <li>Poor data preprocessing</li>
              </ul>
              <p>
                <strong>Solutions:</strong>
              </p>
              <ul>
                <li>Try different learning rates</li>
                <li>Check loss function matches your task</li>
                <li>Verify data is properly normalized</li>
              </ul>
            </div>
            <div className="issue">
              <h4>❌ Overfitting</h4>
              <p>
                <strong>Signs:</strong>
              </p>
              <ul>
                <li>Training accuracy high, validation accuracy low</li>
                <li>
                  Validation loss increasing while training loss decreases
                </li>
              </ul>
              <p>
                <strong>Solutions:</strong>
              </p>
              <ul>
                <li>Add Dropout layers</li>
                <li>Reduce model complexity</li>
                <li>Increase training data</li>
                <li>Use early stopping</li>
              </ul>
            </div>
            <div className="issue">
              <h4>❌ Poor Performance</h4>
              <p>
                <strong>Possible causes:</strong>
              </p>
              <ul>
                <li>Insufficient model capacity</li>
                <li>Inappropriate architecture for the task</li>
                <li>Poor hyperparameter choices</li>
              </ul>
              <p>
                <strong>Solutions:</strong>
              </p>
              <ul>
                <li>Increase model complexity</li>
                <li>Try different architectures</li>
                <li>Experiment with hyperparameters</li>
              </ul>
            </div>
          </div>

          <h3>Workflow Best Practices</h3>
          <div className="workflow-tips">
            <div className="workflow-tip">
              <i className="fas fa-save"></i>
              <h4>Save Frequently</h4>
              <p>
                Always save your model before training. The system will warn you
                if you forget!
              </p>
            </div>
            <div className="workflow-tip">
              <i className="fas fa-chart-line"></i>
              <h4>Monitor Training</h4>
              <p>
                Watch the training metrics to catch issues early and stop if
                needed.
              </p>
            </div>
            <div className="workflow-tip">
              <i className="fas fa-copy"></i>
              <h4>Use Templates</h4>
              <p>
                Start with templates and modify them rather than building from
                scratch.
              </p>
            </div>
            <div className="workflow-tip">
              <i className="fas fa-download"></i>
              <h4>Export Your Work</h4>
              <p>
                Export successful models to preserve your work and share with
                others.
              </p>
            </div>
          </div>

          <div className="tutorial-note success">
            <i className="fas fa-check-circle"></i>
            <strong>Remember:</strong> Machine learning is iterative. Don't be
            afraid to experiment and learn from your mistakes!
          </div>
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
              <span className="nav-icon">
                <i className={section.icon}></i>
              </span>
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
