# Drag-and-Drop Neural Network Builder

A web-based tool that allows users to visually design, train, and export custom neural networks using an intuitive drag-and-drop interface. This project is designed for users ranging from machine learning beginners to advanced practitioners, making neural network development more accessible and interactive.

## 🚀 Key Features

- **Drag-and-Drop Interface:** Build custom neural networks by dragging and connecting different types of layers.
- **Predefined Templates:** Quickly start with ready-made neural network templates for common tasks.
- **Real-Time Training Visualization:** Monitor training progress with real-time graphs and logs.
- **Export Options:** Export trained models in Python Script, Jupyter Notebook, TensorFlow SavedModel and Keras Model.
- **Customizable Parameters:** Edit layer-specific parameters (neurons, activation functions, dropout rate, etc.).
- **ResNet Support:** Build and train deep residual networks with ResNet blocks.

## 🎯 Target Audience

- Machine learning beginners and enthusiasts.
- Data scientists and researchers needing a quick model prototyping tool.
- Developers interested in neural network visualization tools.

## 💡 Motivation

Developing neural networks can be complex and time-consuming, especially for beginners. This project simplifies that process by offering an interactive visual builder and eliminating the need for manual coding during model design.

---

## 🛠 Tech Stack

### **Frontend:**

- **React** with **TypeScript** for the interactive UI
- **React Flow** for the drag-and-drop interface
- **SCSS** for custom styling
- **Bootstrap** for responsive design
- **Chart.js (react-chartjs-2)** for visualizing training progress
- **Axios** for HTTP requests
- **Socket.IO Client** for real-time updates

### **Backend:**

- **Flask** for REST API development
- **Flask-SocketIO** for real-time communication
- **TensorFlow** and **Keras** for building and training neural networks
- **Scikit-learn** for evaluation metrics (confusion matrix, regression metrics)
- **Pandas** for data handling
- **Matplotlib** and **Seaborn** for plotting metrics

### **Tools:**

- **Postman** for API testing
- **GitHub** for version control and collaboration
- **Unittest** and **Coverage.py** for automated testing

---

## ⚙️ Installation Guide

### 1. **Prerequisites**

- **Python:** 3.12.4 or later
- **Node.js:** Latest LTS version (18.x or later)
- **npm:** Latest version (comes with Node.js)
- **pip:** Python package installer
- **Poetry:** Python dependency management tool (recommended)

### 2. **Backend Setup (Flask)**

#### Option 1: Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/aryamangoenka/dnd-neural-network.git
cd dnd-neural-network

# Install Poetry if you don't have it already
# For macOS/Linux/WSL:
curl -sSL https://install.python-poetry.org | python3 -

# For Windows PowerShell:
# (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Install dependencies using Poetry
poetry install

# Run the Flask server using the provided script
chmod +x ./run_backend.sh  # Make the script executable (macOS/Linux only)
./run_backend.sh
```

The `run_backend.sh` script automatically:
- Sets the correct Python path to ensure imports work properly
- Activates the Poetry virtual environment
- Runs the Flask server with the proper configuration

#### About run_backend.sh

The `run_backend.sh` is a convenience script that simplifies starting the backend server. It handles several important tasks:

```bash
#!/bin/bash
cd "$(dirname "$0")"                     # Change to the directory containing the script
export PYTHONPATH="$PYTHONPATH:$(pwd)"   # Add the current directory to Python path
poetry run python -m backend.main        # Run the backend through Poetry's environment
```

This script ensures:
1. The proper Python module paths are set up
2. The application runs in the Poetry-managed virtual environment
3. You don't need to manually activate the virtual environment
4. The backend server starts with consistent configuration

#### Why Poetry?

Poetry offers several advantages over traditional Python package management:

- **Dependency Resolution**: Poetry automatically resolves dependencies and their versions to avoid conflicts.
- **Reproducible Environments**: The `poetry.lock` file ensures everyone uses the exact same dependency versions.
- **Virtual Environment Management**: Poetry automatically creates and manages virtual environments for your projects.
- **Simple Commands**: Poetry simplifies commands for installing, updating, and managing packages.
- **Project Isolation**: Each project's dependencies are isolated, preventing conflicts between different projects.

#### Managing Dependencies with Poetry

```bash
# Add a new dependency
poetry add package-name

# Add a development dependency
poetry add --dev package-name

# Update all dependencies
poetry update

# Show currently installed packages
poetry show

# Activate the virtual environment shell
poetry shell
```

#### Option 2: Using pip and venv

```bash
# Clone the repository
git clone https://github.com/aryamangoenka/dnd-neural-network.git
cd dnd-neural-network

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install backend dependencies
pip install -r requirements.txt

# Run the Flask server
cd backend
python3 main.py
```

### 3. **Frontend Setup (React)**

```bash
# Navigate to the frontend directory
cd frontend

# Install frontend dependencies
npm install

# Start the development server
npm run dev
```

### 4. **Access the Application**

- Open your browser and navigate to `http://localhost:3000` to access the frontend.
- The backend API will run on `http://localhost:5173` by default.

### 5. **Common Issues & Fixes**

- **Issue:** Flask server not starting.  
  **Fix:** Ensure the virtual environment is activated and all dependencies are installed.

- **Issue:** React app not loading.  
  **Fix:** Ensure Node.js and npm are correctly installed. Run `npm install` again if needed.

---

## 📝 Testing

The project includes comprehensive unit tests to ensure reliability and correct functionality. Tests cover model building, API endpoints, WebSocket communication, and ResNet functionality.

### Running Tests

#### Option 1: Using Poetry (Recommended)

```bash
# Run tests with the provided script
chmod +x ./run_tests.sh  # Make the script executable (macOS/Linux only)
./run_tests.sh
```

The `run_tests.sh` script automatically:
- Sets the correct Python path
- Activates the Poetry virtual environment
- Runs the test suite with proper configuration

You can also pass additional options to the test runner:

```bash
# Run tests with verbose output
./run_tests.sh -v

# Generate HTML coverage reports
./run_tests.sh --html
```

#### About run_tests.sh

The `run_tests.sh` is a convenience script that simplifies running tests:

```bash
#!/bin/bash
cd "$(dirname "$0")"                     # Change to the directory containing the script
export PYTHONPATH="$PYTHONPATH:$(pwd)"   # Add the current directory to Python path
poetry run python -m backend.run_tests "$@"  # Run tests through Poetry's environment with any arguments passed
```

This script ensures:
1. Tests run in the Poetry-managed virtual environment
2. The proper Python module paths are set up
3. Command-line arguments are passed to the test runner
4. Tests are run with consistent configuration

#### Option 2: Using venv

To run all tests with coverage reporting:

```bash
# Navigate to the backend directory
cd backend

# Install test dependencies
pip install coverage

# Run tests with coverage
python run_tests.py
```

For more verbose output and HTML coverage reports:

```bash
# Run with verbose output
python run_tests.py -v

# Generate HTML coverage reports
python run_tests.py --html
```

### Test Components

1. **Model Builder Tests** (`test_main.py`):

   - Tests for building various neural network architectures
   - ResNet block creation and functionality tests
   - Custom attention layer tests

2. **API Endpoint Tests** (`test_api.py`):
   - REST API endpoint tests
   - Model export functionality tests
   - WebSocket communication tests

### Continuous Integration

It's recommended to run tests before committing changes to ensure all functionality works as expected.

---

## 📖 Usage Guide

### 1. **Select a Dataset**

- On the **Home Page**, choose from predefined datasets like **Iris**, **MNIST**, **Breast Cancer**, **California Housing**, or **CIFAR-10**.

### 2. **Build the Neural Network**

- Go to the **Build Page**.
- Drag and drop different layers (Dense, Convolutional, Dropout, etc.) from the sidebar onto the canvas.
- Connect layers logically to form a valid neural network structure.

### 3. **Configure Layer Parameters**

- Click on any layer to open the **Parameter Sidebar**.
- Customize settings like the number of neurons, activation functions, dropout rate, etc.

### 4. **Train the Model**

- Click the **Train** button.
- Monitor real-time progress through **interactive** graphs.
- View logs and metrics during the training process.

### 5. **Deploy the Trained Model**

- Once training is complete, go to the **Deploy Page**.
- Choose from multiple export formats:
  - **Python Script** (`.py`)
  - **Jupyter Notebook** (`.ipynb`)
  - **TensorFlow SavedModel**
  - **Keras Model** (`.keras`)
  - **PyTorch Model** (`.py`)
- Download the model for further use or deployment.

---

## 🔮 Future Enhancements

- **Custom Dataset Upload:** Enable users to upload their own datasets for model training.
- **Advanced Architectures:** Support more advanced model architectures and layer types.
- **Mobile Responsiveness:** Optimize the interface for mobile and tablet devices.
- **Advanced Visualizations:** Add more in-depth training visualizations (e.g., ROC Curve, Precision-Recall Curve).
- **Template Marketplace:** Introduce a marketplace for sharing and downloading user-created model templates.

---

## 🙏 Acknowledgments

- **React Flow:** For providing the core drag-and-drop functionality.
- **TensorFlow/Keras:** For model building and training.
- **Flask-SocketIO:** For real-time communication.
- **Poetry:** For Python dependency management.
- I would like to thank My Prof. Cooper Sigrist for his mentorship and guidance in this project
- Inspiration from existing neural network visual builders and educational tools.

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License
Permission is hereby granted, free of charge, to any person obtaining a copy of this software...