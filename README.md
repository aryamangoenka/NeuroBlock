# NeuroBlock - Drag-and-Drop Neural Network Builder

A comprehensive web-based tool that allows users to visually design, train, and export custom neural networks using an intuitive drag-and-drop interface. This project consists of a Python Flask backend, a React frontend application, and a Next.js landing page.

## 🚀 Quick Start Guide

Follow these step-by-step instructions to clone and run the project on your local machine.

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.12** (Python 3.13 is not supported yet)
- **Node.js** (Latest LTS version)
- **Poetry** for Python dependency management
- **Git**

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/aryamangoenka/DND-neural.git

# Navigate to the project directory
cd DND-neural
```

### Step 2: Backend Setup

```bash
# Navigate to the project directory
cd DND-Neural-Network

# Add Poetry to your PATH (if you get "poetry not found" error)
export PATH="/Users/$USER/.local/bin:$PATH"

# Set Python version to 3.12 (required for compatibility)
poetry env use python3.12

# Update dependencies
poetry lock

# Install all dependencies
poetry install

# Start the backend server
./run_backend.sh
```

The backend will start on `http://localhost:5000`

### Step 3: Frontend Setup

Open a new terminal window and run:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will start on `http://localhost:5173`

## 🖥️ Accessing the Application

After completing the setup:

1. Backend API will be running on `http://localhost:5000`
2. Frontend application will be available at `http://localhost:5173`

## 🔧 Troubleshooting

### Common Issues

**Backend: Poetry not found**

```bash
# Add Poetry to PATH
export PATH="/Users/$USER/.local/bin:$PATH"
```

**Backend: Python version error**

```bash
# Ensure you're using Python 3.12
poetry env use python3.12
poetry lock
poetry install
```

**Backend: Permission denied for run_backend.sh**

```bash
chmod +x ./run_backend.sh
```

**Backend: Port already in use**

```bash
# Check if port 5000 is in use
lsof -i :5000
# Kill the process if needed
kill -9 <PID>
```

## 🛠️ Project Structure

```
DND-Neural-Network/
├── backend/                 # Flask backend
├── frontend/               # React frontend
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Poetry configuration
└── README.md              # This file
```

## 🎯 Key Features

- **Drag-and-Drop Interface:** Build neural networks visually
- **Real-Time Training:** Monitor progress with live graphs
- **Multiple Export Formats:** Python, Jupyter, TensorFlow, Keras
- **Pre-built Templates:** Start with ready-made templates
- **Dataset Support:** MNIST, CIFAR-10, Iris, etc.
- **ResNet Support:** Create deep residual networks

## 📝 Development Tips

1. Keep both terminal windows open (one for backend, one for frontend)
2. Backend must be running before starting the frontend
3. Use Python 3.12 for full compatibility
4. Monitor backend logs for debugging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 💡 Need Help?

- Check the [Issues](https://github.com/aryamangoenka/DND-neural/issues) page
- Ensure you're using Python 3.12
- Verify both backend and frontend are running on correct ports

**Happy building! 🎉**
