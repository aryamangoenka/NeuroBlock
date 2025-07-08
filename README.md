# NeuroBlock - Drag-and-Drop Neural Network Builder

A comprehensive web-based tool that allows users to visually design, train, and export custom neural networks using an intuitive drag-and-drop interface. This project consists of three main components: a Python Flask backend, a React frontend application, and a Next.js landing page.

## 🚀 Quick Start Guide

Follow these step-by-step instructions to clone and run the project on your local machine.

### Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python 3.10 or higher** (but less than 3.13) - [Download Python](https://www.python.org/downloads/)
- **Node.js 18.x or later** - [Download Node.js](https://nodejs.org/)
- **npm** (comes with Node.js)
- **Git** - [Download Git](https://git-scm.com/)

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/aryamangoenka/DND-neural.git

# Navigate to the project directory
cd DND-neural/DND-Neural-Network
```

### Step 2: Backend Setup (Python Flask)

The backend handles neural network training, model management, and API endpoints.

#### Option A: Using Poetry (Recommended)

```bash
# Install Poetry if you don't have it
# For macOS/Linux/WSL:
curl -sSL https://install.python-poetry.org | python3 -

# For Windows PowerShell:
# (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Install backend dependencies
poetry install

# Make the run script executable (macOS/Linux only)
chmod +x ./run_backend.sh

# Start the backend server
./run_backend.sh
```

#### Option B: Using pip and venv

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables and start the server
export PYTHONPATH="$PYTHONPATH:$(pwd)"  # Linux/macOS
# set PYTHONPATH=%PYTHONPATH%;%cd%  # Windows CMD
# $env:PYTHONPATH="$env:PYTHONPATH;$(Get-Location)"  # Windows PowerShell

# Start the backend
python -m backend.main
```

**The backend will start on:** `http://localhost:5000`

### Step 3: Frontend Setup (React + Vite)

The frontend provides the drag-and-drop neural network builder interface.

```bash
# Open a new terminal window/tab
# Navigate to the frontend directory
cd frontend

# Install frontend dependencies
npm install

# Start the development server
npm run dev
```

**The frontend will start on:** `http://localhost:5173`

### Step 4: Landing Page Setup (Next.js)

The landing page provides marketing content and project information.

```bash
# Open another new terminal window/tab
# Navigate to the landing page directory
cd landing

# Install landing page dependencies
npm install

# Start the development server
npm run dev
```

**The landing page will start on:** `http://localhost:3000`

## 🖥️ Accessing the Application

After completing the setup:

1. **Landing Page:** Visit `http://localhost:3000` - Marketing page with project overview
2. **Main Application:** Visit `http://localhost:5173` - The neural network builder interface
3. **Backend API:** Running on `http://localhost:5000` - Handles all backend operations

## 🛠️ Project Structure

```
DND-Neural-Network/
├── backend/                 # Flask backend (Python)
│   ├── api/                # API endpoints
│   ├── models/             # Neural network models
│   ├── datasets/           # Dataset management
│   ├── training/           # Training logic
│   └── main.py            # Flask application entry point
├── frontend/               # React frontend (TypeScript + Vite)
│   ├── src/               # Source code
│   ├── public/            # Static assets
│   └── package.json       # Frontend dependencies
├── landing/                # Next.js landing page
│   ├── src/               # Source code
│   ├── public/            # Static assets
│   └── package.json       # Landing page dependencies
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Poetry configuration
└── README.md              # This file
```

## 🎯 Key Features

- **Drag-and-Drop Interface:** Build neural networks visually by connecting layers
- **Real-Time Training:** Monitor training progress with live graphs and metrics
- **Multiple Export Formats:** Export as Python script, Jupyter notebook, TensorFlow SavedModel, or Keras model
- **Pre-built Templates:** Start with ready-made templates for common tasks
- **Dataset Support:** Built-in support for popular datasets (MNIST, CIFAR-10, Iris, etc.)
- **ResNet Support:** Create deep residual networks with ResNet blocks

## 📝 Usage Guide

1. **Start with the Landing Page** (`localhost:3000`) to understand the project
2. **Access the Main App** (`localhost:5173`) to build neural networks
3. **Select a Dataset:** Choose from predefined datasets or upload your own
4. **Build Your Network:** Drag and drop layers to create your architecture
5. **Configure Parameters:** Click layers to adjust settings
6. **Train the Model:** Monitor real-time training progress
7. **Export Your Model:** Download in your preferred format

## 🧪 Running Tests

```bash
# Backend tests
./run_tests.sh

# Or with pip:
cd backend
python run_tests.py
```

## 🔧 Troubleshooting

### Common Issues

**Backend won't start:**

```bash
# Check if port 5000 is in use
lsof -i :5000  # macOS/Linux
netstat -ano | findstr :5000  # Windows

# Kill the process if needed
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows
```

**Frontend won't start:**

```bash
# Clear npm cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Permission denied on scripts:**

```bash
# Make scripts executable (macOS/Linux)
chmod +x ./run_backend.sh
chmod +x ./run_tests.sh
```

**Python module not found:**

```bash
# Ensure PYTHONPATH is set correctly
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

### Port Configuration

If you need to change the default ports, update these files:

- **Backend:** Modify `PORT` in `backend/config.py`
- **Frontend:** Change port in `frontend/vite.config.ts`
- **Landing:** Use `npm run dev -- -p 3001` for custom port

## 🚀 Development Tips

1. **Keep all three terminals open** - one for each component
2. **Backend must be running** before frontend can function properly
3. **Check browser console** for frontend errors
4. **Monitor backend logs** for API debugging
5. **Use Poetry** for better Python dependency management

## 📦 Building for Production

### Backend

```bash
poetry install --no-dev
poetry run gunicorn -w 4 -k eventlet -b 0.0.0.0:8080 backend.main:app
```

### Frontend

```bash
cd frontend
npm run build
npm run preview
```

### Landing Page

```bash
cd landing
npm run build
npm start
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `./run_tests.sh`
5. Commit changes: `git commit -m "Add feature"`
6. Push to branch: `git push origin feature-name`
7. Submit a pull request

## 🙏 Acknowledgments

- **React Flow** for drag-and-drop functionality
- **TensorFlow/Keras** for neural network capabilities
- **Flask-SocketIO** for real-time communication
- **Prof. Cooper Sigrist** for mentorship and guidance

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 💡 Need Help?

- Check the [Issues](https://github.com/aryamangoenka/DND-neural/issues) page
- Review the troubleshooting section above
- Ensure all prerequisites are properly installed
- Verify all three components are running on their respective ports

**Happy building! 🎉**
