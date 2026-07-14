# NeuroBlock — Drag-and-Drop Neural Network Builder

NeuroBlock lets you **design, train, and export neural networks visually** — no code required. Drag layers onto a canvas, connect them, pick a dataset, and watch training happen live with real-time loss/accuracy charts. When you're happy with the model, export it as runnable Python, a Jupyter notebook, a Keras model file, or a PyTorch script.

Built for teaching: it is used in the UMass CICS Turing summer program to introduce high-school students to deep learning.

## Features

- **Visual model builder** — drag-and-drop layers (Dense, Convolution, MaxPooling, Global Average Pooling, Flatten, Dropout, Attention, ResNet blocks) on a ReactFlow canvas
- **Built-in datasets** — Iris, MNIST, CIFAR-10, California Housing, Breast Cancer
- **Custom datasets** — upload your own CSV/Excel files or image archives; each browser session gets isolated storage (safe for classrooms)
- **Live training** — real-time per-epoch metrics streamed over WebSockets, with charts and a confusion matrix
- **Templates** — ready-made architectures to start from
- **Export anywhere** — Python script (`py`), Jupyter notebook (`ipynb`), Keras (`keras`), TensorFlow SavedModel (`savedmodel`), PyTorch script (`pytorch`)
- **In-browser prediction** — test trained image models by drawing digits or uploading images

## Architecture

```
DND-Neural-Network/
├── backend/     Flask + Flask-SocketIO + TensorFlow API server (port 8080)
│   ├── api/         REST routes + WebSocket training events
│   ├── datasets/    Built-in dataset loaders
│   ├── models/      Graph → Keras model builder
│   ├── export/      Code/model export generators
│   ├── training/    Real-time training callbacks
│   └── utils/       Sessions, logging, image processing
├── frontend/    React 18 + TypeScript + Vite app (port 5173)
├── landing/     Next.js marketing/landing page (optional, separate app)
└── docs/        Additional guides (deployment, dataset API, testing)
```

## Quick Start

### Prerequisites

- **Python 3.10 – 3.12** (3.13+ not yet supported by pinned TensorFlow)
- **Node.js 18+** (latest LTS recommended)
- **[Poetry](https://python-poetry.org/docs/#installation)** for Python dependency management

### 1. Clone

```bash
git clone https://github.com/aryamangoenka/DND-Neural-Network.git
cd DND-Neural-Network
```

### 2. Backend (terminal 1)

```bash
poetry env use python3.12   # or python3.10 / python3.11
poetry install
./run_backend.sh
```

The API server starts on **http://localhost:8080**.

> **Note:** the first startup can take 1–3 minutes while TensorFlow initializes —
> the server is ready when you see the SocketIO/werkzeug "Running on ..." log line.
> Verify with: `curl http://localhost:8080/api/health`

### 3. Frontend (terminal 2)

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** — you should see the builder canvas.

## Using the App

1. **Choose a dataset** in the right panel (start with Iris — it trains in seconds)
2. **Drag layers** from the left palette onto the canvas
3. **Connect them**: Input → hidden layers → Output (drag between node handles)
4. **Set hyperparameters** (batch size, epochs, optimizer, loss, learning rate)
5. **Save Model**, then hit **Train** and watch live metrics
6. **Export** your model from the Export menu in the format you want

## Configuration

| Setting | Default | How to change |
|---|---|---|
| Backend port | `8080` | `PORT` env var |
| Backend config | `development` | `FLASK_CONFIG` env var (`development`/`production`/`testing`) |
| Frontend → backend URL (dev) | `http://localhost:8080` | `VITE_BACKEND_URL` in `frontend/.env` (see `.env.example`) |

## Tests

```bash
./run_tests.sh          # backend test suite (pytest)
cd frontend && npm run lint
```

## Troubleshooting

**`poetry: command not found`** — add Poetry to your PATH:
`export PATH="$HOME/.local/bin:$PATH"`

**Wrong Python version** — force the env before installing:
`poetry env use python3.12 && poetry install`

**Port 8080 already in use** — find and stop the other process:
`lsof -i :8080` then `kill <PID>`, or run with `PORT=8081 ./run_backend.sh`
(and set `VITE_BACKEND_URL=http://localhost:8081` in `frontend/.env`)

**Backend seems stuck on startup** — normal for the first 1–3 minutes (TensorFlow
loading). If it's still silent after that, check the terminal for a traceback.

**Frontend can't reach backend** — the backend must be running *before* you load
the frontend page; check `curl http://localhost:8080/api/health`.

## Deployment

NeuroBlock deploys as a single container (Flask serves the app, API, and
WebSockets from one origin). See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
for the AWS EC2 quickstart and the Cloud Run alternative.

## License

MIT — see [LICENSE](LICENSE).
