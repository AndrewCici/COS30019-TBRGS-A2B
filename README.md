# Traffic-Based Route Guidance System

A deep-learning-powered route guidance system that predicts real-time traffic
flow using SCATS sensor data and finds the top 5 optimal routes between
intersections using Yen's K-Shortest Paths algorithm. Routes are visualised
on an interactive OpenStreetMap map embedded inside a PyQt5 desktop GUI.

---

## Features

### Machine Learning Traffic Prediction
Three PyTorch model architectures are available for traffic flow forecasting:

| Model | Architecture |
|---|---|
| **LSTM** | 2-layer stacked LSTM + Dropout + Fully Connected head |
| **GRU** | 2-layer stacked GRU + Dropout + Fully Connected head |
| **Transformer** | TransformerEncoder with sinusoidal Positional Encoding + FC head |

All models are trained on 15-minute interval SCATS traffic volume data
(96 intervals per day), scaled with MinMaxScaler, and split 70 / 15 / 15
(train / val / test) while preserving temporal order.

### Yen's K-Shortest Paths Algorithm
- Finds the **5 shortest routes** between any two SCATS intersections
- Edge weights are travel times derived from ML-predicted traffic flow
  using the fundamental diagram formula:
  `flow = -1.4648375 × speed² + 93.75 × speed`
- Speed is capped at 60 km/h for flow ≤ 351 veh/hr; the quadratic is
  solved for the green (under-capacity) branch otherwise
- Includes a 0.5-minute intersection delay per node

### Interactive Folium Map
- Displays all 5 routes simultaneously on an **OpenStreetMap** base layer
- Each route is colour-coded: Red · Blue · Green · Purple · Orange
- Route 1 (fastest) is drawn thicker and fully opaque
- Clickable popups show path and estimated travel time for each route
- Embedded directly into the PyQt5 window via `QWebEngineView`
- Saved as `map.html` for standalone use

---

## Project Structure

```
Workspace/
├── data/
│   └── scats_clean.csv        # SCATS traffic volume data (96 intervals/day)
├── models/                     # Saved model weights (created by train.py)
│   ├── lstm_best.pth
│   ├── gru_best.pth
│   └── transformer_best.pth
├── plots/                      # Loss curve PNGs (created by train.py)
├── data_processing.py          # TrafficDataProcessor — load, scale, sequence, split
├── models.py                   # TrafficLSTM, TrafficGRU, TrafficTransformer
├── train.py                    # Training loop, early stopping, evaluation, plots
├── routing.py                  # Travel-time formula, TrafficGraph, Yen's algorithm
├── main_gui.py                 # PyQt5 GUI with folium map integration
├── graph.py                    # Base Graph / Node classes and file parser
├── frontier.py                 # Stack / Queue / Priority queue for search
├── search.py                   # DFS, BFS, A*, GBFS, UCS, IDA* search algorithms
├── test_system.py              # 15 pytest test cases
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10 or later
- `pip`

### 1. Clone or download the project

```bash
git clone <repository-url>
cd Workspace
```

### 2. Create and activate a virtual environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`** (create this file with the following content):
```
torch>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
PyQt5>=5.15.0
PyQtWebEngine>=5.15.0
folium>=0.15.0
pytest>=7.4.0
```

> **Note for macOS (Apple Silicon):** install PyTorch via the official channel:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

---

## How to Train the Models

Run the training script from the `Workspace/` directory:

```bash
python train.py
```

This will:

1. Load and preprocess `data/scats_clean.csv` using `TrafficDataProcessor`
2. Train all three models (LSTM, GRU, Transformer) for up to 50 epochs each
   with early stopping (patience = 10)
3. Save the best weights to `models/lstm_best.pth`, `models/gru_best.pth`,
   and `models/transformer_best.pth`
4. Save Train vs Validation loss curve PNGs to `plots/`
5. Print a final RMSE and MAE summary table to the console

Expected output (values vary by run):
```
[1/4] Loading and preprocessing data …
[TrafficDataProcessor] Series length : 9,408
  Train : 6,577  |  Validation : 1,409  |  Test : 1,410

============================================================
[2/4] Training LSTM …
  Epoch [001/050]  Train MSE: 0.012345  Val MSE: 0.014321
  ...
  Best weights saved → models/lstm_best.pth

Final Test Metrics Summary
============================================================
  Model           RMSE        MAE
  -------------------------------------
  LSTM          18.3421    12.7654
  GRU           17.9012    12.2341
  Transformer   19.1234    13.4512
```

---

## How to Run the GUI

```bash
python main_gui.py
```

### Usage

1. **Select an Origin SCATS site** — type a SCATS number from the list shown
   in the left panel (e.g. `2000`)
2. **Select a Destination SCATS site** — type a different SCATS number
   (e.g. `970`)
3. **Choose a model** — select LSTM, GRU, or Transformer from the dropdown
4. **Click "Find Top 5 Routes"** — the system will:
   - Load the saved model weights and predict traffic flow for each site
   - Run Yen's K-Shortest Paths algorithm with travel-time edge weights
   - Display the 5 routes and estimated times in the left panel
   - Render all 5 routes on the OpenStreetMap in the right panel

> **If model weights are not found** (i.e. `train.py` has not been run yet),
> the system automatically falls back to a mock flow of 300 veh/hr
> (free-flow, speed = 60 km/h) so routing remains functional.

---

## How to Run the Tests

```bash
pytest test_system.py -v
```

### Test Coverage (15 tests)

| Group | Tests | What is verified |
|---|---|---|
| **A — Travel-time maths** | 01–05 + 05b | Speed cap at 60 km/h, boundary at flow=351, quadratic branch at flow=800, monotone property, distance scaling, round-trip accuracy |
| **B — Pathfinding** | 06–10 | Up to 5 distinct paths, ascending sort, correct start/end nodes, congestion increases time, edge removal in `travel_time_adj` |
| **C — Edge cases** | 11–13 | Origin == Destination, no path between isolated nodes, graceful handling of negative/extreme flows |
| **D — Data pipeline** | 14 | DataLoader batch shape `(32, 12, 1)`, feature values in `[0, 1]` |
| **E — Model fallback** | 15 | `FileNotFoundError` raised with correct filename when `.pth` is missing |

---

## Algorithm Reference

### Flow-to-Speed Conversion (from SCATS Traffic Model)
```
flow = -1.4648375 × speed² + 93.75 × speed
```
- flow ≤ 351 veh/hr → speed = 60 km/h (speed limit)
- flow > 351 veh/hr → solve quadratic for the **higher root** (green /
  under-capacity branch):
  ```
  speed = (−B − √(B² + 4·A·flow)) / (2·A)
  ```
  where A = −1.4648375, B = 93.75

### Travel Time
```
travel_time (min) = (distance_km / speed_km_h) × 60 + 0.5
```
The 0.5-minute constant accounts for intersection delay at each SCATS node.

### Yen's K-Shortest Paths
1. Find the shortest path with Dijkstra → **A[0]**
2. For k = 1 … K−1:
   - For each spur node along **A[k−1]**:
     - Remove edges shared with all confirmed paths at the spur
     - Remove root-path nodes (except spur) to ensure loop-free spurs
     - Run Dijkstra from spur to destination
     - Push `root_path + spur_path` onto the candidate min-heap
   - Pop the best candidate → **A[k]**
3. Return all K paths sorted by total travel time

---

## Notes

- The current dataset (`scats_clean.csv`) contains one SCATS site (ID 970).
  The GUI uses a **10-node mock Melbourne network** for demonstration;
  replace `MOCK_SCATS` in `main_gui.py` with real multi-site SCATS data for
  production use.
- Model weights in `models/` are not committed to the repository; run
  `train.py` to generate them before launching the GUI.
- The `map.html` file is overwritten each time a route search is performed.
