# Predictive Monitoring System
## IUAC Research Prototype — v1.0.0

A real-time AI-based sensor monitoring system with anomaly detection, built for a particle accelerator laboratory research internship.

---

## Features

- **Three simulated sensors** — Temperature, Pressure, Water Flow — with realistic noise and fault injection
- **Two predictive models** — Moving Average (statistical baseline) and LSTM (deep learning) — switchable at runtime
- **Real-time anomaly detection** — configurable WARNING / CRITICAL thresholds
- **Live scrolling graphs** — actual vs predicted overlay per sensor
- **Professional industrial dashboard** — built with PySide6 + pyqtgraph
- **Modular architecture** — the simulator layer can be replaced with OpenModelica, PLC, Arduino, or ESP32 with no changes to the model or GUI layers

---

## Project Structure

```
predictive_monitor/
├── main.py                   # Application entry point
├── requirements.txt
│
├── simulator/                # Sensor abstraction layer
│   ├── base_sensor.py        # Abstract BaseSensor + SensorReading
│   ├── temperature_sensor.py # ~25 °C + overheat / rapid-drop faults
│   ├── pressure_sensor.py    # ~2.5 bar + spike / drop faults
│   ├── flow_sensor.py        # ~10 L/min + blockage / leakage faults
│   └── sensor_manager.py     # Orchestrates all sensors → SensorSnapshot
│
├── data/
│   ├── generator.py          # Runs simulator → saves CSV
│   └── preprocessor.py       # MinMaxScaler, sequence windows, train/val/test
│
├── model/
│   ├── base_predictor.py     # Abstract BasePredictor + PredictionResult
│   ├── moving_average.py     # MA baseline (no training required)
│   ├── lstm_model.py         # LSTM build / train / save / load / predict
│   └── anomaly_detector.py   # Threshold classifier + event log
│
├── gui/
│   ├── main_window.py        # Top-level window, QTimer loop, orchestration
│   ├── sensor_card.py        # Per-sensor display card
│   ├── live_graph.py         # pyqtgraph scrolling actual-vs-predicted plot
│   ├── control_panel.py      # Left sidebar with all action buttons
│   ├── event_log.py          # Colour-coded anomaly event list
│   ├── status_bar.py         # Top bar with clock and system status
│   ├── training_dialog.py    # Progress dialog during LSTM training
│   └── styles.py             # QSS stylesheet + colour constants
│
├── utils/
│   ├── config.py             # Central configuration (APP_CONFIG singleton)
│   ├── ring_buffer.py        # Thread-safe fixed-length circular buffer
│   ├── logger.py             # Rotating file + console logger
│   └── worker_threads.py     # DatasetWorker + TrainWorker (QThread)
│
├── data/                     # Generated at runtime
│   ├── dataset.csv
│   └── scaler.pkl
│
├── model/saved/              # Generated at runtime
│   ├── lstm_model.keras
│   ├── best_weights.keras
│   └── training_history.json
│
├── logs/                     # Generated at runtime
│   └── predictive_monitor.log
│
└── docs/
    └── architecture.md
```

---

## Setup

### Requirements

- Python 3.10 or later
- pip

### Installation

```bash
# 1. Clone / download the project
cd predictive_monitor

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** TensorFlow may take a few minutes to download (~500 MB).  
> On Apple Silicon, use `pip install tensorflow-macos` instead.

---

## Running the Application

```bash
python main.py
```

---

## Workflow

Follow these steps in the GUI for a complete run:

| Step | Action | Notes |
|------|--------|-------|
| 1 | Click **▶ Start Sim** | Sensor data begins streaming |
| 2 | Click **⚙ Generate Data** | Creates `data/dataset.csv` (5 000 rows) |
| 3 | Select **Moving Average** in the combo box | Start predicting with the baseline immediately |
| 4 | Click **⚡ Start Pred** | Real-time prediction begins |
| 5 | Observe cards, graphs, and event log | Anomalies appear within ~30 seconds |
| 6 | Click **🧠 Train LSTM** | Background training (early stopping, ~50 epochs) |
| 7 | Switch combo box to **LSTM** | Model auto-selected after training |
| 8 | Compare inference times and error rates | Research comparison available |

---

## Model Comparison

| Metric | Moving Average | LSTM |
|--------|---------------|------|
| Training required | ✗ | ✓ |
| Inference time | < 0.1 ms | ~5–30 ms |
| Handles non-linear patterns | ✗ | ✓ |
| Anomaly sensitivity | Moderate | High |

Switch between models live using the **ACTIVE MODEL** combo box in the control panel.

---

## Anomaly Thresholds

| Score (mean % error) | Level | Card colour |
|----------------------|-------|-------------|
| 0 – 5 % | NORMAL | Green |
| 5 – 15 % | WARNING | Amber |
| > 15 % | CRITICAL | Red |

Thresholds are configured in `utils/config.py`:

```python
ANOMALY_WARNING_PCT  = 5.0
ANOMALY_CRITICAL_PCT = 15.0
```

---

## Replacing the Simulator

The `BaseSensor` abstract class in `simulator/base_sensor.py` defines the interface.  
To plug in a real data source:

1. Subclass `BaseSensor` and implement `read()` to return a `SensorReading`.
2. Register your sensor with `SensorManager.register_sensor(key, your_sensor)`.
3. No changes required anywhere else.

Example adapters to write later:

```python
class OpenModelicaAdapter(BaseSensor): ...
class ArduinoSerialSensor(BaseSensor): ...
class ESP32MQTTSensor(BaseSensor): ...
```

---

## Configuration

All tuneable parameters are in `utils/config.py`:

```python
DATASET_N_SAMPLES  = 5_000      # rows in generated CSV
TICK_INTERVAL_MS   = 1_000      # GUI refresh rate
SEQUENCE_LENGTH    = 30         # LSTM look-back window
LSTM_UNITS         = 64
LSTM_EPOCHS        = 50
MA_WINDOW          = 10
ANOMALY_WARNING_PCT  = 5.0
ANOMALY_CRITICAL_PCT = 15.0
GRAPH_HISTORY_LEN  = 120        # visible ticks on graph
```

---

## License

Research prototype for IUAC internship. Not for production use.
