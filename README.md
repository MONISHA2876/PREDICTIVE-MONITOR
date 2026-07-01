# Predictive Monitoring System

A real-time AI-based sensor monitoring system with anomaly detection, built using Python, TensorFlow, PySide6, and pyqtgraph.

---

## Features

- **Three simulated sensors** — Temperature, Pressure, and Water Flow with realistic noise and fault injection
- **Two predictive models** — Moving Average (statistical baseline) and LSTM (deep learning), switchable at runtime
- **Real-time anomaly detection** with configurable WARNING and CRITICAL thresholds
- **Live scrolling graphs** showing actual vs predicted sensor values
- **Modern desktop dashboard** built with PySide6 and pyqtgraph
- **Modular architecture** allowing simulated sensors to be replaced with real hardware such as PLCs, Arduino, ESP32, or OpenModelica without changing the prediction or GUI layers

---

## Project Structure

```text
predictive_monitor/
├── main.py
├── requirements.txt
│
├── simulator/
│   ├── base_sensor.py
│   ├── temperature_sensor.py
│   ├── pressure_sensor.py
│   ├── flow_sensor.py
│   └── sensor_manager.py
│
├── data/
│   ├── generator.py
│   └── preprocessor.py
│
├── model/
│   ├── base_predictor.py
│   ├── moving_average.py
│   ├── lstm_model.py
│   └── anomaly_detector.py
│
├── gui/
│   ├── main_window.py
│   ├── sensor_card.py
│   ├── live_graph.py
│   ├── control_panel.py
│   ├── event_log.py
│   ├── status_bar.py
│   ├── training_dialog.py
│   └── styles.py
│
├── utils/
│   ├── config.py
│   ├── ring_buffer.py
│   ├── logger.py
│   └── worker_threads.py
│
├── data/
│   ├── dataset.csv
│   └── scaler.pkl
│
├── model/saved/
│   ├── lstm_model.keras
│   ├── best_weights.keras
│   └── training_history.json
│
├── logs/
│   └── predictive_monitor.log
│
└── docs/
    └── architecture.md
```

---

# Installation

## Requirements

- Python 3.10+
- pip

## Setup

```bash
# Clone the repository
git clone <repository-url>

cd predictive_monitor

# Create virtual environment
python -m venv .venv

# Activate

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** TensorFlow may take a few minutes to install because of its size.

---

# Running

```bash
python main.py
```

---

# Application Workflow

| Step | Action |
|------|--------|
| 1 | Start the sensor simulation |
| 2 | Generate a dataset |
| 3 | Select **Moving Average** |
| 4 | Start prediction |
| 5 | Observe live graphs and anomaly detection |
| 6 | Train the LSTM model |
| 7 | Switch to **LSTM** |
| 8 | Compare prediction quality and inference speed |

---

# Model Comparison

| Feature | Moving Average | LSTM |
|---------|----------------|------|
| Training Required | ❌ | ✅ |
| Inference Speed | Very Fast | Fast |
| Learns Non-linear Patterns | ❌ | ✅ |
| Prediction Accuracy | Moderate | High |
| Real-time Support | ✅ | ✅ |

---

# Anomaly Detection

The application classifies anomalies using the percentage prediction error.

| Error | Status |
|-------|--------|
| 0–5% | Normal |
| 5–15% | Warning |
| >15% | Critical |

Thresholds can be modified in:

```python
utils/config.py
```

```python
ANOMALY_WARNING_PCT = 5.0
ANOMALY_CRITICAL_PCT = 15.0
```

---

# Configuration

All configurable parameters are located in:

```text
utils/config.py
```

Example:

```python
DATASET_N_SAMPLES = 5000
TICK_INTERVAL_MS = 1000

SEQUENCE_LENGTH = 30

LSTM_UNITS = 64
LSTM_EPOCHS = 50

MA_WINDOW = 10

ANOMALY_WARNING_PCT = 5.0
ANOMALY_CRITICAL_PCT = 15.0

GRAPH_HISTORY_LEN = 120
```

---

# Replacing Simulated Sensors

The project uses an abstract `BaseSensor` class.

To connect real hardware:

1. Create a class inheriting from `BaseSensor`.
2. Implement the `read()` method.
3. Register the sensor inside `SensorManager`.

Example:

```python
class OpenModelicaAdapter(BaseSensor):
    ...

class ArduinoSerialSensor(BaseSensor):
    ...

class ESP32MQTTSensor(BaseSensor):
    ...
```

No modifications are required in the GUI or prediction modules.

---

# Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- PySide6
- pyqtgraph

---

# Future Improvements

- Real hardware integration (Arduino / ESP32 / PLC)
- MQTT support
- OpenModelica integration
- Multiple AI models (GRU, Transformer)
- Database storage
- Historical analytics dashboard
- Export anomaly reports
- Remote monitoring over a network

---

# License

This project is intended for learning, experimentation, and personal development.
