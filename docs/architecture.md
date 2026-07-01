# Architecture Documentation

## System Overview

The system is divided into four independent layers. Data flows strictly downward; no layer imports from a higher layer.

```
┌─────────────────────────────────────────────────────────────────┐
│                        GUI Layer                                 │
│  MainWindow ── SensorCard × 3 ── LiveGraph × 3 ── EventLog      │
│            └── ControlPanel ── TopStatusBar                      │
│                       ▲                                          │
│                  QTimer (1 Hz)                                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │ SensorSnapshot
┌─────────────────────────▼───────────────────────────────────────┐
│                     Model Layer                                  │
│  BasePredictor ◄── MovingAveragePredictor                        │
│               ◄── LSTMPredictor                                  │
│  AnomalyDetector ── PredictionResult → AnomalyLevel              │
└─────────────────────────┬───────────────────────────────────────┘
                          │ raw np.ndarray (window)
┌─────────────────────────▼───────────────────────────────────────┐
│                     Data Layer                                   │
│  DataPreprocessor ── MinMaxScaler ── sequence windows            │
│  DatasetGenerator ── SensorManager.read_all() × N               │
└─────────────────────────┬───────────────────────────────────────┘
                          │ SensorReading
┌─────────────────────────▼───────────────────────────────────────┐
│                   Simulator Layer                                 │
│  BaseSensor (abstract)                                           │
│    ├── TemperatureSensor  (~25 °C, overheat, rapid-drop)         │
│    ├── PressureSensor     (~2.5 bar, spike, drop)                │
│    └── FlowSensor         (~10 L/min, blockage, leakage)         │
│  SensorManager  →  SensorSnapshot                                │
└─────────────────────────────────────────────────────────────────┘
         ↑ replaceable with OpenModelica / PLC / Arduino
```

---

## Real-Time Data Flow (per tick)

```
QTimer.timeout (every 1 s)
        │
        ▼
SensorManager.read_all()
        │  SensorSnapshot{temp, pres, flow}
        ▼
RingBuffer.append(actual_values)   ← maintains SEQUENCE_LENGTH history
        │
        ▼
BasePredictor.predict(window, actual)
        │  PredictionResult{predicted, error_abs, error_pct, inference_ms}
        ▼
AnomalyDetector.evaluate(result)
        │  AnomalyLevel{NORMAL | WARNING | CRITICAL}
        ▼
┌───────┼──────────────────────────────────┐
│       ▼                                  │
│  SensorCard.update_values()         LiveGraph.append()
│       ▼                                  │
│  TopStatusBar.set_status()          EventLog.add_anomaly()
└──────────────────────────────────────────┘
```

---

## Class Diagram (abbreviated)

```
BaseSensor (ABC)
├── sensor_id: str
├── read() → SensorReading          [abstract]
├── reset()
└── configure(**kwargs)

    TemperatureSensor(BaseSensor)
    ├── _baseline, _noise_std, _anomaly_prob
    ├── _drift_phase, _drift_amplitude
    └── _in_anomaly, _anomaly_mode, _anomaly_tick

    PressureSensor(BaseSensor)   [same structure]
    FlowSensor(BaseSensor)       [same structure]

SensorManager
├── _sensors: Dict[str, BaseSensor]
├── read_all() → SensorSnapshot
├── register_sensor(key, sensor)
└── configure_sensor(key, **kwargs)

─────────────────────────────────────────

BasePredictor (ABC)
├── predict(window, actual) → PredictionResult   [abstract]
├── is_ready() → bool                            [abstract]
└── _make_result(predicted, actual, ms) → PredictionResult

    MovingAveragePredictor(BasePredictor)
    └── window_size: int

    LSTMPredictor(BasePredictor)
    ├── build_model()
    ├── train(X_tr, y_tr, X_val, y_val, preprocessor) → history
    ├── save_model(path)
    └── load_model(path)

AnomalyDetector
├── warning_threshold: float
├── critical_threshold: float
├── evaluate(result) → AnomalyLevel
├── score(result) → float
└── event_log: List[AnomalyEvent]

─────────────────────────────────────────

MainWindow (QMainWindow)
├── _sensor_manager: SensorManager
├── _anomaly_detector: AnomalyDetector
├── _ma_predictor: MovingAveragePredictor
├── _lstm_predictor: LSTMPredictor
├── _active_predictor: BasePredictor
├── _history: deque[np.ndarray]     ← ring buffer for LSTM window
├── _timer: QTimer
├── _on_tick()                      ← real-time loop body
└── sub-widgets: TopStatusBar, ControlPanel,
                 SensorCard×3, LiveGraph×3, EventLog
```

---

## Training Pipeline

```
DatasetGenerator.generate()
    └── SensorManager.read_all() × 5000  →  dataset.csv

DataPreprocessor.fit_transform(csv_path)
    ├── MinMaxScaler.fit(train_portion)
    ├── _make_sequences(scaled_data)     → (X, y) arrays
    └── saves scaler.pkl

LSTMPredictor.train(X_tr, y_tr, X_val, y_val, ...)
    ├── EarlyStopping (patience=8)
    ├── ModelCheckpoint (best_weights.keras)
    └── saves lstm_model.keras + training_history.json
```

---

## Future Integration Points

| Component | How to Replace |
|-----------|---------------|
| Simulator | Subclass `BaseSensor`, register with `SensorManager` |
| Model     | Subclass `BasePredictor`, implement `predict()` and `is_ready()` |
| GUI theme | Edit `gui/styles.py` palette constants |
| Thresholds | Edit `utils/config.py` or call `AnomalyDetector.warning_threshold = x` |
| Tick rate  | Edit `APP_CONFIG.TICK_INTERVAL_MS` |
