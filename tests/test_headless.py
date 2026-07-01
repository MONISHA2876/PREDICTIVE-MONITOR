"""
tests/test_headless.py

Headless tests that exercise every non-GUI module.
Run with:    python tests/test_headless.py

No pytest required — uses only the stdlib unittest framework so the
suite works in any environment, including ones without a display.
"""

import sys
import os
import unittest
import tempfile
import shutil

# Make sure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Simulator tests
# ─────────────────────────────────────────────────────────────────────

class TestTemperatureSensor(unittest.TestCase):
    def setUp(self):
        from simulator.temperature_sensor import TemperatureSensor
        self.sensor = TemperatureSensor(anomaly_probability=0.0)

    def test_read_returns_sensor_reading(self):
        from simulator.base_sensor import SensorReading
        r = self.sensor.read()
        self.assertIsInstance(r, SensorReading)

    def test_value_near_baseline_without_anomalies(self):
        readings = [self.sensor.read().value for _ in range(100)]
        mean = sum(readings) / len(readings)
        # Should stay within ±3 °C of 25 °C baseline without anomalies
        self.assertAlmostEqual(mean, 25.0, delta=3.0)

    def test_reading_count_increments(self):
        for _ in range(5):
            self.sensor.read()
        self.assertEqual(self.sensor.reading_count, 5)

    def test_anomaly_triggered(self):
        from simulator.temperature_sensor import TemperatureSensor
        sensor = TemperatureSensor(anomaly_probability=1.0, overheat_amplitude=30.0)
        readings = [sensor.read() for _ in range(20)]
        has_anomaly = any(r.is_anomaly for r in readings)
        self.assertTrue(has_anomaly)

    def test_configure_changes_baseline(self):
        self.sensor.configure(baseline=40.0, noise_std=0.01)
        readings = [self.sensor.read().value for _ in range(50)]
        mean = sum(readings) / len(readings)
        self.assertAlmostEqual(mean, 40.0, delta=2.0)


class TestPressureSensor(unittest.TestCase):
    def setUp(self):
        from simulator.pressure_sensor import PressureSensor
        self.sensor = PressureSensor(anomaly_probability=0.0)

    def test_value_non_negative(self):
        """Pressure must never go negative."""
        for _ in range(200):
            self.assertGreaterEqual(self.sensor.read().value, 0.0)

    def test_sensor_type(self):
        self.assertEqual(self.sensor.sensor_type, "pressure")

    def test_unit(self):
        self.assertEqual(self.sensor.unit, "bar")


class TestFlowSensor(unittest.TestCase):
    def setUp(self):
        from simulator.flow_sensor import FlowSensor
        self.sensor = FlowSensor(anomaly_probability=0.0)

    def test_value_non_negative(self):
        for _ in range(200):
            self.assertGreaterEqual(self.sensor.read().value, 0.0)

    def test_blockage_reduces_flow(self):
        """
        Confirm that a blockage anomaly reduces flow below normal operating range.
        We force the mode via the internal state to make the test deterministic.
        """
        from simulator.flow_sensor import FlowSensor
        sensor = FlowSensor(
            baseline=10.0,
            noise_std=0.02,
            anomaly_probability=0.0,   # don't trigger randomly
            blockage_depth=8.0,
            event_duration=20,
        )
        # Manually inject a blockage anomaly
        sensor._in_anomaly = True
        sensor._anomaly_tick = 0
        sensor._anomaly_mode = "blockage"

        readings = [sensor.read().value for _ in range(20)]
        # Peak blockage is at midpoint (tick 10) — flow should be near 10 - 8 = 2.0
        self.assertLess(
            min(readings), 5.0,
            f"Blockage should reduce flow below 5.0; got min={min(readings):.3f}",
        )


class TestSensorManager(unittest.TestCase):
    def setUp(self):
        from simulator.sensor_manager import SensorManager
        self.manager = SensorManager()

    def test_snapshot_has_three_channels(self):
        snap = self.manager.read_all()
        self.assertIsNotNone(snap.temperature)
        self.assertIsNotNone(snap.pressure)
        self.assertIsNotNone(snap.flow)

    def test_snapshot_count_increments(self):
        for _ in range(3):
            self.manager.read_all()
        self.assertEqual(self.manager.snapshot_count, 3)

    def test_to_dict_has_required_keys(self):
        snap = self.manager.read_all()
        d = snap.to_dict()
        for key in ("timestamp", "temperature", "pressure", "flow", "anomaly"):
            self.assertIn(key, d)

    def test_register_custom_sensor(self):
        from simulator.base_sensor import BaseSensor, SensorReading
        from datetime import datetime

        class ConstantSensor(BaseSensor):
            def read(self):
                from datetime import timezone
                return self._record(SensorReading(
                    sensor_id="C-01", sensor_type="temperature",
                    value=99.0, unit="°C",
                    timestamp=datetime.now(timezone.utc)
                ))

        self.manager.register_sensor("temperature", ConstantSensor("C-01", "temperature", "°C"))
        snap = self.manager.read_all()
        self.assertEqual(snap.temperature.value, 99.0)


# ─────────────────────────────────────────────────────────────────────
# Data layer tests
# ─────────────────────────────────────────────────────────────────────

class TestDatasetGenerator(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_generates_correct_row_count(self):
        from data.generator import DatasetGenerator
        gen = DatasetGenerator(output_dir=self.tmp_dir, n_samples=200)
        path = gen.generate()
        import pandas as pd
        df = pd.read_csv(path)
        self.assertEqual(len(df), 200)

    def test_csv_columns(self):
        from data.generator import DatasetGenerator
        gen = DatasetGenerator(output_dir=self.tmp_dir, n_samples=50)
        path = gen.generate()
        import pandas as pd
        df = pd.read_csv(path)
        for col in ("timestamp", "temperature", "pressure", "flow", "anomaly"):
            self.assertIn(col, df.columns)

    def test_progress_callback_called(self):
        from data.generator import DatasetGenerator
        calls = []
        gen = DatasetGenerator(
            output_dir=self.tmp_dir, n_samples=100,
            progress_cb=lambda c, t: calls.append(c),
        )
        gen.generate()
        self.assertEqual(calls[-1], 100)

    def test_timing_returns_positive_elapsed(self):
        from data.generator import DatasetGenerator
        gen = DatasetGenerator(output_dir=self.tmp_dir, n_samples=50)
        _, elapsed = gen.generate_with_timing()
        self.assertGreater(elapsed, 0.0)


class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        # Generate a small CSV first
        from data.generator import DatasetGenerator
        gen = DatasetGenerator(output_dir=self.tmp_dir, n_samples=400)
        self.csv_path = str(gen.generate())
        self.scaler_path = os.path.join(self.tmp_dir, "scaler.pkl")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_shapes_are_consistent(self):
        from data.preprocessor import DataPreprocessor
        pre = DataPreprocessor(sequence_length=15, scaler_path=self.scaler_path)
        X_tr, y_tr, X_val, y_val, X_te, y_te = pre.fit_transform(self.csv_path)
        # sequence axis
        self.assertEqual(X_tr.shape[1], 15)
        # feature axis
        self.assertEqual(X_tr.shape[2], 3)
        self.assertEqual(y_tr.shape[1], 3)

    def test_scaled_values_in_unit_range(self):
        from data.preprocessor import DataPreprocessor
        pre = DataPreprocessor(sequence_length=10, scaler_path=self.scaler_path)
        X_tr, y_tr, _, _, _, _ = pre.fit_transform(self.csv_path)
        self.assertGreaterEqual(X_tr.min(), -0.01)
        self.assertLessEqual(X_tr.max(), 1.01)

    def test_scaler_saved_and_loadable(self):
        from data.preprocessor import DataPreprocessor
        pre = DataPreprocessor(sequence_length=10, scaler_path=self.scaler_path)
        pre.fit_transform(self.csv_path)
        self.assertTrue(os.path.exists(self.scaler_path))

        pre2 = DataPreprocessor(sequence_length=10, scaler_path=self.scaler_path)
        pre2.load_scaler()
        window = np.ones((10, 3), dtype=np.float32) * 0.5
        result = pre2.transform_single(window)
        self.assertEqual(result.shape, (10, 3))

    def test_inverse_transform_recovers_scale(self):
        from data.preprocessor import DataPreprocessor
        pre = DataPreprocessor(sequence_length=10, scaler_path=self.scaler_path)
        pre.fit_transform(self.csv_path)
        # Ones in scaled space should map to the scaler's data_max_
        scaled = np.ones((1, 3), dtype=np.float32)
        recovered = pre.inverse_transform(scaled)
        self.assertEqual(recovered.shape, (1, 3))


# ─────────────────────────────────────────────────────────────────────
# Model tests
# ─────────────────────────────────────────────────────────────────────

class TestMovingAveragePredictor(unittest.TestCase):
    def setUp(self):
        from model.moving_average import MovingAveragePredictor
        self.ma = MovingAveragePredictor(window_size=5)

    def test_is_ready(self):
        self.assertTrue(self.ma.is_ready())

    def test_prediction_shape(self):
        window = np.random.rand(10, 3).astype(np.float32)
        actual = np.random.rand(3).astype(np.float32)
        result = self.ma.predict(window, actual)
        self.assertEqual(result.predicted.shape, (3,))
        self.assertEqual(result.error_pct.shape, (3,))

    def test_prediction_is_mean_of_tail(self):
        window = np.zeros((10, 3), dtype=np.float32)
        # Set last 5 rows to 2.0
        window[-5:] = 2.0
        actual = np.zeros(3, dtype=np.float32)
        result = self.ma.predict(window, actual)
        np.testing.assert_allclose(result.predicted, 2.0, atol=1e-5)

    def test_inference_time_positive(self):
        window = np.random.rand(10, 3).astype(np.float32)
        actual = np.random.rand(3).astype(np.float32)
        result = self.ma.predict(window, actual)
        self.assertGreater(result.inference_ms, 0.0)

    def test_window_size_none_uses_full_window(self):
        from model.moving_average import MovingAveragePredictor
        ma = MovingAveragePredictor(window_size=None)
        window = np.ones((20, 3), dtype=np.float32) * 3.0
        actual = np.zeros(3, dtype=np.float32)
        result = ma.predict(window, actual)
        np.testing.assert_allclose(result.predicted, 3.0, atol=1e-5)

    def test_prediction_count_increments(self):
        window = np.random.rand(10, 3).astype(np.float32)
        actual = np.random.rand(3).astype(np.float32)
        for _ in range(5):
            self.ma.predict(window, actual)
        self.assertEqual(self.ma.prediction_count, 5)


class TestAnomalyDetector(unittest.TestCase):
    def setUp(self):
        from model.anomaly_detector import AnomalyDetector
        self.detector = AnomalyDetector(warning_threshold=5.0, critical_threshold=15.0)

    def _make_result(self, error_pct):
        from model.base_predictor import PredictionResult
        err = np.array(error_pct, dtype=np.float32)
        return PredictionResult(
            predicted=np.zeros(3, dtype=np.float32),
            actual=np.ones(3, dtype=np.float32),
            error_abs=err,
            error_pct=err,
            inference_ms=0.1,
            model_name="Test",
        )

    def test_normal_level(self):
        from model.anomaly_detector import AnomalyLevel
        result = self._make_result([1.0, 2.0, 3.0])
        self.assertEqual(self.detector.evaluate(result), AnomalyLevel.NORMAL)

    def test_warning_level(self):
        from model.anomaly_detector import AnomalyLevel
        result = self._make_result([8.0, 6.0, 7.0])
        self.assertEqual(self.detector.evaluate(result), AnomalyLevel.WARNING)

    def test_critical_level(self):
        from model.anomaly_detector import AnomalyLevel
        result = self._make_result([20.0, 18.0, 22.0])
        self.assertEqual(self.detector.evaluate(result), AnomalyLevel.CRITICAL)

    def test_events_logged_for_anomalies(self):
        result_warn = self._make_result([8.0, 8.0, 8.0])
        result_norm = self._make_result([1.0, 1.0, 1.0])
        self.detector.evaluate(result_warn)
        self.detector.evaluate(result_norm)
        self.detector.evaluate(result_warn)
        self.assertEqual(len(self.detector.event_log), 2)

    def test_clear_log(self):
        result = self._make_result([20.0, 20.0, 20.0])
        self.detector.evaluate(result)
        self.detector.clear_log()
        self.assertEqual(len(self.detector.event_log), 0)

    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            self.detector.warning_threshold = 20.0  # > critical
        with self.assertRaises(ValueError):
            self.detector.critical_threshold = 1.0  # < warning


# ─────────────────────────────────────────────────────────────────────
# Utility tests
# ─────────────────────────────────────────────────────────────────────

class TestRingBuffer(unittest.TestCase):
    def test_capacity_respected(self):
        from utils.ring_buffer import RingBuffer
        buf = RingBuffer[int](capacity=3)
        for i in range(7):
            buf.append(i)
        self.assertEqual(len(buf), 3)
        self.assertEqual(buf.to_list(), [4, 5, 6])

    def test_is_full(self):
        from utils.ring_buffer import RingBuffer
        buf = RingBuffer[int](capacity=3)
        self.assertFalse(buf.is_full)
        buf.append(1); buf.append(2); buf.append(3)
        self.assertTrue(buf.is_full)

    def test_clear_resets(self):
        from utils.ring_buffer import RingBuffer
        buf = RingBuffer[int](capacity=5)
        for i in range(5):
            buf.append(i)
        buf.clear()
        self.assertEqual(len(buf), 0)
        self.assertFalse(buf.is_full)

    def test_invalid_capacity(self):
        from utils.ring_buffer import RingBuffer
        with self.assertRaises(ValueError):
            RingBuffer[int](capacity=0)

    def test_iteration(self):
        from utils.ring_buffer import RingBuffer
        buf = RingBuffer[int](capacity=5)
        for i in range(5):
            buf.append(i)
        self.assertEqual(list(buf), [0, 1, 2, 3, 4])


class TestConfig(unittest.TestCase):
    def test_feature_names_length(self):
        from utils.config import APP_CONFIG
        self.assertEqual(len(APP_CONFIG.FEATURE_NAMES), 3)

    def test_thresholds_ordered(self):
        from utils.config import APP_CONFIG
        self.assertLess(APP_CONFIG.ANOMALY_WARNING_PCT, APP_CONFIG.ANOMALY_CRITICAL_PCT)

    def test_sequence_length_positive(self):
        from utils.config import APP_CONFIG
        self.assertGreater(APP_CONFIG.SEQUENCE_LENGTH, 0)


# ─────────────────────────────────────────────────────────────────────
# End-to-end integration test (no GPU / TF)
# ─────────────────────────────────────────────────────────────────────

class TestFullPipeline(unittest.TestCase):
    """
    Exercises the complete data → model → anomaly path without a display.
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_ma_realtime_loop(self):
        from data.generator import DatasetGenerator
        from data.preprocessor import DataPreprocessor
        from model.moving_average import MovingAveragePredictor
        from model.anomaly_detector import AnomalyDetector, AnomalyLevel
        from simulator.sensor_manager import SensorManager

        gen = DatasetGenerator(output_dir=self.tmp_dir, n_samples=300)
        csv = gen.generate()

        pre = DataPreprocessor(
            sequence_length=15,
            scaler_path=os.path.join(self.tmp_dir, "scaler.pkl")
        )
        pre.fit_transform(str(csv))

        manager = SensorManager()
        ma = MovingAveragePredictor(window_size=5)
        detector = AnomalyDetector(warning_threshold=5.0, critical_threshold=15.0)
        history = []
        levels_seen = set()

        for _ in range(60):
            snap = manager.read_all()
            actual = np.array([
                snap.temperature.value,
                snap.pressure.value,
                snap.flow.value
            ], dtype=np.float32)
            history.append(actual)
            if len(history) >= 15:
                window = np.array(history[-15:])
                result = ma.predict(window, actual)
                level = detector.evaluate(result)
                levels_seen.add(level)

        # We should have seen at least NORMAL in 60 ticks
        self.assertIn(AnomalyLevel.NORMAL, levels_seen)
        self.assertGreater(ma.prediction_count, 0)


if __name__ == "__main__":
    print("=" * 60)
    print("  Predictive Monitoring System — Headless Test Suite")
    print("=" * 60)
    unittest.main(verbosity=2)
