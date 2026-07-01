"""
scripts/train_cli.py

Command-line interface for dataset generation and LSTM training.

Use this to pre-train the model before launching the GUI, or to
re-train with different hyper-parameters in a headless environment
(e.g. a remote server or CI runner).

Usage:
    python scripts/train_cli.py --samples 5000 --epochs 50 --seq-len 30
    python scripts/train_cli.py --samples 2000 --epochs 20   # quick test
"""

import argparse
import os
import sys
import time

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from data.generator import DatasetGenerator
from data.preprocessor import DataPreprocessor
from model.lstm_model import LSTMPredictor
from utils.config import APP_CONFIG
from utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the LSTM predictor from the command line."
    )
    p.add_argument(
        "--samples", type=int, default=APP_CONFIG.DATASET_N_SAMPLES,
        help="Number of simulated samples to generate (default: %(default)s)",
    )
    p.add_argument(
        "--epochs", type=int, default=APP_CONFIG.LSTM_EPOCHS,
        help="Maximum training epochs (default: %(default)s)",
    )
    p.add_argument(
        "--seq-len", type=int, default=APP_CONFIG.SEQUENCE_LENGTH,
        help="RNN look-back window length (default: %(default)s)",
    )
    p.add_argument(
        "--batch", type=int, default=APP_CONFIG.LSTM_BATCH_SIZE,
        help="Mini-batch size (default: %(default)s)",
    )
    p.add_argument(
        "--units", type=int, default=APP_CONFIG.LSTM_UNITS,
        help="LSTM units per layer (default: %(default)s)",
    )
    p.add_argument(
        "--skip-gen", action="store_true",
        help="Skip dataset generation (use existing CSV)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log = setup_logger("train_cli")

    # ── Step 1: Dataset generation ────────────────────────────────────
    csv_path = str(APP_CONFIG.DATASET_CSV)

    if not args.skip_gen:
        log.info("Generating %d samples → %s", args.samples, csv_path)
        t0 = time.perf_counter()

        def _progress(cur: int, tot: int) -> None:
            if cur % 500 == 0 or cur == tot:
                pct = int(cur / tot * 100)
                print(f"\r  Generating… {pct:3d}%  [{cur}/{tot}]", end="", flush=True)

        gen = DatasetGenerator(
            output_dir=str(APP_CONFIG.DATA_DIR),
            n_samples=args.samples,
            progress_cb=_progress,
        )
        gen.generate()
        print()
        log.info("Dataset ready in %.1f s", time.perf_counter() - t0)
    else:
        if not os.path.exists(csv_path):
            log.error("No dataset found at %s. Run without --skip-gen first.", csv_path)
            sys.exit(1)
        log.info("Using existing dataset: %s", csv_path)

    # ── Step 2: Preprocessing ─────────────────────────────────────────
    log.info("Preprocessing (seq_len=%d)…", args.seq_len)
    preprocessor = DataPreprocessor(
        sequence_length=args.seq_len,
        val_ratio=APP_CONFIG.VAL_RATIO,
        test_ratio=APP_CONFIG.TEST_RATIO,
        scaler_path=str(APP_CONFIG.SCALER_PKL),
    )
    X_tr, y_tr, X_val, y_val, X_te, y_te = preprocessor.fit_transform(csv_path)
    log.info(
        "Shapes  train=%s  val=%s  test=%s",
        X_tr.shape, X_val.shape, X_te.shape,
    )

    # ── Step 3: Build and train LSTM ──────────────────────────────────
    lstm = LSTMPredictor(
        sequence_length=args.seq_len,
        n_features=3,
        model_dir=str(APP_CONFIG.MODEL_DIR),
        units=args.units,
    )
    lstm.build_model()

    log.info("Training LSTM (max %d epochs, batch=%d)…", args.epochs, args.batch)

    def _epoch_cb(epoch: int, total: int, loss: float, val_loss: float) -> None:
        print(
            f"\r  Epoch {epoch:3d}/{total}  "
            f"loss={loss:.6f}  val_loss={val_loss:.6f}",
            end="", flush=True,
        )

    t0 = time.perf_counter()
    history = lstm.train(
        X_tr, y_tr, X_val, y_val,
        preprocessor=preprocessor,
        epochs=args.epochs,
        batch_size=args.batch,
        progress_cb=_epoch_cb,
    )
    print()
    elapsed = time.perf_counter() - t0
    actual_epochs = len(history["loss"])
    final_loss = history["val_loss"][-1]

    log.info(
        "Training complete: %d epochs in %.1f s  final val_loss=%.6f",
        actual_epochs, elapsed, final_loss,
    )

    model_path = lstm.save_model()
    log.info("Model saved → %s", model_path)

    # ── Step 4: Quick test inference ──────────────────────────────────
    import numpy as np
    from model.moving_average import MovingAveragePredictor
    from model.anomaly_detector import AnomalyDetector

    window = X_te[0]   # already scaled
    # Inverse-scale for raw-unit prediction
    raw_window = preprocessor.inverse_transform(
        window.reshape(-1, 3)
    ).reshape(args.seq_len, 3)
    actual_raw = preprocessor.inverse_transform(y_te[0:1])[0]

    result = lstm.predict(raw_window, actual_raw)
    log.info(
        "Sample prediction | "
        "T=%.2f→%.2f  P=%.3f→%.3f  F=%.3f→%.3f  (%.1f ms)",
        actual_raw[0], result.predicted[0],
        actual_raw[1], result.predicted[1],
        actual_raw[2], result.predicted[2],
        result.inference_ms,
    )
    log.info("Mean prediction error: %.2f%%", result.mean_error_pct)
    print("\n✓ CLI training complete. Launch the GUI and select 'Load Model' to use it.\n")


if __name__ == "__main__":
    main()
