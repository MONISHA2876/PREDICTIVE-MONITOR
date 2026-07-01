"""
main.py

Application entry point for the Predictive Monitoring System.

Run with:
    python main.py

Initialises logging, creates the Qt application, and shows the main window.
"""

import sys
import os

# Ensure the project root is on sys.path regardless of where the script is run from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from gui.main_window import MainWindow
from utils.logger import setup_logger


def main() -> int:
    """Create and run the Qt application. Returns the exit code."""
    # Set up logging before anything else
    log = setup_logger("predictive_monitor")
    log.info("Starting Predictive Monitoring System…")

    # Suppress TensorFlow INFO/WARNING banners before import
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    app = QApplication(sys.argv)
    app.setApplicationName("Predictive Monitoring System")
    app.setOrganizationName(" ")
    app.setApplicationVersion(" ")

    # High-DPI support (PySide6 handles this automatically on Qt6)

    window = MainWindow()
    window.show()

    log.info("GUI launched.")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
