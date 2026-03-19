"""
main.py

Entry point for the audit platform application. 
Initializes the application context, UI, and controller, then starts the Qt event loop.
"""

from __future__ import annotations

import sys
from pathlib import Path
import signal

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.controller.app_controller import AppController
from app.models.app_context import AppContext
from app.ui.qt_ui_adapter import AuditMainWindow, UIAdapter


def main() -> None:
    app = QApplication(sys.argv)

    repo_root = Path(__file__).resolve().parents[2]
    context = AppContext(repo_root=repo_root)

    window = AuditMainWindow()
    ui = UIAdapter(window)
    controller = AppController(context, ui)

    window.set_controller(controller)

    # Let the preview service used by states also feed the Qt preview
    if context.preview_service is not None:
        ui.set_preview_service(context.preview_service)

    controller.start()

    # Allow Ctrl+C from terminal to quit the Qt app cleanly
    signal.signal(signal.SIGINT, lambda *args: app.quit())

    # Let Python process signals while Qt event loop is running
    sigint_timer = QTimer()
    sigint_timer.start(200)
    sigint_timer.timeout.connect(lambda: None)
    signal.signal(signal.SIGINT, lambda *args: window.close())

    # Full-screen for Pi touchscreen kiosk use
    window.showFullScreen()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()