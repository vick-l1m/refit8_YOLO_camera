#!/usr/bin/env python3
# Metadata:
# File: audit_platform_ws/app/ui/april_tag_measure_screen.py
# Purpose: AprilTag measurement page widget with canvas and action buttons.
# Date: 19/03/2026
# Version: 1.0
# Maintainer: Victor Lim - victor@polymaya.tech

"""
april_tag_measure_screen.py

Provides a reusable AprilTag measure screen for the audit app UI.
"""
from __future__ import annotations


from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from app.ui.widgets.measurement_canvas import MeasurementCanvas
from src.april_tag_measure_ws.measurement_session import MeasurementSession


class AprilTagMeasureScreen(QWidget):
    def __init__(self, controller=None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._session: Optional[MeasurementSession] = None
        self._image_path: Optional[Path] = None

        self._build_ui()

    def set_controller(self, controller) -> None:
        self.controller = controller

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)

        left = QVBoxLayout()
        right = QVBoxLayout()

        self.header_label = QLabel("AprilTag Measure")
        self.header_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.details_label = QLabel(
            "Tag size: 0.05m\nSearching for April tag: tag25h9"
        )
        self.details_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.details_label.setWordWrap(True)

        self.canvas = MeasurementCanvas()
        self.canvas.line_added.connect(self._on_line_added)
        self.canvas.measurement_failed.connect(self._on_measurement_failed)

        self.status_label = QLabel("Draw a line on the image to measure it.")
        self.status_label.setWordWrap(True)

        left.addWidget(self.header_label)
        left.addWidget(self.canvas, stretch=1)
        left.addWidget(self.status_label)

        right.addStretch()

        self.measure_button = QPushButton("Measure")
        self.measure_button.setMinimumHeight(80)
        self.measure_button.clicked.connect(self._dispatch_measure)

        self.clear_button = QPushButton("Clear Lines")
        self.clear_button.setMinimumHeight(60)
        self.clear_button.clicked.connect(self._clear_lines)

        self.back_button = QPushButton("Back")
        self.back_button.setMinimumHeight(60)
        self.back_button.clicked.connect(self._dispatch_back)

        right.addWidget(self.measure_button)
        right.addWidget(self.clear_button)
        right.addWidget(self.back_button)
        right.addStretch()

        layout.addLayout(left, stretch=3)
        layout.addLayout(right, stretch=1)
    
    def set_session(
        self,
        session: MeasurementSession,
        image_path: Optional[Path] = None,
    ) -> None:
        self._session = session
        self._image_path = image_path

        title = "AprilTag Measure"
        if image_path is not None:
            title = f"Measure {image_path.stem}"
        self.header_label.setText(title)

        self.details_label.setText(
            f"Tag size: {session.tag_size_m:.2f}m\n"
            f"Searching for April tag: {session.tag_family}"
        )

        self.status_label.setText("Draw a line on the image to measure it.")
        self.canvas.set_session(session)


    def _clear_lines(self) -> None:
        self.canvas.clear_measurements()
        self.status_label.setText("All measurement lines cleared.")

    def _dispatch_measure(self) -> None:
        if self.controller is not None:
            self.controller.dispatch("MEASURE_APRIL_TAG")

    def _dispatch_back(self) -> None:
        if self.controller is not None:
            self.controller.dispatch("BACK_FROM_APRIL_TAG_MEASURE")

    def _on_line_added(self, length_m: float, mode: str) -> None:
        self.status_label.setText(
            f"Added line: {length_m:.4f} m using {mode} mode."
        )

    def _on_measurement_failed(self, message: str) -> None:
        self.status_label.setText(f"Measurement failed: {message}")