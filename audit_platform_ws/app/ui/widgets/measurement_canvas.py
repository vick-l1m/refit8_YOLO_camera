#!/usr/bin/env python3
# Metadata:
# File: audit_platform_ws/app/ui/widgets/measurement_canvas.py
# Purpose: Interactive PySide canvas for AprilTag line measurement.
# Date: 19/03/2026
# Version: 1.0
# Maintainer: Victor Lim - victor@polymaya.tech

"""
measurement_canvas.py

Provides a QWidget that displays a MeasurementSession image and allows the user
to click-and-drag to add measurement lines.
"""
from __future__ import annotations


from typing import Optional, Tuple

import cv2
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget, QVBoxLayout

from src.april_tag_measure_ws.measurement_session import MeasurementSession


class MeasurementCanvas(QWidget):
    line_added = Signal(float, str)
    measurement_failed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._session: Optional[MeasurementSession] = None
        self._pixmap: Optional[QPixmap] = None
        self._scaled_pixmap: Optional[QPixmap] = None

        self._dragging = False
        self._drag_start_img: Optional[Tuple[int, int]] = None
        self._drag_current_img: Optional[Tuple[int, int]] = None

        self._image_size: Optional[Tuple[int, int]] = None
        self._draw_rect = None

        self.label = QLabel("Measurement canvas")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label.setMouseTracking(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)

        self.setMouseTracking(True)

    def set_session(self, session: MeasurementSession) -> None:
        self._session = session
        self._dragging = False
        self._drag_start_img = None
        self._drag_current_img = None
        self._refresh_display()

    def clear_measurements(self) -> None:
        if self._session is None:
            return
        self._session.clear_lines()
        self._refresh_display()

    def refresh(self) -> None:
        self._refresh_display()

    def _refresh_display(self) -> None:
        if self._session is None:
            self.label.setText("No AprilTag measurement session loaded.")
            self.label.setPixmap(QPixmap())
            return

        preview_line = None
        if (
            self._dragging
            and self._drag_start_img is not None
            and self._drag_current_img is not None
        ):
            preview_line = (self._drag_start_img, self._drag_current_img)

        img_bgr = self._session.render(preview_line=preview_line)
        self._image_size = (img_bgr.shape[1], img_bgr.shape[0])

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w

        qimg = QImage(
            img_rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888,
        ).copy()

        self._pixmap = QPixmap.fromImage(qimg)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return

        scaled = self._pixmap.scaled(
            self.label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._scaled_pixmap = scaled
        self.label.setPixmap(scaled)
        self.label.setText("")

        x = (self.label.width() - scaled.width()) // 2
        y = (self.label.height() - scaled.height()) // 2
        self._draw_rect = (x, y, scaled.width(), scaled.height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _map_widget_point_to_image(self, pos: QPoint) -> Optional[Tuple[int, int]]:
        if self._draw_rect is None or self._image_size is None:
            return None

        x0, y0, draw_w, draw_h = self._draw_rect
        px = pos.x()
        py = pos.y()

        if px < x0 or py < y0 or px >= x0 + draw_w or py >= y0 + draw_h:
            return None

        img_w, img_h = self._image_size
        rel_x = (px - x0) / draw_w
        rel_y = (py - y0) / draw_h

        ix = int(rel_x * img_w)
        iy = int(rel_y * img_h)

        ix = max(0, min(ix, img_w - 1))
        iy = max(0, min(iy, img_h - 1))
        return (ix, iy)

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)

        if event.button() != Qt.LeftButton or self._session is None:
            return

        mapped = self._map_widget_point_to_image(event.position().toPoint())
        if mapped is None:
            return

        self._dragging = True
        self._drag_start_img = mapped
        self._drag_current_img = mapped
        self._refresh_display()

    def mouseMoveEvent(self, event) -> None:
        super().mouseMoveEvent(event)

        if not self._dragging or self._session is None:
            return

        mapped = self._map_widget_point_to_image(event.position().toPoint())
        if mapped is None:
            return

        self._drag_current_img = mapped
        self._refresh_display()

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)

        if event.button() != Qt.LeftButton or not self._dragging or self._session is None:
            return

        mapped = self._map_widget_point_to_image(event.position().toPoint())
        if mapped is None:
            mapped = self._drag_current_img

        self._dragging = False

        if self._drag_start_img is None or mapped is None:
            self._drag_start_img = None
            self._drag_current_img = None
            self._refresh_display()
            return

        try:
            line = self._session.add_line(self._drag_start_img, mapped)
            self.line_added.emit(line.length_m, line.mode)
        except Exception as e:
            self.measurement_failed.emit(str(e))

        self._drag_start_img = None
        self._drag_current_img = None
        self._refresh_display()