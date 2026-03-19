"""
qt_ui_adapter.py

Defines the UIAdapter class, which serves as a bridge between the Qt-based UI (AuditMainWindow) 
and the application logic (AppController and states).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QSizePolicy,
    QFrame,
)


class AuditMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Audit App")
        self.resize(1024, 600)

        self.controller = None
        self.preview_service = None

        self._build_ui()
        self._build_timers()
        self._apply_styles()

    def set_controller(self, controller) -> None:
        self.controller = controller

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        self.state_title = QLabel("STATE")
        self.state_title.setAlignment(Qt.AlignCenter)
        self.state_title.setObjectName("stateTitle")
        root.addWidget(self.state_title)

        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setWordWrap(True)
        self.info_label.setObjectName("infoLabel")
        root.addWidget(self.info_label)

        self.stack = QStackedWidget()
        root.addWidget(self.stack, stretch=1)

        self.start_page = self._build_start_page()
        self.location_page = self._build_location_page()
        self.asset_category_page = self._build_asset_category_page()
        self.item_menu_page = self._build_item_menu_page()
        self.camera_capture_page = self._build_camera_capture_page()
        self.additional_images_page = self._build_additional_images_page()
        self.camera_capture_additional_page = self._build_camera_capture_additional_page()

        self.stack.addWidget(self.start_page)
        self.stack.addWidget(self.location_page)
        self.stack.addWidget(self.asset_category_page)
        self.stack.addWidget(self.item_menu_page)
        self.stack.addWidget(self.camera_capture_page)
        self.stack.addWidget(self.additional_images_page)
        self.stack.addWidget(self.camera_capture_additional_page)

    def _build_timers(self) -> None:
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._update_live_preview)
        self.preview_timer.start(50)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f4f6f8;
            }
            QLabel#stateTitle {
                font-size: 28px;
                font-weight: bold;
                color: #1f2937;
                padding: 8px;
            }
            QLabel#infoLabel {
                background: #e5eef7;
                color: #1f2937;
                border-radius: 8px;
                padding: 8px;
                min-height: 28px;
            }
            QPushButton {
                background: #2563eb;
                color: white;
                border: none;
                border-radius: 12px;
                padding: 14px;
                font-size: 18px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #1d4ed8;
            }
            QPushButton:pressed {
                background: #1e40af;
            }
            QPushButton.secondary {
                background: #6b7280;
            }
            QPushButton.danger {
                background: #dc2626;
            }
            QListWidget {
                background: white;
                border: 1px solid #d1d5db;
                border-radius: 12px;
                font-size: 18px;
                padding: 6px;
            }
            QListWidget::item {
                padding: 12px;
                margin: 4px;
                border-radius: 8px;
            }
            QListWidget::item:selected {
                background: #dbeafe;
                color: #111827;
            }
            QFrame#previewFrame {
                background: #111827;
                border-radius: 14px;
            }
            QLabel#previewLabel {
                color: #e5e7eb;
                font-size: 18px;
            }
            """
        )

    def _make_preview_label(self) -> QLabel:
        frame = QLabel("Preview")
        frame.setAlignment(Qt.AlignCenter)
        frame.setObjectName("previewLabel")
        frame.setMinimumSize(400, 300)
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return frame

    def _make_preview_container(self, label: QLabel) -> QFrame:
        frame = QFrame()
        frame.setObjectName("previewFrame")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(label)
        return frame

    def _build_start_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addStretch()

        self.btn_create_audit = QPushButton("Create New Audit")
        self.btn_create_audit.setMinimumHeight(80)
        self.btn_create_audit.clicked.connect(
            lambda: self._dispatch("CREATE_NEW_AUDIT")
        )

        layout.addWidget(self.btn_create_audit, alignment=Qt.AlignCenter)
        layout.addStretch()

        bottom = QHBoxLayout()
        self.btn_edit_audit = QPushButton("Edit Audit")
        self.btn_edit_audit.setProperty("class", "secondary")
        self.btn_edit_audit.clicked.connect(
            lambda: self.show_info("Edit Audit is not implemented yet.")
        )

        self.btn_quit_start = QPushButton("Quit")
        self.btn_quit_start.setProperty("class", "danger")
        self.btn_quit_start.clicked.connect(self.close)

        bottom.addWidget(self.btn_edit_audit)
        bottom.addWidget(self.btn_quit_start)
        bottom.addStretch()

        layout.addLayout(bottom)
        return page

    def _build_location_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)

        left = QVBoxLayout()
        self.btn_back_location = QPushButton("Back")
        self.btn_back_location.clicked.connect(lambda: self._dispatch("BACK_TO_START"))
        self.btn_quit_location = QPushButton("Quit")
        self.btn_quit_location.clicked.connect(self.close)

        left.addWidget(self.btn_back_location, alignment=Qt.AlignTop | Qt.AlignLeft)
        left.addStretch()
        left.addWidget(self.btn_quit_location, alignment=Qt.AlignLeft | Qt.AlignBottom)

        right = QVBoxLayout()
        right.addWidget(QLabel("Location List"))
        self.location_list = QListWidget()
        self.location_list.itemClicked.connect(self._handle_location_selected)
        right.addWidget(self.location_list)

        layout.addLayout(left, stretch=3)
        layout.addLayout(right, stretch=1)

        return page

    def _build_asset_category_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)

        left = QVBoxLayout()
        self.btn_back_asset = QPushButton("Back")
        self.btn_back_asset.clicked.connect(
            lambda: self._dispatch("BACK_TO_LOCATION_SELECT")
        )
        self.btn_quit_asset = QPushButton("Quit")
        self.btn_quit_asset.clicked.connect(self.close)

        left.addWidget(self.btn_back_asset, alignment=Qt.AlignTop | Qt.AlignLeft)
        left.addStretch()
        left.addWidget(self.btn_quit_asset, alignment=Qt.AlignLeft | Qt.AlignBottom)

        right = QVBoxLayout()
        right.addWidget(QLabel("Asset Category List"))
        self.asset_category_list = QListWidget()
        self.asset_category_list.itemClicked.connect(self._handle_asset_category_selected)
        right.addWidget(self.asset_category_list)

        layout.addLayout(left, stretch=3)
        layout.addLayout(right, stretch=1)

        return page

    def _build_item_menu_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        top = QVBoxLayout()
        self.btn_back_item_location = QPushButton("Back to Location Select")
        self.btn_back_item_location.clicked.connect(
            lambda: self._dispatch("BACK_TO_LOCATION_SELECT")
        )
        self.btn_back_item_asset = QPushButton("Back to Asset Category Select")
        self.btn_back_item_asset.clicked.connect(
            lambda: self._dispatch("BACK_TO_ASSET_CATEGORY_SELECT")
        )

        top.addWidget(self.btn_back_item_location, alignment=Qt.AlignLeft)
        top.addWidget(self.btn_back_item_asset, alignment=Qt.AlignLeft)
        layout.addLayout(top)

        layout.addStretch()

        self.btn_add_new_item = QPushButton("Add New Item")
        self.btn_add_new_item.setMinimumHeight(80)
        self.btn_add_new_item.clicked.connect(
            lambda: self._dispatch("GO_TO_CAMERA_CAPTURE")
        )
        layout.addWidget(self.btn_add_new_item, alignment=Qt.AlignCenter)

        layout.addStretch()

        self.btn_quit_item = QPushButton("Quit")
        self.btn_quit_item.clicked.connect(self.close)
        layout.addWidget(self.btn_quit_item, alignment=Qt.AlignLeft)

        return page

    def _build_camera_capture_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)

        self.camera_preview_label = self._make_preview_label()
        preview_container = self._make_preview_container(self.camera_preview_label)

        right = QVBoxLayout()
        right.addStretch()

        self.btn_take_picture = QPushButton("📷 Take Picture")
        self.btn_take_picture.setMinimumHeight(80)
        self.btn_take_picture.clicked.connect(lambda: self._dispatch("TAKE_PICTURE"))

        self.btn_back_camera = QPushButton("Back")
        self.btn_back_camera.clicked.connect(
            lambda: self._dispatch("CANCEL_CAMERA_CAPTURE")
        )

        right.addWidget(self.btn_take_picture)
        right.addWidget(self.btn_back_camera)
        right.addStretch()

        self.btn_quit_camera = QPushButton("Quit")
        self.btn_quit_camera.clicked.connect(self.close)
        right.addWidget(self.btn_quit_camera)

        layout.addWidget(preview_container, stretch=1)
        layout.addLayout(right, stretch=1)

        return page

    def _build_additional_images_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)

        self.additional_image_preview_label = self._make_preview_label()
        preview_container = self._make_preview_container(self.additional_image_preview_label)

        right = QVBoxLayout()
        right.addStretch()

        self.btn_take_additional = QPushButton("➕ Take Additional Images")
        self.btn_take_additional.setMinimumHeight(80)
        self.btn_take_additional.clicked.connect(
            lambda: self._dispatch("TAKE_ADDITIONAL_IMAGE")
        )

        self.btn_back_additional = QPushButton("Back")
        self.btn_back_additional.clicked.connect(
            lambda: self._dispatch("END_ITEM_IMAGES")
        )

        right.addWidget(self.btn_take_additional)
        right.addWidget(self.btn_back_additional)
        right.addStretch()

        self.btn_quit_additional = QPushButton("Quit")
        self.btn_quit_additional.clicked.connect(self.close)
        right.addWidget(self.btn_quit_additional)

        layout.addWidget(preview_container, stretch=1)
        layout.addLayout(right, stretch=1)

        return page

    def _build_camera_capture_additional_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)

        self.additional_camera_preview_label = self._make_preview_label()
        preview_container = self._make_preview_container(self.additional_camera_preview_label)

        right = QVBoxLayout()
        right.addStretch()

        self.btn_take_picture_additional = QPushButton("📷 Take Picture")
        self.btn_take_picture_additional.setMinimumHeight(80)
        self.btn_take_picture_additional.clicked.connect(
            lambda: self._dispatch("TAKE_PICTURE")
        )

        self.btn_back_camera_additional = QPushButton("Back")
        self.btn_back_camera_additional.clicked.connect(
            lambda: self._dispatch("CANCEL_CAMERA_CAPTURE")
        )

        right.addWidget(self.btn_take_picture_additional)
        right.addWidget(self.btn_back_camera_additional)
        right.addStretch()

        self.btn_quit_camera_additional = QPushButton("Quit")
        self.btn_quit_camera_additional.clicked.connect(self.close)
        right.addWidget(self.btn_quit_camera_additional)

        layout.addWidget(preview_container, stretch=1)
        layout.addLayout(right, stretch=1)

        return page

    def _dispatch(self, event: str) -> None:
        if self.controller is not None:
            self.controller.dispatch(event)

    def _handle_location_selected(self, item: QListWidgetItem) -> None:
        location_id = item.data(Qt.UserRole)
        self._dispatch(f"SELECT_LOCATION:{location_id}")

    def _handle_asset_category_selected(self, item: QListWidgetItem) -> None:
        category_id = item.data(Qt.UserRole)
        self._dispatch(f"SELECT_ASSET_CATEGORY:{category_id}")

    def set_state_title(self, title: str) -> None:
        self.state_title.setText(title)

    def set_page(self, page_name: str) -> None:
        mapping = {
            "START": self.start_page,
            "LOCATION_SELECT": self.location_page,
            "ASSET_CATEGORY_SELECT": self.asset_category_page,
            "ITEM_MENU": self.item_menu_page,
            "CAMERA_CAPTURE": self.camera_capture_page,
            "ADDITIONAL_IMAGES_MENU": self.additional_images_page,
            "CAMERA_CAPTURE_ADDITIONAL": self.camera_capture_additional_page,
        }
        page = mapping.get(page_name)
        if page is not None:
            self.stack.setCurrentWidget(page)

    def set_locations(self, locations: List[Dict[str, Any]]) -> None:
        self.location_list.clear()
        for loc in locations:
            item = QListWidgetItem(loc["name"])
            item.setData(Qt.UserRole, loc["id"])
            self.location_list.addItem(item)

    def set_asset_categories(self, categories: List[Dict[str, Any]]) -> None:
        self.asset_category_list.clear()
        for cat in categories:
            item = QListWidgetItem(cat["name"])
            item.setData(Qt.UserRole, cat["id"])
            self.asset_category_list.addItem(item)

    def show_info_banner(self, message: str) -> None:
        self.info_label.setText(message)

    def show_error_dialog(self, message: str) -> None:
        self.info_label.setText(message)
        QMessageBox.critical(self, "Error", message)

    def show_image_preview(self, image_path: Path, target: str = "additional_menu") -> None:
        pix = QPixmap(str(image_path))
        if pix.isNull():
            self.info_label.setText(f"Could not load image: {image_path}")
            return

        if target == "additional_menu":
            label = self.additional_image_preview_label
        else:
            label = self.additional_camera_preview_label

        scaled = pix.scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # refresh image scaling after resize
        current_pixmap = self.additional_image_preview_label.pixmap()
        if current_pixmap is not None and not current_pixmap.isNull():
            self.additional_image_preview_label.setPixmap(
                current_pixmap.scaled(
                    self.additional_image_preview_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

        current_pixmap2 = self.additional_camera_preview_label.pixmap()
        if current_pixmap2 is not None and not current_pixmap2.isNull():
            self.additional_camera_preview_label.setPixmap(
                current_pixmap2.scaled(
                    self.additional_camera_preview_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
            
    def _update_live_preview(self) -> None:
        if self.preview_service is None:
            return

        frame = self.preview_service.get_latest_frame()
        if frame is None:
            return

        h, w, ch = frame.shape
        bytes_per_line = ch * w

        image = QImage(
            frame.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888,
        )

        pix = QPixmap.fromImage(image)

        current_widget = self.stack.currentWidget()
        target_label: Optional[QLabel] = None

        if current_widget == self.camera_capture_page:
            target_label = self.camera_preview_label
        elif current_widget == self.camera_capture_additional_page:
            target_label = self.additional_camera_preview_label

        if target_label is not None:
            scaled = pix.scaled(
                target_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            target_label.setPixmap(scaled)


class UIAdapter:
    def __init__(self, window: AuditMainWindow) -> None:
        self.window = window

    def set_preview_service(self, preview_service) -> None:
        self.window.preview_service = preview_service

    def show_start_screen(self) -> None:
        self.window.set_state_title("START")
        self.window.set_page("START")

    def show_location_screen(self, locations: List[Dict[str, Any]]) -> None:
        self.window.set_state_title("LOCATION SELECT")
        self.window.set_locations(locations)
        self.window.set_page("LOCATION_SELECT")

    def show_asset_category_screen(self, categories: List[Dict[str, Any]]) -> None:
        self.window.set_state_title("ASSET CATEGORY SELECT")
        self.window.set_asset_categories(categories)
        self.window.set_page("ASSET_CATEGORY_SELECT")

    def show_item_menu_screen(self) -> None:
        self.window.set_state_title("ITEM MENU")
        self.window.set_page("ITEM_MENU")

    def show_camera_capture_screen(self) -> None:
        self.window.set_state_title("CAMERA CAPTURE")
        self.window.set_page("CAMERA_CAPTURE")
        self.window.show_info_banner("Live camera preview running.")

    def show_item_image_menu_screen(self, image_path: Path) -> None:
        self.window.set_state_title("ADDITIONAL IMAGES MENU")
        self.window.set_page("ADDITIONAL_IMAGES_MENU")
        self.window.show_image_preview(image_path, target="additional_menu")

    def show_camera_capture_additional_screen(self) -> None:
        self.window.set_state_title("CAMERA CAPTURE ADDITIONAL")
        self.window.set_page("CAMERA_CAPTURE_ADDITIONAL")
        self.window.show_info_banner("Take more images of the same object.")

    def show_image_review_screen(self, image_path: Path) -> None:
        self.window.set_state_title("IMAGE REVIEW")
        self.window.set_page("ADDITIONAL_IMAGES_MENU")
        self.window.show_image_preview(image_path, target="additional_menu")

    def show_error(self, message: str) -> None:
        self.window.show_error_dialog(message)

    def show_info(self, message: str) -> None:
        self.window.show_info_banner(message)