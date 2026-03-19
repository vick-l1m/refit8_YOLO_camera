#!/usr/bin/env python3
"""
camera_capture_state.py


Defines the CameraCaptureState class, which represents the state where the user
captures an image of the item using the camera.


When the user enters this state, a live camera preview is launched.
When the user takes a picture or leaves the state, the preview is stopped.


Date: 19/03/2026
Version: 1.1
Maintainer: Victor Lim - victor@polymaya.tech
"""


from __future__ import annotations


from pathlib import Path

from app.models.app_context import AppContext
from app.services.april_tag_service import AprilTagService
from app.services.item_capture_service import ItemCaptureService
from app.services.live_camera_preview_service import LiveCameraPreviewService
from app.states.state import State
from app.ui.qt_ui_adapter import UIAdapter


class CameraCaptureState(State):
    name = "CAMERA_CAPTURE"

    def _start_preview(self, context: AppContext, ui: UIAdapter) -> None:
        try:
            if context.preview_service is None:
                context.preview_service = LiveCameraPreviewService()

            context.preview_service.start()

            if hasattr(ui, "set_preview_service"):
                ui.set_preview_service(context.preview_service)

            ui.show_info("Live camera preview started.")
        except Exception as e:
            ui.show_error(f"Could not start live preview: {e}")

    def _stop_preview(self, context: AppContext) -> None:
        try:
            if context.preview_service is not None:
                context.preview_service.stop()
        except Exception:
            pass
        finally:
            context.preview_service = None

    def on_enter(self, context: AppContext, ui: UIAdapter) -> None:
        context.current_state = self.name
        ui.show_camera_capture_screen()
        self._start_preview(context, ui)

    def handle_event(
        self,
        event: str,
        context: AppContext,
        controller: "AppController",
    ) -> None:
        if event == "TAKE_PICTURE":
            try:
                service = ItemCaptureService(context.repo_root)
                result = service.capture_base_item_image(context)

                context.last_captured_image_name = result["image_name"]
                context.last_captured_image_path = result["image_path"]
                context.current_item_dir = result["item_dir"]
                context.current_item_base_image_name = result["image_name"]

                self._stop_preview(context)

                controller.ui.show_info(f"Saved base image: {result['image_name']}")

                apriltag_service = AprilTagService(context.repo_root)
                session = apriltag_service.detect_and_build_session_with_defaults(
                    image_path=result["image_path"]
                )

                if session is None:
                    context.apriltag_detected = False
                    context.apriltag_selected_tag = None
                    context.apriltag_session = None
                    context.apriltag_source_image_path = None
                    context.apriltag_annotated_image_path = None
                    context.apriltag_measurements_csv_path = None
                    controller.transition_to("ADDITIONAL_IMAGES_MENU")
                    return

                context.apriltag_detected = True
                context.apriltag_detection_family = apriltag_service.FAMILY_DEFAULT
                context.apriltag_tag_size_m = apriltag_service.TAG_SIZE_M_DEFAULT
                context.apriltag_intrinsics_path = apriltag_service.get_default_intrinsics_path()
                context.apriltag_source_image_path = result["image_path"]
                context.apriltag_selected_tag = session.tag_det
                context.apriltag_session = session

                image_stem = Path(result["image_name"]).stem
                context.apriltag_measurements_csv_path = (
                    result["item_dir"] / f"{image_stem}_data.csv"
                )

                controller.transition_to("APRIL_TAG_MEASURE")

            except Exception as e:
                controller.ui.show_error(f"Failed to capture image: {e}")

        elif event == "CANCEL_CAMERA_CAPTURE":
            self._stop_preview(context)
            controller.transition_to("ITEM_MENU")

        else:
            controller.ui.show_error(
                f"Unsupported event '{event}' in state '{self.name}'"
            )