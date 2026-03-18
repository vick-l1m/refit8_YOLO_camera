"""
camera_capture_state.py

Defines the CameraCaptureState class, which represents the state where the user
captures an image of the item using the camera.

When the user enters this state, a live camera preview is launched.
When the user takes a picture or leaves the state, the preview is stopped.
"""

from __future__ import annotations

from app.models.app_context import AppContext
from app.services.item_capture_service import ItemCaptureService
from app.services.live_camera_preview_service import LiveCameraPreviewService
from app.states.state import State
from app.ui.test_ui_adapter import UIAdapter


class CameraCaptureState(State):
    name = "CAMERA_CAPTURE"

    def _start_preview(self, context: AppContext, ui: UIAdapter) -> None:
        try:
            if context.preview_service is None:
                context.preview_service = LiveCameraPreviewService()

            context.preview_service.start()
            ui.show_info("Live camera preview started.")
        except Exception as e:
            ui.show_error(f"Could not start live preview: {e}")

    def _stop_preview(self, context: AppContext) -> None:
        try:
            if context.preview_service is not None:
                context.preview_service.stop()
        except Exception:
            pass

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
                # Stop preview before still capture so the camera is free
                self._stop_preview(context)

                service = ItemCaptureService(context.repo_root)
                result = service.capture_base_item_image(context)

                context.last_captured_image_name = result["image_name"]
                context.last_captured_image_path = result["image_path"]
                context.current_item_dir = result["item_dir"]
                context.current_item_base_image_name = result["image_name"]

                controller.ui.show_info(f"Saved base image: {result['image_name']}")
                controller.transition_to("ADDITIONAL_IMAGES_MENU")

            except Exception as e:
                controller.ui.show_error(f"Failed to capture image: {e}")

                # Restart preview if capture failed and user is still in this state
                self._start_preview(context, controller.ui)

        elif event == "CANCEL_CAMERA_CAPTURE":
            self._stop_preview(context)
            controller.transition_to("ITEM_MENU")

        else:
            controller.ui.show_error(
                f"Unsupported event '{event}' in state '{self.name}'"
            )