"""
camera_capture_additional_state.py

Defines the CameraCaptureAdditionalState class, which represents the state where the user captures additional images of the item using the camera after taking the initial picture.

When the user enters this state, a live camera preview is launched along with a preview of the original image.
When the user takes a picture or leaves the state, the previews are stopped.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from app.models.app_context import AppContext
from app.services.item_capture_service import ItemCaptureService
from app.services.live_camera_preview_service import LiveCameraPreviewService
from app.states.state import State
# from app.ui.test_ui_adapter import UIAdapter


class CameraCaptureAdditionalState(State):
    name = "CAMERA_CAPTURE_ADDITIONAL"

    # def _start_preview(self, context: AppContext, ui: UIAdapter) -> None:
    #     try:
    #         if context.preview_service is None:
    #             context.preview_service = LiveCameraPreviewService()

    #         context.preview_service.start()
    #         ui.show_info("Live camera preview started.")
    #     except Exception as e:
    #         ui.show_error(f"Could not start live preview: {e}")

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

    def _show_base_image_preview(self, context: AppContext, ui: UIAdapter) -> None:
        try:
            if shutil.which("feh") is None:
                ui.show_error("feh is not installed. Install it with: sudo apt install feh")
                return

            if not context.current_item_dir or not context.current_item_base_image_name:
                ui.show_error("No base image information found for this item.")
                return

            image_path = Path(context.current_item_dir) / context.current_item_base_image_name

            if not image_path.exists():
                ui.show_error(f"Base image not found: {image_path}")
                return

            # Close any existing preview first
            self._close_base_image_preview(context)

            context.base_image_preview_process = subprocess.Popen(
                ["feh", str(image_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            ui.show_info(f"Showing base image preview: {image_path.name}")

        except Exception as e:
            ui.show_error(f"Could not show base image preview: {e}")

    def _close_base_image_preview(self, context: AppContext) -> None:
        try:
            proc = context.base_image_preview_process
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        except Exception:
            pass
        finally:
            context.base_image_preview_process = None

    def _cleanup(self, context: AppContext) -> None:
        self._stop_preview(context)
        self._close_base_image_preview(context)

    def on_enter(self, context: AppContext, ui: UIAdapter) -> None:
        context.current_state = self.name
        ui.show_camera_capture_additional_screen()
        self._show_base_image_preview(context, ui)
        self._start_preview(context, ui)

    def handle_event(
        self,
        event: str,
        context: AppContext,
        controller: "AppController",
    ) -> None:
        if event == "TAKE_PICTURE":
            try:
                self._cleanup(context)

                service = ItemCaptureService(context.repo_root)
                result = service.capture_additional_item_image(context)

                context.last_captured_image_name = result["image_name"]
                context.last_captured_image_path = result["image_path"]

                controller.ui.show_info(
                    f"Saved additional image: {result['image_name']}"
                )
                controller.transition_to("ADDITIONAL_IMAGES_MENU")

            except Exception as e:
                controller.ui.show_error(f"Failed to capture image: {e}")
                self._show_base_image_preview(context, controller.ui)
                self._start_preview(context, controller.ui)

        elif event == "CANCEL_CAMERA_CAPTURE":
            self._cleanup(context)
            controller.transition_to("ADDITIONAL_IMAGES_MENU")

        else:
            controller.ui.show_error(
                f"Unsupported event '{event}' in state '{self.name}'"
            )