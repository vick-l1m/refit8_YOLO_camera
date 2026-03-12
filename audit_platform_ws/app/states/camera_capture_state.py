"""
camera_capture_state.py

Defines the CameraCaptureState class, which represents the state where the user captures an image of the item using the camera.
It updates the AppContext with the captured image information and transitions to the next state for item image menu.
"""

from __future__ import annotations

from app.models.app_context import AppContext
from app.services.item_capture_service import ItemCaptureService
from app.states.state import State
from app.ui.test_ui_adapter import UIAdapter


class CameraCaptureState(State):
    name = "CAMERA_CAPTURE"

    def on_enter(self, context: AppContext, ui: UIAdapter) -> None:
        context.current_state = self.name
        ui.show_camera_capture_screen()

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

                controller.ui.show_info(f"Saved base image: {result['image_name']}")
                controller.transition_to("ADDITIONAL_IMAGES_MENU")

            except Exception as e:
                controller.ui.show_error(f"Failed to capture image: {e}")

        elif event == "CANCEL_CAMERA_CAPTURE":
            controller.transition_to("ITEM_MENU")

        else:
            controller.ui.show_error(
                f"Unsupported event '{event}' in state '{self.name}'"
            )