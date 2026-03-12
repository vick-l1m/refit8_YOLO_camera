from __future__ import annotations

from app.models.app_context import AppContext
from app.services.item_capture_service import ItemCaptureService
from app.states.state import State
from app.ui.test_ui_adapter import UIAdapter


class CameraCaptureAdditionalState(State):
    name = "CAMERA_CAPTURE_ADDITIONAL"

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
            if context.current_item_dir is None:
                controller.ui.show_error("No current item folder exists.")
                controller.transition_to("ITEM_MENU")
                return

            try:
                service = ItemCaptureService(context.repo_root)
                result = service.capture_additional_item_image(context.current_item_dir)

                context.last_captured_image_name = result["image_name"]
                context.last_captured_image_path = result["image_path"]

                controller.ui.show_info(f"Saved additional image: {result['image_name']}")
                controller.transition_to("ADDITIONAL_IMAGES_MENU")

            except Exception as e:
                controller.ui.show_error(f"Failed to capture additional image: {e}")

        elif event == "CANCEL_CAMERA_CAPTURE":
            controller.transition_to("ADDITIONAL_IMAGES_MENU")

        else:
            controller.ui.show_error(
                f"Unsupported event '{event}' in state '{self.name}'"
            )