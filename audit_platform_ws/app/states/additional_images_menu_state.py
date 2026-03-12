from __future__ import annotations

from app.models.app_context import AppContext
from app.states.state import State
from app.ui.test_ui_adapter import UIAdapter


class AdditionalImagesMenuState(State):
    name = "ADDITIONAL_IMAGES_MENU"

    def on_enter(self, context: AppContext, ui: UIAdapter) -> None:
        context.current_state = self.name

        if context.last_captured_image_path is None:
            ui.show_error("No captured image available.")
            return

        ui.show_item_image_menu_screen(context.last_captured_image_path)

    def handle_event(
        self,
        event: str,
        context: AppContext,
        controller: "AppController",
    ) -> None:
        if event == "TAKE_ADDITIONAL_IMAGE":
            controller.transition_to("CAMERA_CAPTURE_ADDITIONAL")
        elif event == "END_ITEM_IMAGES":
            controller.transition_to("ITEM_MENU")
        else:
            controller.ui.show_error(
                f"Unsupported event '{event}' in state '{self.name}'"
            )