from __future__ import annotations

from app.models.app_context import AppContext
from app.states.state import State
from app.ui.test_ui_adapter import UIAdapter


class ItemMenuState(State):
    name = "ITEM_MENU"

    def on_enter(self, context: AppContext, ui: UIAdapter) -> None:
        context.current_state = self.name
        ui.show_item_menu_screen()

    def handle_event(
        self,
        event: str,
        context: AppContext,
        controller: "AppController",
    ) -> None:
        if event == "GO_TO_CAMERA_CAPTURE":
            controller.transition_to("CAMERA_CAPTURE")
        elif event == "BACK_TO_LOCATION_SELECT":
            controller.transition_to("LOCATION_SELECT")
        elif event == "BACK_TO_ASSET_CATEGORY_SELECT":
            controller.transition_to("ASSET_CATEGORY_SELECT")
        else:
            controller.ui.show_error(
                f"Unsupported event '{event}' in state '{self.name}'"
            )