"""
start_state.py

Defines the StartState class, which represents the initial state of the application.
"""

from __future__ import annotations

from app.models.app_context import AppContext
from refit8_YOLO_camera.audit_platform_ws.app.controller.app_controller import AppController
from refit8_YOLO_camera.audit_platform_ws.app.states.state import State
from app.ui.test_ui_adapter import UIAdapter


class StartState(State):
    name = "START"

    def on_enter(self, context: AppContext, ui: UIAdapter) -> None:
        context.current_state = self.name
        ui.show_start_screen()

    def handle_event(
        self,
        event: str,
        context: AppContext,
        controller: "AppController",
    ) -> None:
        if event == "CREATE_NEW_AUDIT":
            controller.create_new_audit_and_go_to_location_select()
        else:
            controller.ui.show_error(
                f"Unsupported event '{event}' in state '{self.name}'"
            )