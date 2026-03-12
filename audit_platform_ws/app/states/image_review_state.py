"""
image_review_state.py

Defines the ImageReviewState class, which represents the state where the user reviews the captured image of the item.
"""

from __future__ import annotations

from app.models.app_context import AppContext
from app.states.state import State
from app.ui.test_ui_adapter import UIAdapter


class ImageReviewState(State):
    name = "IMAGE_REVIEW"

    def on_enter(self, context: AppContext, ui: UIAdapter) -> None:
        context.current_state = self.name

        if context.last_captured_image_path is None:
            ui.show_error("No captured image to review.")
            return

        ui.show_image_review_screen(context.last_captured_image_path)

    def handle_event(
        self,
        event: str,
        context: AppContext,
        controller: "AppController",
    ) -> None:
        controller.ui.show_error(
            f"Unsupported event '{event}' in state '{self.name}'"
        )