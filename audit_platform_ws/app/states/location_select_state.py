"""
location_select_state.py

Defines the LocationSelectState class, which represents the state where the user selects a location for the audit.
"""
from __future__ import annotations

from app.models.app_context import AppContext
from app.services.location_service import LocationService
from app.states.state import State
# from app.ui.test_ui_adapter import UIAdapter
from app.ui.qt_ui_adapter import UIAdapter


class LocationSelectState(State):
    name = "LOCATION_SELECT"

    def on_enter(self, context: AppContext, ui: UIAdapter) -> None:
        context.current_state = self.name

        try:
            definitions_dir = context.repo_root / "audit_platform_ws" / "app" / "data" / "definitions"
            service = LocationService(definitions_dir)
            locations = service.load_locations()

            context.available_locations = locations
            ui.show_location_screen(locations)

        except Exception as e:
            ui.show_error(str(e))

    def handle_event(
        self,
        event: str,
        context: AppContext,
        controller: "AppController",
    ) -> None:
        if event.startswith("SELECT_LOCATION:"):
            location_id = event.split(":", 1)[1].strip()

            selected = next(
                (loc for loc in context.available_locations if loc["id"] == location_id),
                None,
            )

            if selected is None:
                controller.ui.show_error(f"Unknown location id: {location_id}")
                return

            context.current_location = selected

            if context.current_audit is not None:
                context.current_audit.current_location_id = selected["id"]
                context.current_audit.current_location_name = selected["name"]

            controller.ui.show_info(
                f"Selected location: {selected['name']} ({selected['id']})"
            )

            controller.transition_to("ASSET_CATEGORY_SELECT")

        elif event == "BACK_TO_START":
            controller.transition_to("START")

        else:
            controller.ui.show_error(
                f"Unsupported event '{event}' in state '{self.name}'"
            )