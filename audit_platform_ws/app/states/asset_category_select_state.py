"""
asset_category_select_state.py

Defines the AssetCategorySelectState class, which represents the state where the user selects an asset category for the audit.
"""
from __future__ import annotations

from app.models.app_context import AppContext
from app.services.asset_category_service import AssetCategoryService
from app.states.state import State
# from app.ui.test_ui_adapter import UIAdapter
from app.ui.qt_ui_adapter import UIAdapter


class AssetCategorySelectState(State):
    name = "ASSET_CATEGORY_SELECT"

    def on_enter(self, context: AppContext, ui: UIAdapter) -> None:
        context.current_state = self.name

        try:
            definitions_dir = context.repo_root / "audit_platform_ws" / "app" / "data" / "definitions"
            service = AssetCategoryService(definitions_dir)
            categories = service.load_asset_categories()

            context.available_asset_categories = categories
            ui.show_asset_category_screen(categories)

        except Exception as e:
            ui.show_error(str(e))

    def handle_event(
        self,
        event: str,
        context: AppContext,
        controller: "AppController",
    ) -> None:
        if event.startswith("SELECT_ASSET_CATEGORY:"):
            if context.selected_location is None:
                controller.ui.show_error(
                    "You must select a location before selecting an asset category."
                )
                controller.transition_to("LOCATION_SELECT")
                return

            category_id = event.split(":", 1)[1].strip()

            selected = next(
                (
                    category
                    for category in context.available_asset_categories
                    if category["id"] == category_id
                ),
                None,
            )

            if selected is None:
                controller.ui.show_error(f"Unknown asset category id: {category_id}")
                return

            context.current_asset_category = selected

            if context.current_audit is not None:
                context.current_audit.current_asset_category_id = selected["id"]
                context.current_audit.current_asset_category_name = selected["name"]

            controller.ui.show_info(
                f"Selected asset category: {selected['name']} ({selected['id']})"
            )

            controller.transition_to("ITEM_MENU")

        elif event == "BACK_TO_LOCATION_SELECT":
            controller.transition_to("LOCATION_SELECT")

        else:
            controller.ui.show_error(
                f"Unsupported event '{event}' in state '{self.name}'"
            )