from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.controller.app_controller import AppController
from app.models.app_context import AppContext
from app.ui.test_ui_adapter import UIAdapter


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    context = AppContext(repo_root=repo_root)
    ui = UIAdapter()
    controller = AppController(context, ui)

    controller.start()

    while True:
        raw = input("\nEnter command: ").strip()

        if raw.lower() == "quit":
            break

        current_state = context.current_state

        if current_state == "START":
            if raw == "1":
                controller.dispatch("CREATE_NEW_AUDIT")
            elif raw == "2":
                ui.show_info("Edit Audit is not implemented yet.")
            else:
                print("Commands:")
                print("  1 -> Create New Audit")
                print("  2 -> Edit Audit")
                print("  quit -> Exit")

        elif current_state == "LOCATION_SELECT":
            if raw.isdigit():
                index = int(raw) - 1
                if 0 <= index < len(context.available_locations):
                    location_id = context.available_locations[index]["id"]
                    controller.dispatch(f"SELECT_LOCATION:{location_id}")
                else:
                    ui.show_error("Invalid location number.")
            else:
                print("Commands:")
                print("  1, 2, 3, ... -> Select location")
                print("  quit -> Exit")

        elif current_state == "ASSET_CATEGORY_SELECT":
            if raw.isdigit():
                index = int(raw) - 1
                if 0 <= index < len(context.available_asset_categories):
                    category_id = context.available_asset_categories[index]["id"]
                    controller.dispatch(f"SELECT_ASSET_CATEGORY:{category_id}")
                else:
                    ui.show_error("Invalid asset category number.")
            else:
                print("Commands:")
                print("  1, 2, 3, ... -> Select asset category")
                print("  quit -> Exit")

        elif current_state == "ITEM_MENU":
            if raw == "1":
                controller.dispatch("GO_TO_CAMERA_CAPTURE")
            elif raw == "2":
                controller.dispatch("BACK_TO_LOCATION_SELECT")
            elif raw == "3":
                controller.dispatch("BACK_TO_ASSET_CATEGORY_SELECT")
            else:
                print("Commands:")
                print("  1 -> Add New Item")
                print("  2 -> Back to Location Select")
                print("  3 -> Back to Asset Category Select")
                print("  quit -> Exit")

        elif current_state == "CAMERA_CAPTURE":
            if raw == "1":
                controller.dispatch("TAKE_PICTURE")
            elif raw == "2":
                controller.dispatch("CANCEL_CAMERA_CAPTURE")
            else:
                print("Commands:")
                print("  1 -> Take Picture")
                print("  2 -> Cancel")
                print("  quit -> Exit")

        elif current_state == "ADDITIONAL_IMAGES_MENU":
            if raw == "1":
                controller.dispatch("TAKE_ADDITIONAL_IMAGE")
            elif raw == "2":
                controller.dispatch("END_ITEM_IMAGES")
            else:
                print("Commands:")
                print("  1 -> Take New Image")
                print("  2 -> End")
                print("  quit -> Exit")

        elif current_state == "CAMERA_CAPTURE_ADDITIONAL":
            if raw == "1":
                controller.dispatch("TAKE_PICTURE")
            elif raw == "2":
                controller.dispatch("CANCEL_CAMERA_CAPTURE")
            else:
                print("Commands:")
                print("  1 -> Take Picture")
                print("  2 -> Cancel")
                print("  quit -> Exit")

        else:
            ui.show_error(f"Unknown current state: {current_state}")


if __name__ == "__main__":
    main()