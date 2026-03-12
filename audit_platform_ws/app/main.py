from __future__ import annotations

from pathlib import Path

from app.controller.app_controller import AppController
from app.models.app_context import AppContext
from app.ui.test_ui_adapter import UIAdapter


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    context = AppContext(repo_root=repo_root)
    ui = UIAdapter()
    controller = AppController(context, ui)

    controller.start()

    while True:
        raw = input("\nEnter command: ").strip()

        if raw == "quit":
            break
        elif raw == "1":
            controller.dispatch("CREATE_NEW_AUDIT")
        elif raw.startswith("loc "):
            location_id = raw.replace("loc ", "", 1).strip()
            controller.dispatch(f"SELECT_LOCATION:{location_id}")
        else:
            print("Commands:")
            print("  1           -> Create New Audit")
            print("  loc <id>    -> Select a location")
            print("  quit        -> Exit")


if __name__ == "__main__":
    main()