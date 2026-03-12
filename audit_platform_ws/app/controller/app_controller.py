"""
app_controller.py

Defines the AppController class, which manages the application state and handles events triggered by the UI.
"""
from __future__ import annotations

from app.controller.state_machine import StateMachine
from app.models.app_context import AppContext
from app.services.audit_service import AuditService

# States
from app.states.start_state import StartState
from app.states.location_select_state import LocationSelectState
from app.states.asset_category_select_state import AssetCategorySelectState
from app.states.item_menu_state import ItemMenuState
from app.states.camera_capture_state import CameraCaptureState
from app.states.image_review_state import ImageReviewState
from app.states.additional_images_menu_state import AdditionalImagesMenuState
from app.states.camera_capture_additional_state import CameraCaptureAdditionalState

# UI Adapter
from app.ui.test_ui_adapter import UIAdapter


class AppController:
    def __init__(self, context: AppContext, ui: UIAdapter) -> None:
        self.context = context
        self.ui = ui
        self.audit_service = AuditService()

        self.state_machine = StateMachine(
            states={
                "START": StartState(),
                "LOCATION_SELECT": LocationSelectState(),
                "ASSET_CATEGORY_SELECT": AssetCategorySelectState(),
                "ITEM_MENU": ItemMenuState(),
                "CAMERA_CAPTURE": CameraCaptureState(),
                "IMAGE_REVIEW": ImageReviewState(),        
                "ADDITIONAL_IMAGES_MENU": AdditionalImagesMenuState(),
                "CAMERA_CAPTURE_ADDITIONAL": CameraCaptureAdditionalState(),
            }
        )

    def start(self) -> None:
        self.state_machine.current_state.on_enter(self.context, self.ui)

    def dispatch(self, event: str) -> None:
        self.state_machine.current_state.handle_event(event, self.context, self)

    def transition_to(self, state_name: str) -> None:
        print(f"[DEBUG] Transitioning to {state_name}")
        self.state_machine.set_state(state_name)
        self.state_machine.current_state.on_enter(self.context, self.ui)

    def create_new_audit_and_go_to_location_select(self) -> None:
        self.context.current_audit = self.audit_service.create_new_audit()
        self.ui.show_info(f"Created audit session: {self.context.current_audit.audit_id}")
        self.transition_to("LOCATION_SELECT")