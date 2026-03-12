from __future__ import annotations

from app.controller.state_machine import StateMachine
from app.models.app_context import AppContext
from app.services.audit_service import AuditService
from app.states.location_select_state import LocationSelectState
from app.states.start_state import StartState
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
            }
        )

    def start(self) -> None:
        self.state_machine.current_state.on_enter(self.context, self.ui)

    def dispatch(self, event: str) -> None:
        self.state_machine.current_state.handle_event(event, self.context, self)

    def transition_to(self, state_name: str) -> None:
        self.state_machine.set_state(state_name)
        self.state_machine.current_state.on_enter(self.context, self.ui)

    def create_new_audit_and_go_to_location_select(self) -> None:
        self.context.current_audit = self.audit_service.create_new_audit()
        self.ui.show_info(f"Created audit session: {self.context.current_audit.audit_id}")
        self.transition_to("LOCATION_SELECT")