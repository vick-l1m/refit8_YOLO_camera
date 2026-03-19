"""
state.py

Defines the State class, which is the abstract base class for all application states.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.app_context import AppContext
# from app.ui.test_ui_adapter import UIAdapter
from app.ui.qt_ui_adapter import UIAdapter


class State(ABC):
    name: str = "STATE"

    @abstractmethod
    def on_enter(self, context: AppContext, ui: UIAdapter) -> None:
        pass

    @abstractmethod
    def handle_event(
        self,
        event: str,
        context: AppContext,
        controller: "AppController",
    ) -> None:
        pass