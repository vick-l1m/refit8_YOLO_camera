"""
state_machine.py

Defines the StateMachine class, which manages the current state of the application and handles state transitions.
"""

from __future__ import annotations

from typing import Dict

from app.states.state import State


class StateMachine:
    def __init__(self, states: Dict[str, State]) -> None:
        self.states = states
        self.current_state: State = states["START"]

    def set_state(self, state_name: str) -> None:
        if state_name not in self.states:
            raise ValueError(f"Unknown state: {state_name}")
        self.current_state = self.states[state_name]