#!/usr/bin/env python3
# Metadata:
# File: audit_platform_ws/app/states/april_tag_measure_state.py
# Purpose: State for interactive AprilTag measurement inside the audit app.
# Date: 19/03/2026
# Version: 1.0
# Maintainer: Victor Lim - victor@polymaya.tech

"""
april_tag_measure_state.py

Defines the AprilTagMeasureState class, which represents the state where the
user measures lines on an image containing a detected AprilTag.
"""
from __future__ import annotations


from app.models.app_context import AppContext
from app.services.april_tag_service import AprilTagService
from app.states.state import State
from app.ui.qt_ui_adapter import UIAdapter


class AprilTagMeasureState(State):
    name = "APRIL_TAG_MEASURE"

    def on_enter(self, context: AppContext, ui: UIAdapter) -> None:
        context.current_state = self.name

        try:
            if context.apriltag_source_image_path is None:
                ui.show_error("No AprilTag source image is available.")
                return

            if context.apriltag_session is None:
                ui.show_error("No AprilTag measurement session is available.")
                return

            ui.show_april_tag_measure_screen(
                image_path=context.apriltag_source_image_path,
                session=context.apriltag_session,
            )

        except Exception as e:
            ui.show_error(str(e))

    def handle_event(
        self,
        event: str,
        context: AppContext,
        controller: "AppController",
    ) -> None:
        if event == "MEASURE_APRIL_TAG":
            try:
                if context.apriltag_session is None:
                    controller.ui.show_error("No AprilTag session is active.")
                    return

                service = AprilTagService(context.repo_root)
                export_result = service.save_measurement_outputs(
                    session=context.apriltag_session,
                    source_image_path=context.apriltag_source_image_path,
                    item_dir=context.current_item_dir,
                    item_base_image_name=context.current_item_base_image_name,
                    csv_path=context.apriltag_measurements_csv_path,
                )

                context.apriltag_annotated_image_path = export_result["annotated_image_path"]
                context.apriltag_measurements_csv_path = export_result["csv_path"]

                controller.ui.show_info(
                    "AprilTag measurements saved. Returning to additional images menu."
                )
                controller.transition_to("ADDITIONAL_IMAGES_MENU")

            except Exception as e:
                controller.ui.show_error(f"Failed to save AprilTag measurements: {e}")
            return

        if event == "BACK_FROM_APRIL_TAG_MEASURE":
            controller.ui.show_info("AprilTag measurement cancelled.")
            controller.transition_to("ADDITIONAL_IMAGES_MENU")
            return