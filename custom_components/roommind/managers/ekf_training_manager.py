"""EKF training accumulation and batching for RoomMind."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..const import EKF_UPDATE_MIN_DT
from ..control.gap_response import AIRFLOW_NORMAL

if TYPE_CHECKING:
    from ..control.thermal_model import RoomModelManager

_LOGGER = logging.getLogger(__name__)


class EkfTrainingManager:
    """Manages EKF training accumulation and batched updates per room."""

    def __init__(self, model_manager: RoomModelManager) -> None:
        self._model_manager = model_manager
        self._accumulated_dt: dict[str, float] = {}
        self._accumulated_mode: dict[str, str] = {}
        self._accumulated_airflow: dict[str, str] = {}
        self._accumulated_pf: dict[str, float] = {}
        self.last_temps: dict[str, float] = {}

    def set_model_manager(self, model_manager: RoomModelManager) -> None:
        """Update the model manager reference (used after full thermal reset)."""
        self._model_manager = model_manager

    def flush(
        self,
        area_id: str,
        current_temp: float,
        T_outdoor: float,
        can_heat: bool,
        can_cool: bool,
        q_solar: float,
        q_residual: float = 0.0,
        shading_factor: float = 1.0,
        q_occupancy: float = 0.0,
    ) -> None:
        """Flush accumulated EKF update (on mode/airflow change or window open)."""
        accumulated = self._accumulated_dt.pop(area_id, 0.0)
        prev_mode = self._accumulated_mode.pop(area_id, None)
        prev_airflow = self._accumulated_airflow.pop(area_id, AIRFLOW_NORMAL)
        pf = self._accumulated_pf.pop(area_id, 1.0)
        if accumulated > 0 and prev_mode is not None:
            self._model_manager.update(
                area_id,
                current_temp,
                T_outdoor,
                prev_mode,
                accumulated,
                can_heat=can_heat,
                can_cool=can_cool,
                power_fraction=pf,
                q_solar=q_solar * shading_factor,
                q_residual=q_residual,
                q_occupancy=q_occupancy,
                airflow=prev_airflow,
            )

    def process(
        self,
        area_id: str,
        current_temp: float,
        T_outdoor: float,
        ekf_mode: str | None,
        ekf_pf: float,
        window_open: bool,
        raw_open: bool,
        q_residual: float,
        shading_factor: float,
        q_solar: float,
        can_heat: bool,
        can_cool: bool,
        dt_minutes: float,
        q_occupancy: float = 0.0,
        airflow: str = AIRFLOW_NORMAL,
    ) -> None:
        """Process an EKF training step for a room.

        Contains the full training decision tree: window open, raw open
        (within delay), unobservable mode, or normal accumulation.
        """
        if window_open or raw_open:
            self.flush(
                area_id,
                current_temp,
                T_outdoor,
                can_heat,
                can_cool,
                q_solar,
                q_residual=q_residual,
                shading_factor=shading_factor,
                q_occupancy=q_occupancy,
            )
            self._accumulated_dt.pop(area_id, None)
            self._accumulated_mode.pop(area_id, None)
            self._accumulated_pf.pop(area_id, None)
            self._accumulated_airflow.pop(area_id, None)
            # Always track temperature state to prevent stale _x[0]
            # when normal learning resumes.  Only learn k_window when
            # the signal is clean (no residual heat).
            self._model_manager.update_window_open(
                area_id,
                current_temp,
                T_outdoor,
                dt_minutes,
                learn_k_window=(window_open and q_residual == 0.0),
            )
        elif ekf_mode is None:
            self.flush(
                area_id,
                current_temp,
                T_outdoor,
                can_heat,
                can_cool,
                q_solar,
                q_residual=q_residual,
                shading_factor=shading_factor,
                q_occupancy=q_occupancy,
            )
            self._accumulated_dt.pop(area_id, None)
            self._accumulated_mode.pop(area_id, None)
            self._accumulated_pf.pop(area_id, None)
            self._accumulated_airflow.pop(area_id, None)
        else:
            prev_mode = self._accumulated_mode.get(area_id)
            prev_airflow = self._accumulated_airflow.get(area_id)
            # An airflow change is the same kind of boundary as a mode change:
            # the interval either side was delivered by a different capacity,
            # and averaging across it trains both classes on neither.
            if (prev_mode is not None and prev_mode != ekf_mode) or (
                prev_airflow is not None and prev_airflow != airflow
            ):
                self.flush(
                    area_id,
                    current_temp,
                    T_outdoor,
                    can_heat,
                    can_cool,
                    q_solar,
                    q_residual=q_residual,
                    shading_factor=shading_factor,
                    q_occupancy=q_occupancy,
                )

            old_dt = self._accumulated_dt.get(area_id, 0.0)
            new_dt = old_dt + dt_minutes
            if new_dt > 0:
                old_pf = self._accumulated_pf.get(area_id, 1.0)
                self._accumulated_pf[area_id] = (old_pf * old_dt + ekf_pf * dt_minutes) / new_dt
            self._accumulated_dt[area_id] = new_dt
            self._accumulated_mode[area_id] = ekf_mode
            self._accumulated_airflow[area_id] = airflow

            if self._accumulated_dt[area_id] >= EKF_UPDATE_MIN_DT:
                pf = self._accumulated_pf.pop(area_id, 1.0)
                self._model_manager.update(
                    area_id,
                    current_temp,
                    T_outdoor,
                    ekf_mode,
                    self._accumulated_dt[area_id],
                    can_heat=can_heat,
                    can_cool=can_cool,
                    power_fraction=pf,
                    q_solar=q_solar * shading_factor,
                    q_residual=q_residual,
                    q_occupancy=q_occupancy,
                    airflow=airflow,
                )
                self._accumulated_dt[area_id] = 0.0

        self.last_temps[area_id] = current_temp

    def clear(self, area_id: str) -> None:
        """Clear accumulated EKF state for a room."""
        self._accumulated_dt.pop(area_id, None)
        self._accumulated_mode.pop(area_id, None)
        self._accumulated_pf.pop(area_id, None)

    def remove_room(self, area_id: str) -> None:
        """Clean up all state for a removed room."""
        self.clear(area_id)
        self.last_temps.pop(area_id, None)
